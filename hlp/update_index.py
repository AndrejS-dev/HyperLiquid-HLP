"""
Fetch Hyperliquid HLP vault history and maintain a TVL-normalized index.

Native HL `vaultDetails` windows are sparse historically (~biweekly in allTime).
To get a stable denser research series we:

1. Keep HL account-value + cumulative PnL samples as hard anchors
   (allTime/month/week/day merged; day > week > month > allTime on collisions).
2. Pull DefiLlama daily HLP TVL (`hyperliquid-hlp`) from 2024-12-01 onward.
3. Densify onto a daily (and weekly) calendar by linearly interpolating HL
   cumulative PnL in time between anchors, then:
      period_return[t] = (pnl[t] - pnl[t-1]) / tvl[t-1]
   where tvl is Llama TVL when available, otherwise time-interpolated HL
   account value.

This preserves HL cumulative PnL at every anchor while giving stable daily/weekly
points. Intra-anchor PnL timing is an interpolation assumption (not true
observed daily fills).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

VAULT_ADDRESS = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"
API_URL = "https://api.hyperliquid.xyz/info"
LLAMA_PROTOCOL_URL = "https://api.llama.fi/protocol/hyperliquid-hlp"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "hlp_raw.csv"
INDEX_NATIVE_CSV = DATA_DIR / "hlp_index_native.csv"
INDEX_CSV = DATA_DIR / "hlp_index.csv"  # dense daily (main research series)
DAILY_INDEX_CSV = DATA_DIR / "hlp_daily_index.csv"
WEEKLY_INDEX_CSV = DATA_DIR / "hlp_weekly_index.csv"
LLAMA_TVL_CSV = DATA_DIR / "hlp_llama_tvl.csv"

WINDOW_PRIORITY = {"allTime": 1, "month": 2, "week": 3, "day": 4, "local": 0}


def fetch_vault_details(vault_address: str = VAULT_ADDRESS) -> dict[str, Any]:
    resp = requests.post(
        API_URL,
        json={"type": "vaultDetails", "vaultAddress": vault_address},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or "portfolio" not in data:
        raise RuntimeError(f"Unexpected vaultDetails response: {data!r}")
    return data


def fetch_defillama_tvl() -> pd.DataFrame:
    resp = requests.get(LLAMA_PROTOCOL_URL, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("tvl") or []
    if not rows:
        raise RuntimeError("DefiLlama hyperliquid-hlp returned no tvl history")
    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date"], unit="s", utc=True).dt.floor("D")
    df["tvl"] = pd.to_numeric(df["totalLiquidityUSD"], errors="coerce")
    df = (
        df.dropna(subset=["tvl"])
        .sort_values("date_utc")
        .drop_duplicates("date_utc", keep="last")
        .reset_index(drop=True)
    )
    return df[["date_utc", "tvl"]]


def _history_frame(series: dict[str, Any], source: str) -> pd.DataFrame:
    av = {int(t): float(v) for t, v in series.get("accountValueHistory", [])}
    pnl = {int(t): float(v) for t, v in series.get("pnlHistory", [])}
    timestamps = sorted(set(av) | set(pnl))
    rows = [
        {
            "timestamp_ms": ts,
            "account_value": av.get(ts),
            "window_pnl": pnl.get(ts),
            "source": source,
        }
        for ts in timestamps
    ]
    return pd.DataFrame(rows)


def calibrate_window_to_absolute(
    window_df: pd.DataFrame, all_time_df: pd.DataFrame
) -> pd.DataFrame:
    """Map a resetting window PnL series onto allTime absolute cumulative PnL."""
    if window_df.empty:
        return window_df.copy()

    out = window_df.copy()
    if out["source"].iloc[0] == "allTime":
        out["cumulative_pnl"] = out["window_pnl"]
        return out

    at = all_time_df.dropna(subset=["window_pnl"]).sort_values("timestamp_ms")
    if at.empty:
        raise RuntimeError("allTime series missing; cannot calibrate finer windows")

    common = set(out["timestamp_ms"]) & set(at["timestamp_ms"])
    if common:
        anchor_ts = max(common)
        abs_anchor = float(at.loc[at["timestamp_ms"] == anchor_ts, "window_pnl"].iloc[0])
        win_anchor = float(out.loc[out["timestamp_ms"] == anchor_ts, "window_pnl"].iloc[0])
    else:
        anchor_ts = int(out["timestamp_ms"].max())
        prior = at.loc[at["timestamp_ms"] <= anchor_ts]
        ref = prior if not prior.empty else at
        abs_anchor = float(ref.iloc[-1]["window_pnl"])
        win_anchor = float(out.loc[out["timestamp_ms"] == anchor_ts, "window_pnl"].iloc[0])

    out["cumulative_pnl"] = abs_anchor + (out["window_pnl"] - win_anchor)
    return out


def build_merged_raw(portfolio: list[list[Any]]) -> pd.DataFrame:
    port = {period: series for period, series in portfolio}
    if "allTime" not in port:
        raise RuntimeError("portfolio missing allTime window")

    all_time = _history_frame(port["allTime"], "allTime")
    all_time = calibrate_window_to_absolute(all_time, all_time)

    frames = [all_time]
    for source in ("month", "week", "day"):
        if source not in port:
            continue
        frame = _history_frame(port[source], source)
        frames.append(calibrate_window_to_absolute(frame, all_time))

    merged = pd.concat(frames, ignore_index=True)
    merged["priority"] = merged["source"].map(WINDOW_PRIORITY).fillna(0).astype(int)
    merged = merged.sort_values(["timestamp_ms", "priority"])
    merged = merged.drop_duplicates("timestamp_ms", keep="last")
    merged = merged.drop(columns=["window_pnl", "priority"])
    merged["datetime_utc"] = pd.to_datetime(merged["timestamp_ms"], unit="ms", utc=True)
    return merged.sort_values("timestamp_ms").reset_index(drop=True)


def merge_with_existing(new_raw: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    if not existing_path.exists():
        return new_raw

    old = pd.read_csv(existing_path)
    if old.empty:
        return new_raw

    old["timestamp_ms"] = old["timestamp_ms"].astype("int64")
    for col in ("account_value", "cumulative_pnl"):
        if col in old.columns:
            old[col] = pd.to_numeric(old[col], errors="coerce")
    if "source" not in old.columns:
        old["source"] = "local"

    combined = pd.concat([old, new_raw], ignore_index=True, sort=False)
    combined["priority"] = combined["source"].map(WINDOW_PRIORITY).fillna(0).astype(int)
    combined["_order"] = range(len(combined))
    combined = combined.sort_values(["timestamp_ms", "priority", "_order"])
    combined = combined.drop_duplicates("timestamp_ms", keep="last")
    combined = combined.drop(columns=["priority", "_order"], errors="ignore")
    combined["datetime_utc"] = pd.to_datetime(
        combined["timestamp_ms"], unit="ms", utc=True
    )
    return combined.sort_values("timestamp_ms").reset_index(drop=True)


def build_normalized_index(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna(subset=["account_value", "cumulative_pnl"]).copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    positive = df["account_value"] > 0
    if not positive.any():
        raise RuntimeError("No positive account_value observations in series")
    df = df.loc[positive.idxmax() :].reset_index(drop=True)

    prev_av = df["account_value"].shift(1)
    delta_pnl = df["cumulative_pnl"] - df["cumulative_pnl"].shift(1)
    period_return = delta_pnl / prev_av
    period_return.iloc[0] = 0.0
    delta_pnl.iloc[0] = 0.0
    period_return = period_return.mask(prev_av.notna() & (prev_av <= 0), 0.0)

    index = (1.0 + period_return.fillna(0.0)).cumprod()
    index.iloc[0] = 1.0

    return pd.DataFrame(
        {
            "timestamp_ms": df["timestamp_ms"].astype("int64"),
            "datetime_utc": pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True),
            "account_value": df["account_value"],
            "cumulative_pnl": df["cumulative_pnl"],
            "delta_pnl": delta_pnl,
            "period_return": period_return,
            "index": index,
            "source": df.get("source", "unknown"),
        }
    )


def densify_calendar_series(
    anchors: pd.DataFrame,
    llama_tvl: pd.DataFrame | None,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Build a regular calendar series from HL anchors.

    Cumulative PnL is linearly interpolated in time between anchors (so every
    HL sample is matched exactly). TVL denominator prefers DefiLlama daily TVL
    when present, else time-interpolated HL account value.
    """
    df = anchors.dropna(subset=["account_value", "cumulative_pnl"]).copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df = df.loc[df["account_value"] > 0].reset_index(drop=True)
    if len(df) < 2:
        raise RuntimeError("Need at least 2 positive HL anchors to densify")

    start = df["datetime_utc"].iloc[0].floor("D")
    end = df["datetime_utc"].iloc[-1].floor("D")
    if freq.upper().startswith("W"):
        # Week-ending Sunday grid covering the span.
        week_ends = pd.date_range(start, end, freq="W-SUN", tz="UTC")
        days = pd.DatetimeIndex([start, *week_ends, end]).unique().sort_values()
    else:
        days = pd.date_range(start, end, freq="D", tz="UTC")

    # Evaluate the daily/weekly point at end-of-day UTC so same-day anchors count.
    eod = days + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    grid_ts = np.asarray([ts.timestamp() * 1000.0 for ts in eod], dtype=float)

    anchor_ts = df["timestamp_ms"].astype(float).to_numpy()
    anchor_pnl = df["cumulative_pnl"].astype(float).to_numpy()
    anchor_av = df["account_value"].astype(float).to_numpy()

    cum_pnl = np.interp(grid_ts, anchor_ts, anchor_pnl)
    av_interp = np.interp(grid_ts, anchor_ts, anchor_av)

    llama_map = None
    if llama_tvl is not None and not llama_tvl.empty:
        llama_map = llama_tvl.set_index("date_utc")["tvl"]

    tvl = []
    tvl_source = []
    for i, day in enumerate(days):
        if llama_map is not None and day in llama_map.index:
            # Prefer observed daily TVL from DefiLlama whenever available.
            tvl.append(float(llama_map.loc[day]))
            tvl_source.append("defillama")
        else:
            tvl.append(float(av_interp[i]))
            tvl_source.append("hl_interp")
    tvl_arr = np.asarray(tvl, dtype=float)

    delta_pnl = np.diff(cum_pnl, prepend=cum_pnl[0])
    delta_pnl[0] = 0.0
    prev_tvl = np.roll(tvl_arr, 1)
    period_return = np.zeros_like(delta_pnl)
    valid = prev_tvl > 0
    valid[0] = False
    period_return[valid] = delta_pnl[valid] / prev_tvl[valid]

    index = np.cumprod(1.0 + period_return)
    index[0] = 1.0

    out = pd.DataFrame(
        {
            "date_utc": days.strftime("%Y-%m-%d"),
            "timestamp_ms": grid_ts.astype("int64"),
            "datetime_utc": days,
            "account_value": tvl_arr,
            "cumulative_pnl": cum_pnl,
            "delta_pnl": delta_pnl,
            "period_return": period_return,
            "index": index,
            "source": [
                f"dense_{freq.lower()}_{src}" for src in tvl_source
            ],
            "tvl_source": tvl_source,
        }
    )
    # Keep a conventional return column name for daily consumers.
    if freq.upper().startswith("D"):
        out = out.rename(columns={"period_return": "daily_return"})
        # also keep period_return alias for shared analyzers
        out["period_return"] = out["daily_return"]
    else:
        out["daily_return"] = out["period_return"]
    return out


def update(vault_address: str = VAULT_ADDRESS) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    details = fetch_vault_details(vault_address)
    new_raw = build_merged_raw(details["portfolio"])
    raw = merge_with_existing(new_raw, RAW_CSV)

    llama = fetch_defillama_tvl()
    llama.to_csv(LLAMA_TVL_CSV, index=False)

    native_index = build_normalized_index(raw)
    daily_dense = densify_calendar_series(raw, llama, freq="D")

    # Weekly = last daily dense point each W-SUN period; keep the daily index level
    # so daily/weekly charts agree on overlapping dates.
    week_key = daily_dense["datetime_utc"].dt.tz_convert("UTC").dt.strftime("%G-W%V")
    weekly_dense = (
        daily_dense.sort_values("datetime_utc")
        .groupby(week_key, sort=True)
        .tail(1)
        .reset_index(drop=True)
        .copy()
    )
    # Ensure the series origin is present so weekly starts at index 1.0.
    if weekly_dense["datetime_utc"].iloc[0] != daily_dense["datetime_utc"].iloc[0]:
        weekly_dense = pd.concat(
            [daily_dense.iloc[[0]], weekly_dense], ignore_index=True
        )
        weekly_dense = weekly_dense.drop_duplicates("datetime_utc", keep="last")
        weekly_dense = weekly_dense.sort_values("datetime_utc").reset_index(drop=True)

    weekly_dense["delta_pnl"] = weekly_dense["cumulative_pnl"] - weekly_dense[
        "cumulative_pnl"
    ].shift(1)
    weekly_dense["period_return"] = weekly_dense["index"] / weekly_dense["index"].shift(
        1
    ) - 1.0
    weekly_dense.loc[weekly_dense.index[0], "delta_pnl"] = 0.0
    weekly_dense.loc[weekly_dense.index[0], "period_return"] = 0.0
    weekly_dense["daily_return"] = weekly_dense["period_return"]
    weekly_dense["source"] = "dense_w_from_daily"

    # Main research index = dense daily.
    main_index = daily_dense[
        [
            "timestamp_ms",
            "datetime_utc",
            "account_value",
            "cumulative_pnl",
            "delta_pnl",
            "period_return",
            "index",
            "source",
        ]
    ].copy()

    raw_out = raw[
        ["timestamp_ms", "datetime_utc", "account_value", "cumulative_pnl", "source"]
    ].copy()
    raw_out.to_csv(RAW_CSV, index=False)
    native_index.to_csv(INDEX_NATIVE_CSV, index=False)
    main_index.to_csv(INDEX_CSV, index=False)
    daily_dense.to_csv(DAILY_INDEX_CSV, index=False)
    weekly_dense.to_csv(WEEKLY_INDEX_CSV, index=False)

    return {
        "name": details.get("name"),
        "vault": vault_address,
        "apr_api": details.get("apr"),
        "raw_anchor_points": len(raw_out),
        "native_index_points": len(native_index),
        "daily_dense_points": len(daily_dense),
        "weekly_dense_points": len(weekly_dense),
        "llama_tvl_points": len(llama),
        "index_start": float(main_index["index"].iloc[0]),
        "index_end": float(main_index["index"].iloc[-1]),
        "weekly_index_end": float(weekly_dense["index"].iloc[-1]),
        "first_utc": str(main_index["datetime_utc"].iloc[0]),
        "last_utc": str(main_index["datetime_utc"].iloc[-1]),
        "tvl_last": float(main_index["account_value"].iloc[-1]),
        "methodology": (
            "HL PnL anchors + time interpolation; DefiLlama daily TVL denominator "
            "from 2024-12-01 when available, else interpolated HL account value"
        ),
        "paths": {
            "raw": str(RAW_CSV),
            "index_daily_main": str(INDEX_CSV),
            "index_native": str(INDEX_NATIVE_CSV),
            "daily_index": str(DAILY_INDEX_CSV),
            "weekly_index": str(WEEKLY_INDEX_CSV),
            "llama_tvl": str(LLAMA_TVL_CSV),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Update HLP TVL-normalized index CSVs")
    parser.add_argument("--vault", default=VAULT_ADDRESS, help="Vault address (default: HLP)")
    args = parser.parse_args()
    summary = update(args.vault)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
