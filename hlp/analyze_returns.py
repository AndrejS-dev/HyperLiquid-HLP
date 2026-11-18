"""
HLP TVL-normalized index analysis + Streamlit dashboard.

CLI:
  python hlp/analyze_returns.py
  python hlp/analyze_returns.py --json

Dashboard:
  streamlit run hlp/analyze_returns.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_CSV = DATA_DIR / "hlp_index.csv"  # dense daily main series
DAILY_INDEX_CSV = DATA_DIR / "hlp_daily_index.csv"
WEEKLY_INDEX_CSV = DATA_DIR / "hlp_weekly_index.csv"
INDEX_NATIVE_CSV = DATA_DIR / "hlp_index_native.csv"
RAW_CSV = DATA_DIR / "hlp_raw.csv"
STATS_CSV = DATA_DIR / "hlp_stats.csv"
VAULT_ADDRESS = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"
VAULT_URL = f"https://app.hyperliquid.xyz/vaults/{VAULT_ADDRESS}"


def is_hosted_streamlit() -> bool:
    """True on Streamlit Community Cloud / similar hosted runtimes."""
    if os.environ.get("HLP_HOSTED", "").strip().lower() in {"1", "true", "yes"}:
        return True
    if os.environ.get("STREAMLIT_RUNTIME_ENV", "").strip().lower() == "cloud":
        return True
    # Common Streamlit Cloud mount paths.
    return Path("/mount/src").exists() or Path("/app").exists() and Path("/app/hlp").exists()


def allow_live_fetch() -> bool:
    """
    Live API refresh writes local CSVs. On hosted Streamlit those writes are
    ephemeral, so default off there. Override with secret/env HLP_ALLOW_LIVE_FETCH=1.
    """
    flag = os.environ.get("HLP_ALLOW_LIVE_FETCH", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    return not is_hosted_streamlit()


def data_as_of_label() -> str:
    paths = [
        p
        for p in (DAILY_INDEX_CSV, INDEX_CSV, WEEKLY_INDEX_CSV, INDEX_NATIVE_CSV, RAW_CSV)
        if p.exists()
    ]
    if not paths:
        return "n/a"
    latest = max(p.stat().st_mtime for p in paths)
    return datetime.fromtimestamp(latest, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# Chart theme: ink / copper (avoid default purple Streamlit look)
COLORS = {
    "bg": "#0f1419",
    "paper": "#151b22",
    "grid": "#243041",
    "text": "#e7ecf3",
    "muted": "#8b9bb0",
    "index": "#f0a45d",
    "tvl": "#5ec4c8",
    "drawdown": "#e07a6a",
    "positive": "#6fbf8a",
    "negative": "#e07a6a",
    "zero": "#5a6a7d",
}


def _max_drawdown(index: pd.Series) -> float:
    peak = index.cummax()
    dd = index / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def _drawdown_series(index: pd.Series) -> pd.Series:
    peak = index.cummax()
    return index / peak - 1.0


def _annualization_factor(dates: pd.Series) -> float:
    if len(dates) < 2:
        return float("nan")
    span_days = (dates.iloc[-1] - dates.iloc[0]).total_seconds() / 86400.0
    if span_days <= 0:
        return float("nan")
    return (len(dates) - 1) / span_days * 365.25


def compute_stats(df: pd.DataFrame, return_col: str, freq_label: str) -> dict:
    work = df.copy()
    if "datetime_utc" in work.columns:
        work["datetime_utc"] = pd.to_datetime(work["datetime_utc"], utc=True)
    elif "date_utc" in work.columns:
        work["datetime_utc"] = pd.to_datetime(work["date_utc"], utc=True)
    else:
        work["datetime_utc"] = pd.to_datetime(work["timestamp_ms"], unit="ms", utc=True)

    work = work.sort_values("datetime_utc").reset_index(drop=True)
    rets = work[return_col].astype(float)
    rets_ex_first = rets.iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()

    start = work["datetime_utc"].iloc[0]
    end = work["datetime_utc"].iloc[-1]
    years = (end - start).total_seconds() / (365.25 * 86400.0)
    total_return = float(work["index"].iloc[-1] / work["index"].iloc[0] - 1.0)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else float("nan")

    ann_factor = _annualization_factor(work["datetime_utc"])
    vol = float(rets_ex_first.std(ddof=1)) if len(rets_ex_first) > 1 else float("nan")
    vol_ann = vol * np.sqrt(ann_factor) if np.isfinite(vol) and np.isfinite(ann_factor) else float("nan")
    mean = float(rets_ex_first.mean()) if len(rets_ex_first) else float("nan")
    mean_ann = (
        (1.0 + mean) ** ann_factor - 1.0
        if np.isfinite(mean) and np.isfinite(ann_factor)
        else float("nan")
    )
    median = float(rets_ex_first.median()) if len(rets_ex_first) else float("nan")
    # CAGR if every period earned the median return (same compounding as mean_ann).
    median_ann = (
        (1.0 + median) ** ann_factor - 1.0
        if np.isfinite(median) and np.isfinite(ann_factor)
        else float("nan")
    )

    downside = rets_ex_first[rets_ex_first < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    downside_ann = (
        downside_std * np.sqrt(ann_factor)
        if np.isfinite(downside_std) and np.isfinite(ann_factor)
        else float("nan")
    )

    sharpe = mean_ann / vol_ann if np.isfinite(mean_ann) and vol_ann and vol_ann > 0 else float("nan")
    sortino = (
        mean_ann / downside_ann
        if np.isfinite(mean_ann) and downside_ann and downside_ann > 0
        else float("nan")
    )

    return {
        "freq": freq_label,
        "n_obs": int(len(work)),
        "n_returns": int(len(rets_ex_first)),
        "start_utc": str(start),
        "end_utc": str(end),
        "years": years,
        "index_start": float(work["index"].iloc[0]),
        "index_end": float(work["index"].iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "mean_period_return": mean,
        "median_period_return": median,
        "median_ann": median_ann,
        "vol_period": vol,
        "vol_ann": vol_ann,
        "mean_ann": mean_ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "skew": float(rets_ex_first.skew()) if len(rets_ex_first) > 2 else float("nan"),
        "kurtosis": float(rets_ex_first.kurtosis()) if len(rets_ex_first) > 3 else float("nan"),
        "best_period_return": float(rets_ex_first.max()) if len(rets_ex_first) else float("nan"),
        "worst_period_return": float(rets_ex_first.min()) if len(rets_ex_first) else float("nan"),
        "pct_positive_periods": float((rets_ex_first > 0).mean()) if len(rets_ex_first) else float("nan"),
        "max_drawdown": _max_drawdown(work["index"].astype(float)),
        "tvl_last": float(work["account_value"].iloc[-1]),
        "ann_factor_used": ann_factor,
    }


def rolling_stats(daily: pd.DataFrame, windows: tuple[int, ...] = (7, 30, 90, 365)) -> pd.DataFrame:
    d = daily.copy()
    d["datetime_utc"] = pd.to_datetime(d["datetime_utc"], utc=True)
    d = d.sort_values("datetime_utc").reset_index(drop=True)
    end = d["datetime_utc"].iloc[-1]
    rows = []
    for w in windows:
        start = end - pd.Timedelta(days=w)
        prior = d.loc[d["datetime_utc"] <= start]
        inside = d.loc[d["datetime_utc"] > start]
        if inside.empty:
            continue
        if not prior.empty:
            tail = pd.concat([prior.iloc[[-1]], inside], ignore_index=True)
        else:
            tail = inside.copy()
        if len(tail) < 2:
            continue

        prev_av = tail["account_value"].shift(1)
        delta_pnl = tail["cumulative_pnl"] - tail["cumulative_pnl"].shift(1)
        rets = delta_pnl / prev_av
        rets.iloc[0] = 0.0
        rets = rets.mask(prev_av.notna() & (prev_av <= 0), 0.0).fillna(0.0)
        idx = (1.0 + rets).cumprod()
        idx.iloc[0] = 1.0
        window_df = tail.copy()
        window_df["daily_return"] = rets
        window_df["index"] = idx
        s = compute_stats(window_df, "daily_return", f"daily_last_{w}d")
        s["window_days"] = w
        s["n_calendar_days"] = w
        rows.append(s)
    return pd.DataFrame(rows)


def load_index_frame(path: Path = INDEX_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    if "period_return" not in df.columns and "daily_return" in df.columns:
        df["period_return"] = df["daily_return"]
    if "daily_return" not in df.columns and "period_return" in df.columns:
        df["daily_return"] = df["period_return"]
    return df.sort_values("datetime_utc").reset_index(drop=True)


def slice_and_rebase(
    df: pd.DataFrame,
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep observations on/after start and rebase index to 1.0 at the new origin."""
    out = df.copy()
    out["datetime_utc"] = pd.to_datetime(out["datetime_utc"], utc=True)
    out = out.sort_values("datetime_utc").reset_index(drop=True)
    if out.empty:
        return out

    if start is not None:
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        out = out.loc[out["datetime_utc"] >= start_ts].reset_index(drop=True)
        if out.empty:
            return out

    base = float(out["index"].iloc[0])
    if base != 0.0 and np.isfinite(base):
        out["index"] = out["index"].astype(float) / base
    out.loc[out.index[0], "index"] = 1.0

    for col in ("period_return", "daily_return", "delta_pnl"):
        if col in out.columns:
            out.loc[out.index[0], col] = 0.0
    return out


def cap_upper_returns(
    df: pd.DataFrame,
    return_col: str,
    top_pct: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Cap the top `top_pct`% of period returns at the highest remaining return
    (upper winsorization), then rebuild the equity index from 1.0.

    Example: top_pct=1 caps every return above the 99th percentile down to that
    99th-percentile value (the best return still inside the kept set).
    """
    out = df.copy().reset_index(drop=True)
    if out.empty or len(out) < 3 or top_pct <= 0:
        meta = {
            "enabled": False,
            "top_pct": float(top_pct),
            "cap_return": float("nan"),
            "n_capped": 0,
            "n_returns": max(0, len(out) - 1),
        }
        return out, meta

    raw = out[return_col].astype(float).copy()
    sample = raw.iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()
    if sample.empty:
        meta = {
            "enabled": False,
            "top_pct": float(top_pct),
            "cap_return": float("nan"),
            "n_capped": 0,
            "n_returns": 0,
        }
        return out, meta

    q = max(0.0, min(100.0, float(top_pct)))
    # Cap level = highest return remaining after cutting the top q%.
    cap_level = float(sample.quantile(1.0 - q / 100.0))
    capped = raw.clip(upper=cap_level)
    capped.iloc[0] = 0.0
    n_capped = int((raw.iloc[1:] > cap_level).sum())

    out[return_col] = capped
    if return_col != "period_return" and "period_return" in out.columns:
        out["period_return"] = capped
    if return_col != "daily_return" and "daily_return" in out.columns:
        out["daily_return"] = capped
    if "period_return" not in out.columns:
        out["period_return"] = capped
    if "daily_return" not in out.columns:
        out["daily_return"] = capped

    prev_av = out["account_value"].shift(1)
    out["delta_pnl"] = capped * prev_av
    out.loc[out.index[0], "delta_pnl"] = 0.0

    rebuilt = (1.0 + capped.fillna(0.0)).cumprod()
    rebuilt.iloc[0] = 1.0
    out["index"] = rebuilt
    out["return_capped"] = raw > cap_level
    out.loc[out.index[0], "return_capped"] = False

    meta = {
        "enabled": True,
        "top_pct": q,
        "cap_return": cap_level,
        "n_capped": n_capped,
        "n_returns": int(len(sample)),
        "pct_capped": float(n_capped / len(sample)) if len(sample) else float("nan"),
    }
    return out, meta


def load_daily_frame(path: Path = DAILY_INDEX_CSV) -> pd.DataFrame:
    return load_index_frame(path)


def analyze(*, write_stats: bool | None = None) -> dict:
    if not INDEX_CSV.exists() and not DAILY_INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing {INDEX_CSV}. Run update_index.py first.")

    if write_stats is None:
        # Avoid pointless CSV writes on hosted read-only dashboards.
        write_stats = not is_hosted_streamlit()

    result: dict = {}
    tables: list[pd.DataFrame] = []

    daily_path = DAILY_INDEX_CSV if DAILY_INDEX_CSV.exists() else INDEX_CSV
    daily = load_daily_frame(daily_path)
    daily_stats = compute_stats(daily, "daily_return", "daily_dense")
    result["daily"] = daily_stats
    tables.append(pd.DataFrame([{"section": "daily_dense", **daily_stats}]))

    if WEEKLY_INDEX_CSV.exists():
        weekly = load_index_frame(WEEKLY_INDEX_CSV)
        weekly_stats = compute_stats(weekly, "period_return", "weekly_dense")
        result["weekly"] = weekly_stats
        tables.append(pd.DataFrame([{"section": "weekly_dense", **weekly_stats}]))

    native_path = INDEX_NATIVE_CSV if INDEX_NATIVE_CSV.exists() else None
    if native_path is not None:
        native = load_index_frame(native_path)
        native_stats = compute_stats(native, "period_return", "native_samples")
        result["native_samples"] = native_stats
        tables.append(pd.DataFrame([{"section": "native_samples", **native_stats}]))

    roll = rolling_stats(daily)
    if not roll.empty:
        result["rolling_daily_windows"] = roll.to_dict(orient="records")
        roll_out = roll.copy()
        roll_out.insert(0, "section", "rolling")
        tables.append(roll_out)

    stats_df = pd.concat(tables, ignore_index=True, sort=False)
    if write_stats:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(STATS_CSV, index=False)
        result["stats_csv"] = str(STATS_CSV)
    return result


def _pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}%" if x == x else "n/a"


def _fmt_num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if x == x else "n/a"


def _pp_delta(curr: float, base: float, digits: int = 2) -> str | None:
    if curr != curr or base != base:
        return None
    return f"{100.0 * (curr - base):+.{digits}f} pp vs raw"


def _num_delta(curr: float, base: float, digits: int = 2) -> str | None:
    if curr != curr or base != base:
        return None
    return f"{curr - base:+.{digits}f} vs raw"


def seasonality_stats(
    df: pd.DataFrame,
    return_col: str,
    grain: str,
) -> pd.DataFrame:
    """
    Windowed performance by calendar grain.

    grain: year | quarter | month | week
    Each window rebuilds a local index from period returns, then computes
    total return / CAGR / Sharpe / Sortino / MaxDD.
    """
    work = df.copy()
    work["datetime_utc"] = pd.to_datetime(work["datetime_utc"], utc=True)
    work = work.sort_values("datetime_utc").reset_index(drop=True)
    if work.empty or return_col not in work.columns:
        return pd.DataFrame()

    grain = grain.lower()
    if grain == "year":
        work["_bucket"] = work["datetime_utc"].dt.strftime("%Y")
        label = "Year"
    elif grain == "quarter":
        # Avoid tz warning from Period conversion on UTC timestamps.
        work["_bucket"] = (
            work["datetime_utc"].dt.tz_convert("UTC").dt.strftime("%Y")
            + "-Q"
            + work["datetime_utc"].dt.quarter.astype(str)
        )
        label = "Quarter"
    elif grain == "month":
        work["_bucket"] = work["datetime_utc"].dt.strftime("%Y-%m")
        label = "Month"
    elif grain == "week":
        work["_bucket"] = work["datetime_utc"].dt.strftime("%G-W%V")
        label = "Week"
    else:
        raise ValueError(f"Unknown seasonality grain: {grain}")

    rows: list[dict] = []
    for bucket, group in work.groupby("_bucket", sort=True):
        g = group.reset_index(drop=True)
        if len(g) < 2:
            continue
        rets = g[return_col].astype(float).copy()
        rets.iloc[0] = 0.0
        idx = (1.0 + rets.fillna(0.0)).cumprod()
        idx.iloc[0] = 1.0
        window = g.copy()
        window[return_col] = rets
        window["period_return"] = rets
        window["daily_return"] = rets
        window["index"] = idx
        s = compute_stats(window, return_col, f"{grain}_{bucket}")
        rows.append(
            {
                "grain": label,
                "window": str(bucket),
                "n_obs": s["n_obs"],
                "total_return": s["total_return"],
                "cagr": s["cagr"],
                "sharpe": s["sharpe"],
                "sortino": s["sortino"],
                "max_drawdown": s["max_drawdown"],
                "vol_ann": s["vol_ann"],
                "hit_rate": s["pct_positive_periods"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    metric_cols = [
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "vol_ann",
        "hit_rate",
    ]
    avg = {
        "grain": label,
        "window": "AVG",
        "n_obs": int(out["n_obs"].mean()) if len(out) else 0,
        **{col: float(out[col].mean()) for col in metric_cols},
    }
    med = {
        "grain": label,
        "window": "MEDIAN",
        "n_obs": int(out["n_obs"].median()) if len(out) else 0,
        **{col: float(out[col].median()) for col in metric_cols},
    }
    return pd.concat([out, pd.DataFrame([avg, med])], ignore_index=True)


def _plotly_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="IBM Plex Sans, Segoe UI, sans-serif"),
        margin=dict(l=48, r=24, t=56, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False, showspikes=True),
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
    )
    base.update(kwargs)
    return base


def plot_index_overview(
    df: pd.DataFrame,
    title: str = "HLP normalized index",
    index_scale: str = "Linear",
    tvl_scale: str = "Linear",
    compare_df: pd.DataFrame | None = None,
    compare_name: str = "Uncapped",
) -> go.Figure:
    work = df.copy()
    work["drawdown"] = _drawdown_series(work["index"].astype(float))
    index_log = str(index_scale).lower().startswith("log")
    tvl_log = str(tvl_scale).lower().startswith("log")
    show_legend = compare_df is not None and not compare_df.empty
    index_name = "Capped index" if show_legend else "Index"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(
            f"Index · {('log' if index_log else 'linear')}",
            "Drawdown",
            f"TVL · {('log' if tvl_log else 'linear')}",
        ),
    )

    if show_legend:
        fig.add_trace(
            go.Scatter(
                x=compare_df["datetime_utc"],
                y=compare_df["index"],
                name=compare_name,
                mode="lines",
                line=dict(color=COLORS["muted"], width=1.7, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>"
                + compare_name
                + ": %{y:.4f}<extra></extra>",
                legendgroup="index",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=work["datetime_utc"],
            y=work["index"],
            name=index_name,
            mode="lines",
            line=dict(color=COLORS["index"], width=2.3),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>"
            + index_name
            + ": %{y:.4f}<extra></extra>",
            legendgroup="index",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color=COLORS["zero"],
        line_width=1,
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=work["datetime_utc"],
            y=work["drawdown"] * 100.0,
            name="Drawdown",
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(224,122,106,0.18)",
            line=dict(color=COLORS["drawdown"], width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=work["datetime_utc"],
            y=work["account_value"],
            name="TVL",
            mode="lines",
            line=dict(color=COLORS["tvl"], width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>TVL: $%{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        **_plotly_layout(
            title=dict(
                text=title,
                x=0.0,
                xref="paper",
                y=0.995,
                yanchor="top",
                font=dict(size=15, color=COLORS["text"]),
            ),
            height=840 if show_legend else 800,
            margin=dict(l=56, r=28, t=96 if show_legend else 72, b=48),
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0.0,
                xanchor="left",
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(size=12, color=COLORS["muted"]),
                itemsizing="constant",
                traceorder="normal",
            ),
            hoverlabel=dict(
                bgcolor=COLORS["paper"],
                bordercolor=COLORS["grid"],
                font=dict(color=COLORS["text"], size=12),
            ),
        )
    )
    # Soften subplot titles so they don't compete with the main title/legend.
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color=COLORS["muted"])
        annotation.xanchor = "left"
        annotation.x = 0.0
        if annotation.text.startswith("Index"):
            annotation.y = annotation.y + 0.01

    fig.update_yaxes(
        title_text="Index",
        row=1,
        col=1,
        gridcolor=COLORS["grid"],
        zeroline=False,
        type="log" if index_log else "linear",
        title_font=dict(size=11, color=COLORS["muted"]),
        tickfont=dict(size=11, color=COLORS["muted"]),
    )
    fig.update_yaxes(
        title_text="%",
        row=2,
        col=1,
        gridcolor=COLORS["grid"],
        zeroline=False,
        type="linear",
        title_font=dict(size=11, color=COLORS["muted"]),
        tickfont=dict(size=11, color=COLORS["muted"]),
    )
    fig.update_yaxes(
        title_text="USD",
        row=3,
        col=1,
        gridcolor=COLORS["grid"],
        zeroline=False,
        tickformat="~s",
        type="log" if tvl_log else "linear",
        title_font=dict(size=11, color=COLORS["muted"]),
        tickfont=dict(size=11, color=COLORS["muted"]),
    )
    fig.update_xaxes(
        gridcolor=COLORS["grid"],
        tickfont=dict(size=11, color=COLORS["muted"]),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor=COLORS["grid"],
    )
    return fig


def plot_returns_histogram(df: pd.DataFrame, return_col: str, title: str) -> go.Figure:
    rets = df[return_col].astype(float).iloc[1:].replace([np.inf, -np.inf], np.nan).dropna() * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=rets,
            nbinsx=40,
            name="Returns",
            marker=dict(color=COLORS["index"], line=dict(color=COLORS["bg"], width=0.5)),
            hovertemplate="Return bin<br>count=%{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["zero"])
    fig.update_layout(
        **_plotly_layout(
            title=dict(text=title, x=0.01, font=dict(size=16)),
            height=360,
            xaxis_title="Period return (%)",
            yaxis_title="Count",
            bargap=0.05,
        )
    )
    return fig


def plot_rolling_comparison(roll: pd.DataFrame) -> go.Figure:
    if roll.empty:
        fig = go.Figure()
        fig.update_layout(**_plotly_layout(title="Rolling windows", height=320))
        return fig

    labels = [f"{int(w)}d" for w in roll["window_days"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=roll["total_return"] * 100.0,
            name="Total return",
            marker_color=COLORS["index"],
            hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=roll["vol_ann"] * 100.0,
            name="Ann. vol",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=COLORS["tvl"], width=2),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Vol: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_plotly_layout(
            title=dict(text="Rolling window return vs annualized vol", x=0.01, font=dict(size=16)),
            height=380,
            yaxis=dict(title="Return (%)", gridcolor=COLORS["grid"]),
            yaxis2=dict(
                title="Vol (%)",
                overlaying="y",
                side="right",
                showgrid=False,
                color=COLORS["tvl"],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
    )
    return fig


def print_report(result: dict) -> None:
    for key in ("daily", "weekly", "native_samples"):
        if key not in result:
            continue
        s = result[key]
        print(f"\n=== {key} ===")
        print(f"span: {s['start_utc']} -> {s['end_utc']} ({s['years']:.2f}y)")
        print(f"points: {s['n_obs']}  returns: {s['n_returns']}")
        print(f"index: {s['index_start']:.6f} -> {s['index_end']:.6f}")
        print(f"total return: {_pct(s['total_return'])}   CAGR: {_pct(s['cagr'])}")
        if s["sharpe"] == s["sharpe"]:
            print(f"vol (ann): {_pct(s['vol_ann'])}   Sharpe: {s['sharpe']:.3f}")
        else:
            print(f"vol (ann): {_pct(s['vol_ann'])}")
        print(f"Sortino: {s['sortino']:.3f}" if s["sortino"] == s["sortino"] else "Sortino: n/a")
        print(f"max drawdown: {_pct(s['max_drawdown'])}")
        print(
            f"best/worst period: {_pct(s['best_period_return'])} / {_pct(s['worst_period_return'])}"
        )
        print(f"share positive periods: {_pct(s['pct_positive_periods'])}")
        print(
            f"median period: {_pct(s['median_period_return'], 4)}   "
            f"median CAGR: {_pct(s['median_ann'])}"
        )
        print(f"TVL last: {s['tvl_last']:,.2f}")

    if result.get("rolling_daily_windows"):
        print("\n=== rolling daily windows ===")
        for s in result["rolling_daily_windows"]:
            if s["sharpe"] == s["sharpe"]:
                print(
                    f"{s['window_days']:>4}d  ret={_pct(s['total_return'])}  "
                    f"vol={_pct(s['vol_ann'])}  sharpe={s['sharpe']:.2f}  "
                    f"mdd={_pct(s['max_drawdown'])}"
                )
            else:
                print(
                    f"{s['window_days']:>4}d  ret={_pct(s['total_return'])}  "
                    f"vol={_pct(s['vol_ann'])}  mdd={_pct(s['max_drawdown'])}"
                )
    print(f"\nWrote {result.get('stats_csv')}")


def _run_live_update() -> dict:
    import importlib.util
    import sys

    update_path = Path(__file__).resolve().parent / "update_index.py"
    spec = importlib.util.spec_from_file_location("hlp_update_index", update_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {update_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hlp_update_index"] = mod
    spec.loader.exec_module(mod)
    return mod.update(VAULT_ADDRESS)


def run_dashboard() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="HLP Vault Index",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #0c1015 0%, #121821 40%, #0f1419 100%);
            color: {COLORS["text"]};
        }}
        [data-testid="stSidebar"] {{
            background: #0d1218;
            border-right: 1px solid #243041;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 1.55rem;
        }}
        div[data-testid="stMetric"] {{
            background: #151b22;
            border: 1px solid #243041;
            border-radius: 12px;
            padding: 12px 14px;
        }}
        h1, h2, h3 {{
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
            letter-spacing: -0.02em;
        }}
        .hlp-sub {{
            color: {COLORS["muted"]};
            font-size: 0.95rem;
            margin-top: -0.6rem;
            margin-bottom: 1.2rem;
        }}
        .hlp-pill {{
            display: inline-block;
            border: 1px solid #334155;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            color: {COLORS["muted"]};
            font-size: 0.8rem;
            margin-right: 0.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Hyperliquid HLP vault")
    st.markdown(
        f'<div class="hlp-sub">TVL-normalized performance index starting at 1.0 · '
        f'<a href="{VAULT_URL}" target="_blank" style="color:#f0a45d;">vault page</a></div>',
        unsafe_allow_html=True,
    )

    live_fetch_ok = allow_live_fetch()

    @st.cache_data(show_spinner=False)
    def _load_cached(mtime_key: str) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict]:
        del mtime_key
        daily_frame = load_daily_frame(
            DAILY_INDEX_CSV if DAILY_INDEX_CSV.exists() else INDEX_CSV
        )
        weekly_frame = load_index_frame(WEEKLY_INDEX_CSV) if WEEKLY_INDEX_CSV.exists() else None
        native_frame = load_index_frame(INDEX_NATIVE_CSV) if INDEX_NATIVE_CSV.exists() else None
        stats_full = analyze(write_stats=False)
        return daily_frame, weekly_frame, native_frame, stats_full

    if not INDEX_CSV.exists() and not DAILY_INDEX_CSV.exists():
        st.warning(
            "No local index CSVs found. Run `python hlp/update_index.py` locally "
            "or wait for the GitHub Action to commit `hlp/data/*.csv`."
        )
        with st.sidebar:
            if live_fetch_ok:
                refresh_boot = st.button("Fetch latest from Hyperliquid", use_container_width=True)
            else:
                refresh_boot = False
                st.caption("Live fetch disabled on hosted runtime. Data is refreshed via GitHub Actions.")
        if refresh_boot:
            _run_live_update()
            st.cache_data.clear()
            st.rerun()
        st.stop()

    mtime_parts = []
    for path in (INDEX_CSV, DAILY_INDEX_CSV, WEEKLY_INDEX_CSV, INDEX_NATIVE_CSV):
        if path.exists():
            mtime_parts.append(str(path.stat().st_mtime))
    daily, weekly, native, _stats_full = _load_cached("|".join(mtime_parts))

    with st.sidebar:
        st.header("Controls")
        series_choice = st.radio(
            "Series",
            options=["Daily dense", "Weekly dense", "Native HL samples"],
            index=0,
            help=(
                "Dense series interpolate HL PnL anchors onto a regular calendar; "
                "TVL denominator uses DefiLlama when available. Native keeps raw HL samples only."
            ),
        )

        if series_choice == "Weekly dense" and weekly is not None:
            full_df = weekly
            return_col = "period_return"
            series_label = "weekly dense"
            freq_label = "weekly_dense"
        elif series_choice == "Native HL samples" and native is not None:
            full_df = native
            return_col = "period_return"
            series_label = "native HL"
            freq_label = "native_samples"
        else:
            full_df = daily
            return_col = "daily_return" if "daily_return" in daily.columns else "period_return"
            series_label = "daily dense"
            freq_label = "daily_dense"

        min_day = full_df["datetime_utc"].min().date()
        max_day = full_df["datetime_utc"].max().date()
        start_date = st.date_input(
            "Start date",
            value=min_day,
            min_value=min_day,
            max_value=max_day,
            help=(
                "First included datapoint (on/after this date). "
                "Index is rebased to 1.0 here and all metrics use this window."
            ),
        )
        lookback = st.selectbox(
            "Lookback from end",
            options=["All", "365d", "90d", "30d", "7d"],
            index=0,
            help="Optional extra floor: effective start = max(start date, end − lookback).",
        )
        st.markdown("**Outlier return cap**")
        cap_enabled = st.checkbox(
            "Cap top-performing periods",
            value=False,
            help=(
                "Winsorize the upper tail: replace the top X% returns with the highest "
                "return that remains, then rebuild the equity index and metrics."
            ),
        )
        top_pct = st.select_slider(
            "Top percentile to cap",
            options=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
            value=1.0,
            format_func=lambda x: f"{x:g}%",
            disabled=not cap_enabled,
            help="Example: 1% caps every return above the 99th percentile down to that 99th-percentile level.",
        )
        st.markdown("**Axis scales**")
        index_scale = st.radio(
            "Index axis",
            options=["Linear", "Log"],
            index=0,
            horizontal=True,
            key="index_axis_scale",
        )
        tvl_scale = st.radio(
            "TVL axis",
            options=["Linear", "Log"],
            index=0,
            horizontal=True,
            key="tvl_axis_scale",
        )
        st.caption("Drawdown stays linear (can be ≤ 0).")
        st.divider()
        st.caption(f"Data as of **{data_as_of_label()}**")
        if live_fetch_ok:
            refresh = st.button("Fetch latest from Hyperliquid", use_container_width=True)
            st.caption("Runs update_index.py against the public info API and rewrites local CSVs.")
        else:
            refresh = False
            st.caption(
                "Hosted mode: live fetch is off. CSVs are updated by GitHub Actions "
                "and redeployed with the repo. Set secret `HLP_ALLOW_LIVE_FETCH=1` to override."
            )

        with st.expander("Methodology", expanded=False):
            st.markdown(
                """
                Period return uses capital at risk:

                `r_t = (PnL_t − PnL_{t−1}) / TVL_{t−1}`

                `index_0 = 1.0`, then `index_t = index_{t−1} × (1 + r_t)`

                **Start date** truncates the series and rebases the index to 1.0 at the
                first included point; KPI cards, charts, histogram and rolling windows
                all use that window.

                **Outlier cap:** after the window is fixed, the top X% of period returns
                are replaced by the highest remaining return (upper winsorization). The
                equity index is then rebuilt from those capped returns. This reduces the
                influence of a few extreme up-days on total return / CAGR.

                **Dense daily/weekly:** Hyperliquid PnL samples are hard anchors.
                Between anchors, cumulative PnL is linearly interpolated in time.
                The TVL denominator prefers DefiLlama daily HLP TVL (from 2024-12-01),
                else interpolated HL account value. Intra-anchor timing is an assumption —
                not observed fill-level daily PnL.

                Deposits/withdrawals change TVL but not PnL, so they are not treated as performance.
                """
            )

    if refresh:
        with st.spinner("Fetching vault history and rebuilding index..."):
            summary = _run_live_update()
        st.success(
            f"Updated · index {summary['index_end']:.4f} · "
            f"{summary.get('daily_dense_points', summary.get('index_points', '?'))} daily points · "
            f"TVL ${summary['tvl_last']:,.0f}"
        )
        st.cache_data.clear()
        st.rerun()

    effective_start = pd.Timestamp(start_date, tz="UTC")
    if lookback != "All":
        lookback_days = int(lookback.replace("d", ""))
        lookback_start = full_df["datetime_utc"].max() - pd.Timedelta(days=lookback_days)
        if lookback_start > effective_start:
            effective_start = lookback_start

    raw_df = slice_and_rebase(full_df, start=effective_start)
    if raw_df.empty or len(raw_df) < 2:
        st.error("Not enough points after the selected start date. Choose an earlier start.")
        st.stop()

    raw_headline = compute_stats(raw_df, return_col, freq_label)
    cap_meta = {"enabled": False, "n_capped": 0, "cap_return": float("nan"), "top_pct": 0.0}
    compare_df = None
    if cap_enabled:
        plot_df, cap_meta = cap_upper_returns(raw_df, return_col, float(top_pct))
        compare_df = raw_df
        headline = compute_stats(plot_df, return_col, f"{freq_label}_capped")
    else:
        plot_df = raw_df
        headline = raw_headline

    daily_for_roll = slice_and_rebase(daily, start=effective_start)
    if cap_enabled:
        daily_for_roll, _ = cap_upper_returns(
            daily_for_roll,
            "daily_return" if "daily_return" in daily_for_roll.columns else "period_return",
            float(top_pct),
        )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(
        "Index",
        f"{headline['index_end']:.4f}",
        delta=(
            f"{headline['index_end'] - raw_headline['index_end']:+.4f} vs raw"
            if cap_enabled
            else None
        ),
    )
    c2.metric(
        "Total return",
        _pct(headline["total_return"]),
        delta=(
            f"{100.0 * (headline['total_return'] - raw_headline['total_return']):+.2f} pp vs raw"
            if cap_enabled
            else None
        ),
    )
    c3.metric(
        "CAGR",
        _pct(headline["cagr"]),
        delta=(
            f"{100.0 * (headline['cagr'] - raw_headline['cagr']):+.2f} pp vs raw"
            if cap_enabled
            else None
        ),
    )
    c4.metric(
        "Ann. vol",
        _pct(headline["vol_ann"]),
        delta=(
            f"{100.0 * (headline['vol_ann'] - raw_headline['vol_ann']):+.2f} pp vs raw"
            if cap_enabled
            else None
        ),
    )
    c5.metric(
        "Sharpe",
        _fmt_num(headline["sharpe"], 2),
        delta=(
            f"{headline['sharpe'] - raw_headline['sharpe']:+.2f} vs raw"
            if cap_enabled
            and headline["sharpe"] == headline["sharpe"]
            and raw_headline["sharpe"] == raw_headline["sharpe"]
            else None
        ),
    )
    c6.metric(
        "Max DD",
        _pct(headline["max_drawdown"]),
        delta=(
            f"{100.0 * (headline['max_drawdown'] - raw_headline['max_drawdown']):+.2f} pp vs raw"
            if cap_enabled
            else None
        ),
    )

    actual_start = plot_df["datetime_utc"].iloc[0].strftime("%Y-%m-%d")
    pills = (
        f'<span class="hlp-pill">{headline["n_obs"]} obs</span>'
        f'<span class="hlp-pill">{headline["years"]:.2f}y span</span>'
        f'<span class="hlp-pill">from {actual_start}</span>'
        f'<span class="hlp-pill">TVL ${headline["tvl_last"]:,.0f}</span>'
        f'<span class="hlp-pill">{series_label} series</span>'
    )
    if cap_meta.get("enabled"):
        pills += (
            f'<span class="hlp-pill">top {cap_meta["top_pct"]:g}% capped '
            f'({cap_meta["n_capped"]} periods ≤ {_pct(cap_meta["cap_return"])})</span>'
        )
    st.markdown(pills, unsafe_allow_html=True)

    chart_title = f"HLP normalized index · from {actual_start} (rebased to 1.0)"
    if cap_meta.get("enabled"):
        chart_title += f" · top {cap_meta['top_pct']:g}% up-days capped"
    st.plotly_chart(
        plot_index_overview(
            plot_df,
            title=chart_title,
            index_scale=index_scale,
            tvl_scale=tvl_scale,
            compare_df=compare_df,
            compare_name="Uncapped",
        ),
        use_container_width=True,
        theme=None,
    )

    if cap_meta.get("enabled"):
        st.subheader("Raw vs capped")
        cmp = pd.DataFrame(
            [
                {
                    "Version": "Raw",
                    "Index": f"{raw_headline['index_end']:.4f}",
                    "Total return": _pct(raw_headline["total_return"]),
                    "CAGR": _pct(raw_headline["cagr"]),
                    "Ann. vol": _pct(raw_headline["vol_ann"]),
                    "Sharpe": _fmt_num(raw_headline["sharpe"], 2),
                    "Max DD": _pct(raw_headline["max_drawdown"]),
                    "Best period": _pct(raw_headline["best_period_return"]),
                },
                {
                    "Version": f"Top {cap_meta['top_pct']:g}% capped",
                    "Index": f"{headline['index_end']:.4f}",
                    "Total return": _pct(headline["total_return"]),
                    "CAGR": _pct(headline["cagr"]),
                    "Ann. vol": _pct(headline["vol_ann"]),
                    "Sharpe": _fmt_num(headline["sharpe"], 2),
                    "Max DD": _pct(headline["max_drawdown"]),
                    "Best period": _pct(headline["best_period_return"]),
                },
            ]
        )
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        st.caption(
            f"Cap level = {_pct(cap_meta['cap_return'])} "
            f"({cap_meta['n_capped']} of {cap_meta['n_returns']} periods replaced)."
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_returns_histogram(
                plot_df,
                return_col,
                title=(
                    f"{series_label.capitalize()} return distribution"
                    + (f" · top {cap_meta['top_pct']:g}% capped" if cap_meta.get("enabled") else "")
                ),
            ),
            use_container_width=True,
            theme=None,
        )
    with right:
        extra = [
            (
                "Sortino",
                _fmt_num(headline["sortino"], 2),
                _num_delta(headline["sortino"], raw_headline["sortino"]) if cap_enabled else None,
            ),
            (
                "Hit rate",
                _pct(headline["pct_positive_periods"]),
                _pp_delta(headline["pct_positive_periods"], raw_headline["pct_positive_periods"])
                if cap_enabled
                else None,
            ),
            (
                "Best period",
                _pct(headline["best_period_return"]),
                _pp_delta(headline["best_period_return"], raw_headline["best_period_return"])
                if cap_enabled
                else None,
            ),
            (
                "Worst period",
                _pct(headline["worst_period_return"]),
                _pp_delta(headline["worst_period_return"], raw_headline["worst_period_return"])
                if cap_enabled
                else None,
            ),
            ("Skew", _fmt_num(headline["skew"], 2), _num_delta(headline["skew"], raw_headline["skew"]) if cap_enabled else None),
            (
                "Kurtosis",
                _fmt_num(headline["kurtosis"], 2),
                _num_delta(headline["kurtosis"], raw_headline["kurtosis"]) if cap_enabled else None,
            ),
            (
                "Mean ann.",
                _pct(headline["mean_ann"]),
                _pp_delta(headline["mean_ann"], raw_headline["mean_ann"]) if cap_enabled else None,
            ),
            (
                "Median period",
                _pct(headline["median_period_return"], 4),
                _pp_delta(
                    headline["median_period_return"],
                    raw_headline["median_period_return"],
                    digits=4,
                )
                if cap_enabled
                else None,
            ),
            (
                "Median CAGR",
                _pct(headline["median_ann"]),
                _pp_delta(headline["median_ann"], raw_headline["median_ann"])
                if cap_enabled
                else None,
            ),
        ]
        st.subheader("Descriptive stats")
        for i in range(0, len(extra), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(extra):
                    label, value, delta = extra[i + j]
                    col.metric(label, value, delta=delta)

    if cap_meta.get("enabled") and "return_capped" in plot_df.columns:
        capped_rows = plot_df.loc[plot_df["return_capped"]].copy()
        with st.expander(f"Capped periods ({len(capped_rows)})"):
            if capped_rows.empty:
                st.info("No periods exceeded the cap level.")
            else:
                show_cap = capped_rows[
                    ["datetime_utc", "account_value", return_col, "index", "source"]
                ].copy()
                # Show original return from raw_df for context.
                raw_rets = raw_df.set_index("datetime_utc")[return_col]
                show_cap["raw_return"] = show_cap["datetime_utc"].map(raw_rets)
                show_cap["capped_to"] = cap_meta["cap_return"]
                show_cap["account_value"] = show_cap["account_value"].map(lambda x: f"{x:,.2f}")
                show_cap["raw_return"] = show_cap["raw_return"].map(lambda x: _pct(float(x)))
                show_cap["capped_to"] = show_cap["capped_to"].map(lambda x: _pct(float(x)))
                show_cap[return_col] = show_cap[return_col].map(lambda x: _pct(float(x)))
                show_cap["index"] = show_cap["index"].map(lambda x: f"{x:.6f}")
                show_cap = show_cap.rename(
                    columns={
                        "datetime_utc": "Time",
                        "account_value": "TVL",
                        "raw_return": "Raw return",
                        "capped_to": "Capped to",
                        return_col: "Used return",
                        "index": "Index after",
                        "source": "Source",
                    }
                )
                st.dataframe(show_cap, use_container_width=True, hide_index=True)

    st.subheader("Seasonality")
    st.caption(
        "Calendar windows on the daily series (same start date / outlier cap as above). "
        "AVG / MEDIAN rows summarize across windows."
    )
    season_grain = st.radio(
        "Seasonality grain",
        options=["Year", "Quarter", "Month", "Week"],
        index=2,
        horizontal=True,
        key="seasonality_grain",
    )
    season_ret_col = (
        "daily_return" if "daily_return" in daily_for_roll.columns else "period_return"
    )
    season_df = (
        seasonality_stats(daily_for_roll, season_ret_col, season_grain.lower())
        if len(daily_for_roll) >= 2
        else pd.DataFrame()
    )
    if season_df.empty:
        st.info("Not enough daily history for seasonality windows yet.")
    else:
        detail = season_df.loc[~season_df["window"].isin(["AVG", "MEDIAN"])].copy()
        summary = season_df.loc[season_df["window"].isin(["AVG", "MEDIAN"])].copy()

        # Compact bar chart of window returns (exclude summary rows).
        if not detail.empty:
            colors = [
                COLORS["positive"] if r >= 0 else COLORS["negative"]
                for r in detail["total_return"]
            ]
            fig_season = go.Figure(
                go.Bar(
                    x=detail["window"],
                    y=detail["total_return"] * 100.0,
                    marker_color=colors,
                    hovertemplate=(
                        "%{x}<br>Return: %{y:.2f}%"
                        "<br>Hit rate: %{customdata[0]:.1f}%"
                        "<br>Sharpe: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.column_stack(
                        [
                            detail["hit_rate"].astype(float) * 100.0,
                            detail["sharpe"].astype(float),
                        ]
                    ),
                )
            )
            fig_season.add_hline(y=0, line_color=COLORS["zero"], line_width=1)
            fig_season.update_layout(
                **_plotly_layout(
                    title=dict(
                        text=f"{season_grain} total return",
                        x=0.0,
                        xref="paper",
                        font=dict(size=14, color=COLORS["text"]),
                    ),
                    height=340,
                    margin=dict(l=56, r=24, t=56, b=88),
                    showlegend=False,
                    yaxis=dict(
                        title="Return (%)",
                        gridcolor=COLORS["grid"],
                        zeroline=False,
                        title_font=dict(size=11, color=COLORS["muted"]),
                        tickfont=dict(size=11, color=COLORS["muted"]),
                    ),
                    xaxis=dict(
                        gridcolor=COLORS["grid"],
                        tickangle=-45,
                        tickfont=dict(size=10, color=COLORS["muted"]),
                    ),
                )
            )
            st.plotly_chart(fig_season, use_container_width=True, theme=None)

        def _fmt_season(frame: pd.DataFrame) -> pd.DataFrame:
            show = frame[
                [
                    "window",
                    "n_obs",
                    "total_return",
                    "cagr",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                    "vol_ann",
                    "hit_rate",
                ]
            ].copy()
            for col in ("total_return", "cagr", "max_drawdown", "vol_ann", "hit_rate"):
                show[col] = show[col].map(lambda x: _pct(float(x)))
            for col in ("sharpe", "sortino"):
                show[col] = show[col].map(lambda x: _fmt_num(float(x), 2))
            return show.rename(
                columns={
                    "window": season_grain,
                    "n_obs": "Obs",
                    "total_return": "Return",
                    "cagr": "CAGR",
                    "sharpe": "Sharpe",
                    "sortino": "Sortino",
                    "max_drawdown": "Max DD",
                    "vol_ann": "Vol",
                    "hit_rate": "Hit rate",
                }
            )

        st.markdown("**Window averages**")
        st.dataframe(_fmt_season(summary), use_container_width=True, hide_index=True)
        with st.expander(f"All {season_grain.lower()} windows"):
            st.dataframe(_fmt_season(detail), use_container_width=True, hide_index=True)

    st.subheader("Rolling windows (daily dense)")
    roll = rolling_stats(daily_for_roll) if len(daily_for_roll) >= 2 else pd.DataFrame()
    if not roll.empty:
        st.plotly_chart(plot_rolling_comparison(roll), use_container_width=True, theme=None)
        display = roll[
            [
                "window_days",
                "total_return",
                "cagr",
                "vol_ann",
                "sharpe",
                "sortino",
                "max_drawdown",
                "n_obs",
            ]
        ].copy()
        for col in ("total_return", "cagr", "vol_ann", "max_drawdown"):
            display[col] = display[col].map(lambda x: _pct(float(x)))
        for col in ("sharpe", "sortino"):
            display[col] = display[col].map(lambda x: _fmt_num(float(x), 2))
        display = display.rename(
            columns={
                "window_days": "Window (d)",
                "total_return": "Return",
                "cagr": "CAGR",
                "vol_ann": "Vol",
                "sharpe": "Sharpe",
                "sortino": "Sortino",
                "max_drawdown": "Max DD",
                "n_obs": "Obs",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough daily history for rolling windows yet.")

    with st.expander("Latest index rows"):
        show = plot_df.tail(25)[
            ["datetime_utc", "account_value", "delta_pnl", return_col, "index", "source"]
        ].copy()
        show["account_value"] = show["account_value"].map(lambda x: f"{x:,.2f}")
        show["delta_pnl"] = show["delta_pnl"].map(lambda x: f"{x:,.2f}")
        show[return_col] = show[return_col].map(lambda x: _pct(float(x)))
        show["index"] = show["index"].map(lambda x: f"{x:.6f}")
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.caption(
        f"Data as of {data_as_of_label()} · `{DATA_DIR.name}/` · vault `{VAULT_ADDRESS}` · "
        "index is flow-aware (ΔPnL / prior TVL). "
        + (
            "Hosted: CSV refresh via GitHub Actions."
            if is_hosted_streamlit()
            else "Local: use sidebar fetch or `python hlp/update_index.py`."
        )
    )


def _running_in_streamlit() -> bool:
    # Avoid importing streamlit on the plain CLI path (prevents ScriptRunContext warnings).
    import sys

    return "streamlit.runtime.scriptrunner.script_runner" in sys.modules


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HLP normalized index returns")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of report")
    args = parser.parse_args()
    result = analyze()
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


if __name__ == "__main__":
    if _running_in_streamlit():
        run_dashboard()
    else:
        main()
