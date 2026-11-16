"""
Streamlit Cloud entrypoint.

Main file setting: streamlit_app.py
Local: streamlit run streamlit_app.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_APP = Path(__file__).resolve().parent / "hlp" / "analyze_returns.py"
_SPEC = importlib.util.spec_from_file_location("hlp_analyze_returns", _APP)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load dashboard module from {_APP}")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
_MOD.run_dashboard()
