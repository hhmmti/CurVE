"""CurVE per-tool gate — the M2 adapter over the vendored data-availability gate.

CurVE-decisions §4 Decision 3 (per-tool gate invariant): every tool runs its gate
*before* computing and returns ``{available | blocked | proxy, trust_label, flags}``.
The M0 audit found the vendored ``services/data_availability_gate.py`` emits a
**summary-shaped** report (``total_calculations_checked``, … ) rather than this
per-tool envelope. This module is the **thin adapter** that folds that summary shape
into the per-tool envelope — the "port the gate service → per-tool gate" carry-
forward. It does NOT re-implement the gate: it calls the vendored
``run_data_availability_gate`` and maps the result.

Scope (M2): only ``production_history``. It is the §4 "safest path" tool — telemetry
+ production present → ``available`` / ``Validated`` / ``flags: []``. Other tools'
gate keys arrive in M3/M4 (the ``_TOOL_GATE_KEYS`` map is the extension seam).

This gate is code-enforced and runs before compute — it is NOT a model instruction.
A value is presented as Validated ONLY because this gate labels it so; the tool
carries the label faithfully into its envelope (never hardcoded).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

# Loaded by path to skip the broken vendored services/__init__.py (see curve._vendored).
from ._vendored import data_availability_gate

run_data_availability_gate = data_availability_gate.run_data_availability_gate

# Trust-label precedence, best → worst. When a tool aggregates several underlying
# calculation contracts, the tool's label is the WORST among them (never inflated).
_TRUST_PRECEDENCE: List[str] = ["Validated", "Estimated", "Proxy", "Research prototype"]

# Per-tool → the vendored calculation-contract keys that back it. production_history
# (V1 Allocation Temporal) is driven by the allocation calcs, all Validated.
_TOOL_GATE_KEYS: Dict[str, List[str]] = {
    "production_history": ["liquid_rate", "gor", "water_cut"],
}

# Fallbacks the gate is allowed to use for these keys. ``derive_liquid_rate_from_alloc``
# lets water_cut resolve liquid_rate from oil+water when not precomputed — an internal
# derivation, not a trust downgrade (the contract stays Validated).
_TOOL_GATE_FALLBACKS: Dict[str, set] = {
    "production_history": {"derive_liquid_rate_from_alloc"},
}


def _is_empty(df: Optional[pd.DataFrame]) -> bool:
    return df is None or len(df) == 0


def _worst_label(labels: List[str]) -> str:
    """Return the lowest-trust label among ``labels`` (worst wins; never inflate)."""
    return max(labels, key=lambda label: _TRUST_PRECEDENCE.index(label))


def run_tool_gate(
    tool_name: str,
    telemetry_df: Optional[pd.DataFrame],
    production_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Gate one tool against its fetched data; return the per-tool envelope head.

    Returns ``{"status": available|blocked, "trust_label": str|None, "flags": [...]}``.
    (``proxy`` status is reserved for the M3 proxy-input tools; production_history is
    Validated and never proxies.)
    """
    if tool_name not in _TOOL_GATE_KEYS:
        raise KeyError(f"No gate keys registered for tool: {tool_name}")

    # Presence: telemetry AND production must both be present, else the tool is
    # unavailable (CurVE-decisions §4 F4 — telemetry-absence → tool unavailable).
    if _is_empty(telemetry_df) or _is_empty(production_df):
        return {
            "status": "blocked",
            "trust_label": None,
            "flags": ["telemetry_or_production_absent"],
        }

    context = {
        "dataframes": {"telemetry": telemetry_df, "production": production_df},
        "fallbacks_available": _TOOL_GATE_FALLBACKS.get(tool_name, set()),
        "proxies_available": set(),
    }
    reports = run_data_availability_gate(
        context, calculation_keys=_TOOL_GATE_KEYS[tool_name]
    )

    not_ready = [r for r in reports if not r.ready]
    if not_ready:
        flags = [
            f"{r.calculation}: missing {', '.join(r.missing_fields)}" for r in not_ready
        ]
        return {"status": "blocked", "trust_label": None, "flags": flags}

    trust_label = _worst_label([r.output_label for r in reports])
    return {"status": "available", "trust_label": trust_label, "flags": []}
