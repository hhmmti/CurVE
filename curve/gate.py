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
    # water_cut_gor_history is driven by the same Validated fluid-character calcs.
    "water_cut_gor_history": ["liquid_rate", "gor", "water_cut"],
}

# Fallbacks the gate is allowed to use for these keys. ``derive_liquid_rate_from_alloc``
# lets water_cut resolve liquid_rate from oil+water when not precomputed — an internal
# derivation, not a trust downgrade (the contract stays Validated).
_TOOL_GATE_FALLBACKS: Dict[str, set] = {
    "production_history": {"derive_liquid_rate_from_alloc"},
    "water_cut_gor_history": {"derive_liquid_rate_from_alloc"},
}

# --- field-name alias layer (§5 naming debt) ----------------------------------
# A small GENERAL mechanism: legacy field-name variants → canonical, applied to the
# frames before the gate reads them, so a contract that asks for the canonical name
# resolves even when a pipeline path delivered the legacy spelling.
#
# Canonical is **lower-snake**. The vendored gate reads fields by exact column name
# (``data_availability_gate._collect_candidate_sources``); rather than touch that
# vendored seam, we normalize the frames here — when a frame has the legacy column
# but not the canonical one, we surface the canonical name from the legacy values.
#
# EXTEND BY ADDING A PAIR — nothing here is GOR-specific. Prompt #4 maps the
# ``delta_p_*`` debt by adding e.g. ``"delta_P_pump_psi": "delta_p_pump_psi"`` to
# this dict; no mechanism change is needed.
_FIELD_ALIASES: Dict[str, str] = {
    "GOR_scf_bbl": "gor_scf_bbl",  # §5 naming debt: legacy upper variant → canonical
    "delta_P_pump_psi": "delta_p_pump_psi",  # §5 naming debt (prompt #4): ΔP_pump legacy → canonical
}


def _apply_field_aliases(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Surface canonical field names from legacy variants on a copy of ``df``.

    For each ``legacy → canonical`` pair, when the frame carries the legacy column
    but not the canonical one, add the canonical column from the legacy values. The
    legacy column is left in place (non-destructive). Returns ``df`` unchanged when
    there is nothing to alias (no copy made on the hot path)."""
    if df is None or len(df) == 0:
        return df
    to_add = {
        canonical: legacy
        for legacy, canonical in _FIELD_ALIASES.items()
        if legacy in df.columns and canonical not in df.columns
    }
    if not to_add:
        return df
    df = df.copy()
    for canonical, legacy in to_add.items():
        df[canonical] = df[legacy]
    return df


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

    # Normalize legacy field-name variants → canonical before the gate reads fields.
    telemetry_df = _apply_field_aliases(telemetry_df)
    production_df = _apply_field_aliases(production_df)

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


# --- ΔP history tools: Estimated + flags threading (prompt #4) -----------------
# The first CurVE gate path that carries **Estimated + source flags** end-to-end. It
# runs on the ALREADY-COMPUTED preprocessed frame (the vendored ΔP compute is wrapped,
# not re-run here) and maps the underlying gate's defaulted/fallback status onto the
# CurVE label, merging in the resolution layer's per-input source flags + the PIP
# coverage flag. There is NO Proxy path here — PIP is measured-or-missing.

# The vendored calculation contract that backs ΔP_pump on the preprocessed path. Its
# output_label is "Estimated" (hydrostatic always uses depth/SG), so the worst-label
# fold keeps the tool Estimated even before the resolution-layer flags are merged.
_DELTA_P_GATE_KEY = "pump_delta_p_preprocessed"

# Per-tool projection ("x-axis") columns the figure needs beyond ΔP itself. A missing
# one is a not-ready/blocked condition (distinct from Estimated), per prompt #4.
_DELTA_P_PROJECTION_FIELDS: Dict[str, List[str]] = {
    # ΔP projected against operating frequency over time (V3).
    "delta_p_frequency": ["motor_frequency_hz"],
    # ΔP decomposed into its pressure components over time (V10).
    "delta_p_composition": ["delta_p_hyd_psi", "tubing_pressure_psi", "p_dis_downhole_psi"],
}


def _column_usable(df: pd.DataFrame, name: str) -> bool:
    """True when ``name`` is a column with at least one finite, non-null value."""
    if df is None or name not in df.columns:
        return False
    series = pd.to_numeric(df[name], errors="coerce")
    return bool(series.notna().any())


def run_delta_p_tool_gate(
    tool_name: str,
    analyzed_df: Optional[pd.DataFrame],
    resolved_inputs: Any,
    coverage: Dict[str, Any],
) -> Dict[str, Any]:
    """Gate a ΔP history tool on its computed frame; thread Estimated + flags.

    Args:
        tool_name: ``delta_p_frequency`` or ``delta_p_composition``.
        analyzed_df: the vendored-compute output (already carries ΔP + components).
        resolved_inputs: a :class:`curve.delta_p_inputs.DeltaPInputs` (depth/SG values,
            per-input source flags, and the resolution trust label).
        coverage: the :func:`curve.delta_p_inputs.pip_coverage` dict (PIP present/absent).

    Returns ``{"status": available|blocked, "trust_label": str|None, "flags": [...]}``:
        * zero PIP coverage → blocked ``pip_coverage_zero`` (hard block).
        * missing projection column → blocked ``missing_x_axis_input: <field>``.
        * underlying ΔP contract not ready (e.g. discharge pressure absent) → blocked.
        * otherwise available, ``Estimated`` (worst of the resolution + contract label),
          with the depth/SG source flags + a ``pip_coverage_partial`` flag when some
          PIP-absent rows were excluded.
    """
    if tool_name not in _DELTA_P_PROJECTION_FIELDS:
        raise KeyError(f"No ΔP projection fields registered for tool: {tool_name}")

    if _is_empty(analyzed_df):
        return {"status": "blocked", "trust_label": None, "flags": ["telemetry_or_production_absent"]}

    # Hard block: zero PIP coverage — no fabricated/proxied intake, so nothing to show.
    if coverage.get("zero", True):
        return {"status": "blocked", "trust_label": None, "flags": ["pip_coverage_zero"]}

    # Projection ("x-axis") column presence — a missing one is not-ready, not Estimated.
    missing_proj = [
        f"missing_x_axis_input: {field}"
        for field in _DELTA_P_PROJECTION_FIELDS[tool_name]
        if not _column_usable(analyzed_df, field)
    ]
    if missing_proj:
        return {"status": "blocked", "trust_label": None, "flags": missing_proj}

    # Map the underlying ΔP contract status onto the CurVE label. The vendored gate
    # reads p_dis_downhole_psi + pump_intake_pressure_psi from the computed frame; a
    # not-ready report (e.g. discharge absent) blocks.
    df = _apply_field_aliases(analyzed_df)
    context = {
        "dataframes": {"preprocessed": df},
        "fallbacks_available": {"default_well_depth_ft", "default_sg_values"},
        "proxies_available": set(),  # NO proxy — PIP is measured-or-missing.
    }
    reports = run_data_availability_gate(context, calculation_keys=[_DELTA_P_GATE_KEY])
    not_ready = [r for r in reports if not r.ready]
    if not_ready:
        flags = [f"{r.calculation}: missing {', '.join(r.missing_fields)}" for r in not_ready]
        return {"status": "blocked", "trust_label": None, "flags": flags}

    # The underlying contract only governs READINESS here (p_dis present, PIP not
    # all-missing). Its nominal output_label is a static "Estimated" — we do NOT floor
    # the CurVE label with it, because that would make Validated unreachable even when
    # every ΔP input is measured/real. The CurVE label is decided by the resolution
    # layer's input provenance (the actual mapping of defaulted/fallback → Estimated vs
    # all-real → Validated). This never inflates: the resolution layer downgrades to
    # Estimated for any defaulted/overridden input.
    trust_label = resolved_inputs.trust_label

    # Flags: the per-input source flags (depth_* / sg_*) + the PIP coverage flag when
    # PIP-absent rows were excluded (partial coverage).
    flags = list(resolved_inputs.flags)
    if coverage.get("partial"):
        flags.append(
            f"pip_coverage_partial: {coverage['n_present']}/{coverage['n_total']} rows"
        )

    return {"status": "available", "trust_label": trust_label, "flags": flags}
