"""CurVE tools (M3).

``production_history`` and ``water_cut_gor_history`` are **real, end-to-end**: real
Athena telemetry + production → per-tool gate → vendored physics → vendored Plotly →
the shared ``{status, values, trust_label, flags, figure_ref, figure}`` envelope
(``curve.envelope``; CurVE-decisions §3 D3). Both are connection-free Validated tools
and share the data path, the gate keys, and the success/error envelope shape.
``curve_position`` remains an **M1 stub** (real spec, mock body) — real in M4. The
``bubble_point_screen`` stub was retired in M3 (v2 tool, not in the v1 roster).

Naming convention (documented in README):
  * snake_case
  * capability-named — the verb/noun of the question the tool answers
    (``production_history``, ``water_cut_gor_history``, ``curve_position``), not the
    implementation module it will eventually call. The model routes on the
    capability, so the name + description must read like an operator's intent.

THE ENVELOPE (CurVE-decisions §3 D2/D3):
  A real tool returns ``{values, trust_label, flags, figure_ref, figure}``. The
  engine sends ONLY ``{values, trust_label, flags, status}`` back to the model — the
  ``figure`` (a Plotly object) and ``figure_ref`` go to the UI, never into the model
  (no image tokens; the model narrates from ``values`` only). ``trust_label`` is
  carried faithfully from the gate — never hardcoded.

ORG / WELL ARE INJECTED, NOT MODEL-SUPPLIED:
  ``production_history``'s ``inputSchema`` exposes ONLY the time-window selector. The
  engine backend-injects ``organization_id`` + ``well_id`` from the session record
  (CurVE-decisions §3 D3). The model never sees or supplies them.
"""

from typing import Any, Callable, Dict, Optional

import pandas as pd

from compute.preprocessed_calcs import WELL_DEPTH_FT
from plotting.preprocessed_charts import (
    build_allocation_temporal,
    build_delta_p_composition,
    build_delta_p_pump_vs_frequency,
    build_water_cut_gor_analysis,
)

from . import data, delta_p_inputs, well_depth
from ._vendored import preprocessed_pipeline_service
from .envelope import error_envelope, success_envelope
from .gate import run_delta_p_tool_gate, run_tool_gate

# Vendored pipeline fns (loaded by path to skip the broken services/__init__.py).
prepare_daily_data = preprocessed_pipeline_service.prepare_daily_data
run_preprocessed_analysis = preprocessed_pipeline_service.run_preprocessed_analysis

# Keys the engine strips from a tool result before it reaches the model. The Plotly
# figure and its UI ref never go back to the model (CurVE-decisions §3 D2).
NON_MODEL_RESULT_KEYS = {"figure", "figure_ref"}


# --- real tool: production_history --------------------------------------------


def _round(value: Any, ndigits: int = 1) -> Optional[float]:
    """Round to a JSON-friendly float, mapping NaN/None → None."""
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _project_values(
    well_id: str,
    daily: pd.DataFrame,
    telemetry_rows: int,
) -> Dict[str, Any]:
    """Project the compute output → the model-facing ``values`` for narration + KPIs.

    These are the fields proposed for review (CurVE-decisions §3 D3 per-tool spec):
    period coverage, the latest day's rates/fluid character, and the window trend.
    The model narrates from these; it does not restate every raw number.
    """
    daily = daily.copy()
    daily["observation_day"] = pd.to_datetime(daily["observation_day"], errors="coerce")
    daily = daily.dropna(subset=["observation_day"]).sort_values("observation_day")
    n_days = len(daily)

    period = {
        "start": str(daily["observation_day"].min().date()) if n_days else None,
        "end": str(daily["observation_day"].max().date()) if n_days else None,
        "n_days": int(n_days),
        "n_telemetry_points": int(telemetry_rows),
    }

    latest: Dict[str, Any] = {}
    trend: Dict[str, Any] = {}
    if n_days:
        last = daily.iloc[-1]
        first = daily.iloc[0]
        latest = {
            "observation_day": str(last["observation_day"].date()),
            "oil_rate_bbl_day": _round(last.get("alloc_oil_vol")),
            "water_rate_bbl_day": _round(last.get("alloc_water_vol")),
            "gas_rate_mcf_day": _round(last.get("alloc_gas_vol")),
            "liquid_rate_bbl_day": _round(last.get("liquid_rate_bbl_day")),
            "water_cut": _round(last.get("water_cut"), 3),
            "gor": _round(last.get("gor")),
        }
        oil_first = _round(first.get("alloc_oil_vol"))
        oil_last = _round(last.get("alloc_oil_vol"))
        change_pct = None
        direction = "flat"
        if oil_first not in (None, 0) and oil_last is not None:
            change_pct = round((oil_last - oil_first) / oil_first * 100.0, 1)
            if change_pct <= -5:
                direction = "declining"
            elif change_pct >= 5:
                direction = "rising"
        trend = {
            "oil_rate_first_bbl_day": oil_first,
            "oil_rate_last_bbl_day": oil_last,
            "oil_rate_change_pct": change_pct,
            "direction": direction,
        }

    return {"well_id": well_id, "period": period, "latest": latest, "trend": trend}


def production_history(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``production_history``: fetch → gate (before compute) → compute → figure → envelope.

    ``tool_input`` carries the backend-injected ``organization_id`` + ``well_id``
    (from the session record) merged with the model's window selectors
    (``start_date`` / ``end_date``). The well/org are NOT model-supplied.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    start_date = tool_input.get("start_date")
    end_date = tool_input.get("end_date")

    # Defensive: the engine injects org/well from the session. If they are absent
    # (e.g. a session-less call), do not fetch — report it rather than scan broadly.
    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    try:
        # Data access (the M2 seam). The join + feature engineering is the vendored
        # service, called only after the gate clears.
        telemetry_df, production_df = data.fetch_preprocessed_window(
            organization_id, well_id, start_date, end_date
        )

        # Gate BEFORE compute (code-enforced). Presence + readiness → status/label/flags.
        gate = run_tool_gate("production_history", telemetry_df, production_df)
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        # Vendored physics (join → feature-engineer) and the vendored V1 figure.
        analyzed, meta = run_preprocessed_analysis(
            telemetry_df, production_df, well_depth_ft=WELL_DEPTH_FT
        )
        daily = prepare_daily_data(analyzed)
        figure = build_allocation_temporal(daily, title=f"Production History — {well_id}")

        values = _project_values(
            well_id, daily, telemetry_rows=int(meta.get("telemetry_rows", len(analyzed)))
        )
    except Exception as exc:  # data/compute failure → structured envelope, not a raw exception
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    return success_envelope(
        values=values,
        trust_label=gate["trust_label"],  # carried from the gate, not hardcoded
        flags=gate["flags"],
        figure_ref=f"production_history::{well_id}",
        figure=figure,  # → UI only; the engine strips this before the model sees it
    )


# --- real tool: water_cut_gor_history -----------------------------------------


def _project_wcg_values(
    well_id: str,
    daily: pd.DataFrame,
    telemetry_rows: int,
) -> Dict[str, Any]:
    """Project the compute output → the model-facing ``values`` for the water-cut/GOR story.

    Same shape as ``_project_values`` (period / latest / trend) but the fluid-character
    fields the operator's water-cut/GOR-over-time question is about: latest water cut +
    GOR + liquid rate, and the window trend in each. The model narrates from these.
    """
    daily = daily.copy()
    daily["observation_day"] = pd.to_datetime(daily["observation_day"], errors="coerce")
    daily = daily.dropna(subset=["observation_day"]).sort_values("observation_day")
    n_days = len(daily)

    period = {
        "start": str(daily["observation_day"].min().date()) if n_days else None,
        "end": str(daily["observation_day"].max().date()) if n_days else None,
        "n_days": int(n_days),
        "n_telemetry_points": int(telemetry_rows),
    }

    latest: Dict[str, Any] = {}
    trend: Dict[str, Any] = {}
    if n_days:
        last = daily.iloc[-1]
        first = daily.iloc[0]
        latest = {
            "observation_day": str(last["observation_day"].date()),
            "water_cut": _round(last.get("water_cut"), 3),
            "gor": _round(last.get("gor")),
            "liquid_rate_bbl_day": _round(last.get("liquid_rate_bbl_day")),
            "oil_rate_bbl_day": _round(last.get("alloc_oil_vol")),
            "water_rate_bbl_day": _round(last.get("alloc_water_vol")),
            "gas_rate_mcf_day": _round(last.get("alloc_gas_vol")),
        }

        def _delta(field: str, ndigits: int = 1):
            v_first = _round(first.get(field), ndigits)
            v_last = _round(last.get(field), ndigits)
            change_pct = None
            if v_first not in (None, 0) and v_last is not None:
                change_pct = round((v_last - v_first) / v_first * 100.0, 1)
            return v_first, v_last, change_pct

        wc_first, wc_last, wc_change = _delta("water_cut", 3)
        gor_first, gor_last, gor_change = _delta("gor")
        direction = "flat"
        if wc_change is not None:
            if wc_change >= 5:
                direction = "watering_up"
            elif wc_change <= -5:
                direction = "drying_out"
        trend = {
            "water_cut_first": wc_first,
            "water_cut_last": wc_last,
            "water_cut_change_pct": wc_change,
            "gor_first_scf_bbl": gor_first,
            "gor_last_scf_bbl": gor_last,
            "gor_change_pct": gor_change,
            "direction": direction,
        }

    return {"well_id": well_id, "period": period, "latest": latest, "trend": trend}


def water_cut_gor_history(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``water_cut_gor_history``: fetch → gate (before compute) → compute → figure → envelope.

    Mirrors ``production_history`` 1:1 — same data path, same per-tool gate (Validated
    fluid-character calcs), same shared envelope. Answers the operator's water-cut /
    GOR-over-time question. Org/well are backend-injected, never model-supplied.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    start_date = tool_input.get("start_date")
    end_date = tool_input.get("end_date")

    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    try:
        # Re-fetch its own telemetry+production frame via the SAME data path (per-session
        # re-read; the known double-fetch latency is logged Bucket-C debt, not special-cased).
        telemetry_df, production_df = data.fetch_preprocessed_window(
            organization_id, well_id, start_date, end_date
        )

        gate = run_tool_gate("water_cut_gor_history", telemetry_df, production_df)
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        # Vendored physics (wraps the existing water-cut/GOR/liquid compute fns inside
        # engineer_features) and the vendored V2 water-cut-vs-GOR figure.
        analyzed, meta = run_preprocessed_analysis(
            telemetry_df, production_df, well_depth_ft=WELL_DEPTH_FT
        )
        daily = prepare_daily_data(analyzed)
        figure = build_water_cut_gor_analysis(
            daily, title=f"Water Cut & GOR History — {well_id}"
        )

        values = _project_wcg_values(
            well_id, daily, telemetry_rows=int(meta.get("telemetry_rows", len(analyzed)))
        )
    except Exception as exc:  # data/compute failure → structured envelope, not a raw exception
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    return success_envelope(
        values=values,
        trust_label=gate["trust_label"],  # carried from the gate, not hardcoded
        flags=gate["flags"],
        figure_ref=f"water_cut_gor_history::{well_id}",
        figure=figure,  # → UI only; the engine strips this before the model sees it
    )


# --- real tools: delta_p_frequency + delta_p_composition (ΔP history) ----------
# Both sit on the SAME vendored delta_p_pump preprocessed compute and differ only in
# projection / x-axis / figure. They are the FIRST CurVE tools to carry the Estimated
# trust label end-to-end (depth + SG resolved by curve.delta_p_inputs) and the first to
# resolve a real depth (curve.well_depth rrc read) with an operator override. PIP is
# measured-or-missing — ΔP is computed on PIP-present rows; PIP-absent rows are excluded
# with a coverage flag; zero PIP coverage hard-blocks. Org/well/overrides are
# backend-injected (not model args); the Converse spec exposes only the time window.


def _dp_period(daily: pd.DataFrame, coverage: Dict[str, Any]) -> Dict[str, Any]:
    """Period coverage block shared by both ΔP tools, including PIP coverage."""
    n_days = len(daily)
    return {
        "start": str(daily["observation_day"].min().date()) if n_days else None,
        "end": str(daily["observation_day"].max().date()) if n_days else None,
        "n_days": int(n_days),
        "n_telemetry_points": int(coverage.get("n_present", 0)),
        "pip_coverage": {
            "rows_with_pip": int(coverage.get("n_present", 0)),
            "rows_total": int(coverage.get("n_total", 0)),
            "fraction": coverage.get("fraction", 0.0),
        },
    }


def _trend_of(daily: pd.DataFrame, field_name: str, ndigits: int = 1):
    """First/last/percent-change of a daily field (None-safe)."""
    if not len(daily):
        return None, None, None
    first = _round(daily.iloc[0].get(field_name), ndigits)
    last = _round(daily.iloc[-1].get(field_name), ndigits)
    change_pct = None
    if first not in (None, 0) and last is not None:
        change_pct = round((last - first) / first * 100.0, 1)
    return first, last, change_pct


def _project_dp_frequency_values(
    well_id: str,
    daily: pd.DataFrame,
    resolved: "delta_p_inputs.DeltaPInputs",
    coverage: Dict[str, Any],
) -> Dict[str, Any]:
    """ΔP-vs-frequency narration payload: ΔP_pump + motor frequency, trend, provenance."""
    daily = daily.copy()
    daily["observation_day"] = pd.to_datetime(daily["observation_day"], errors="coerce")
    daily = daily.dropna(subset=["observation_day"]).sort_values("observation_day")

    latest: Dict[str, Any] = {}
    trend: Dict[str, Any] = {}
    if len(daily):
        last = daily.iloc[-1]
        latest = {
            "observation_day": str(last["observation_day"].date()),
            "delta_p_pump_psi": _round(last.get("delta_p_pump_psi")),
            "motor_frequency_hz": _round(last.get("motor_frequency_hz")),
            "pump_intake_pressure_psi": _round(last.get("pump_intake_pressure_psi")),
            "p_dis_downhole_psi": _round(last.get("p_dis_downhole_psi")),
        }
        dp_first, dp_last, dp_change = _trend_of(daily, "delta_p_pump_psi")
        hz_first, hz_last, hz_change = _trend_of(daily, "motor_frequency_hz")
        direction = "flat"
        if dp_change is not None:
            direction = "rising" if dp_change >= 5 else "declining" if dp_change <= -5 else "flat"
        trend = {
            "delta_p_pump_first_psi": dp_first,
            "delta_p_pump_last_psi": dp_last,
            "delta_p_pump_change_pct": dp_change,
            "motor_frequency_first_hz": hz_first,
            "motor_frequency_last_hz": hz_last,
            "motor_frequency_change_pct": hz_change,
            "direction": direction,
        }

    return {
        "well_id": well_id,
        "period": _dp_period(daily, coverage),
        "latest": latest,
        "trend": trend,
        "inputs": resolved.as_values(),
    }


def _project_dp_composition_values(
    well_id: str,
    daily: pd.DataFrame,
    resolved: "delta_p_inputs.DeltaPInputs",
    coverage: Dict[str, Any],
) -> Dict[str, Any]:
    """ΔP-composition narration payload: the pressure-component decomposition + trend."""
    daily = daily.copy()
    daily["observation_day"] = pd.to_datetime(daily["observation_day"], errors="coerce")
    daily = daily.dropna(subset=["observation_day"]).sort_values("observation_day")

    latest: Dict[str, Any] = {}
    trend: Dict[str, Any] = {}
    if len(daily):
        last = daily.iloc[-1]
        latest = {
            "observation_day": str(last["observation_day"].date()),
            "pump_intake_pressure_psi": _round(last.get("pump_intake_pressure_psi")),
            "delta_p_hyd_psi": _round(last.get("delta_p_hyd_psi")),
            "tubing_pressure_psi": _round(last.get("tubing_pressure_psi")),
            "p_dis_downhole_psi": _round(last.get("p_dis_downhole_psi")),
            "delta_p_pump_psi": _round(last.get("delta_p_pump_psi")),
        }
        dp_first, dp_last, dp_change = _trend_of(daily, "delta_p_pump_psi")
        hyd_first, hyd_last, hyd_change = _trend_of(daily, "delta_p_hyd_psi")
        direction = "flat"
        if dp_change is not None:
            direction = "rising" if dp_change >= 5 else "declining" if dp_change <= -5 else "flat"
        trend = {
            "delta_p_pump_first_psi": dp_first,
            "delta_p_pump_last_psi": dp_last,
            "delta_p_pump_change_pct": dp_change,
            "delta_p_hyd_first_psi": hyd_first,
            "delta_p_hyd_last_psi": hyd_last,
            "delta_p_hyd_change_pct": hyd_change,
            "direction": direction,
        }

    return {
        "well_id": well_id,
        "period": _dp_period(daily, coverage),
        "latest": latest,
        "trend": trend,
        "inputs": resolved.as_values(),
    }


def _run_delta_p_tool(
    tool_input: Dict[str, Any],
    *,
    tool_name: str,
    figure_builder: Callable[..., Any],
    value_projector: Callable[..., Dict[str, Any]],
    title: str,
) -> Dict[str, Any]:
    """Shared body for both ΔP history tools: fetch → resolve inputs → compute → gate
    (PIP coverage + projection presence) → figure → envelope.

    The two tools differ only in ``figure_builder`` (the vendored ΔP plot), the
    ``value_projector`` (the per-tool ``values`` projection / x-axis) and ``title``.
    Org/well + the resolved-input overrides arrive backend-injected in ``tool_input``.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    start_date = tool_input.get("start_date")
    end_date = tool_input.get("end_date")
    # Operator-controlled depth/SG overrides — setup-injected like org/well, never a
    # model-facing tool arg (the spec exposes only the time window).
    resolved_inputs_ctx = tool_input.get("resolved_inputs")

    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    try:
        # Re-fetch this tool's own preprocessed frame via the SAME data path.
        telemetry_df, production_df = data.fetch_preprocessed_window(
            organization_id, well_id, start_date, end_date
        )
        # Presence gate BEFORE compute — don't compute on absent data.
        if telemetry_df is None or len(telemetry_df) == 0 or production_df is None or len(production_df) == 0:
            return error_envelope("blocked", ["telemetry_or_production_absent"], well_id=well_id)

        # Resolve depth (real rrc → override → default) + SG (override → default). The
        # real rrc depth read is the ported app query (curve.well_depth).
        rrc_depth_ft = well_depth.fetch_well_depth_ft(well_id)
        resolved = delta_p_inputs.resolve_from_context(resolved_inputs_ctx, rrc_depth_ft)

        # Vendored ΔP_pump preprocessed compute, with the resolved depth/SG. PIP is
        # measured-or-missing inside the service (null/zero intake → NaN ΔP).
        analyzed, _meta = run_preprocessed_analysis(
            telemetry_df,
            production_df,
            well_depth_ft=resolved.depth_ft,
            sg_oil=resolved.sg_oil,
            sg_water=resolved.sg_water,
        )

        # PIP coverage — READ from the computed frame (the #2 intake-state truth).
        coverage = delta_p_inputs.pip_coverage(analyzed)

        # Gate: zero-coverage / missing-projection → blocked; else Estimated + flags.
        gate = run_delta_p_tool_gate(tool_name, analyzed, resolved, coverage)
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        # Compute ΔP only on PIP-present rows; the vendored figure builds from them.
        present = delta_p_inputs.pip_present_rows(analyzed, coverage)
        figure = figure_builder(present, title=f"{title} — {well_id}")
        daily = prepare_daily_data(present)
        values = value_projector(well_id, daily, resolved, coverage)
    except Exception as exc:  # data/compute failure → structured envelope, not an exception
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    return success_envelope(
        values=values,
        trust_label=gate["trust_label"],  # Estimated (or Validated) — carried from the gate
        flags=gate["flags"],
        figure_ref=f"{tool_name}::{well_id}",
        figure=figure,  # → UI only; the engine strips it before the model sees it
    )


def probe_delta_p_readiness(
    tool_name: str,
    telemetry_df: pd.DataFrame,
    production_df: pd.DataFrame,
    well_id: str,
    resolved_inputs_ctx: Optional[Dict[str, Any]],
    rrc_depth_ft: Optional[float],
) -> Dict[str, Any]:
    """Front-load a ΔP tool's gate WITHOUT building a figure (Streamlit setup step).

    Resolves depth/SG (real rrc + any overrides), runs the vendored ΔP compute, reads
    PIP coverage, and runs the per-tool gate — returning the same ``{status,
    trust_label, flags}`` head the tool will carry, plus the coverage + resolved-input
    provenance. Lets the readiness surface show the Estimated label, the PIP coverage,
    and which depth tier fired before the operator asks a question.
    """
    if telemetry_df is None or len(telemetry_df) == 0 or production_df is None or len(production_df) == 0:
        return {
            "gate": {"status": "blocked", "trust_label": None, "flags": ["telemetry_or_production_absent"]},
            "coverage": {"n_present": 0, "n_total": 0, "fraction": 0.0},
            "resolved": None,
        }
    resolved = delta_p_inputs.resolve_from_context(resolved_inputs_ctx, rrc_depth_ft)
    analyzed, _meta = run_preprocessed_analysis(
        telemetry_df,
        production_df,
        well_depth_ft=resolved.depth_ft,
        sg_oil=resolved.sg_oil,
        sg_water=resolved.sg_water,
    )
    coverage = delta_p_inputs.pip_coverage(analyzed)
    gate = run_delta_p_tool_gate(tool_name, analyzed, resolved, coverage)
    return {
        "gate": gate,
        "coverage": {k: coverage[k] for k in ("n_present", "n_total", "fraction") if k in coverage},
        "resolved": resolved.as_values(),
    }


def delta_p_frequency(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``delta_p_frequency``: ΔP_pump vs operating frequency over time (V3).

    Pressure-lift response to motor frequency — the pump's measured differential
    pressure against the frequency that drove it. Estimated trust (hydrostatic depth /
    SG); blocked when PIP coverage is zero or motor frequency is absent.
    """
    return _run_delta_p_tool(
        tool_input,
        tool_name="delta_p_frequency",
        figure_builder=build_delta_p_pump_vs_frequency,
        value_projector=_project_dp_frequency_values,
        title="ΔP Pump vs Motor Frequency",
    )


def delta_p_composition(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``delta_p_composition``: ΔP_pump pressure-component decomposition over time (V10).

    Decomposes downhole discharge pressure into its parts — pump intake, hydrostatic
    column, tubing pressure, and the resulting discharge — so the operator sees what
    builds the ΔP_pump. Same Estimated trust + PIP coverage handling as
    ``delta_p_frequency``; blocked when PIP coverage is zero or the pressure components
    are absent.
    """
    return _run_delta_p_tool(
        tool_input,
        tool_name="delta_p_composition",
        figure_builder=build_delta_p_composition,
        value_projector=_project_dp_composition_values,
        title="Temporal ΔP Composition",
    )


# --- M1 stub tools (still mock; real in M4) ------------------------------------


def curve_position(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    return {"mock": "curve_position output", "received_input": tool_input}


# --- real tool specs (Converse toolConfig shape) ------------------------------
# Converse expects inputSchema wrapped under a "json" key.

_PRODUCTION_HISTORY_SPEC = {
    "toolSpec": {
        "name": "production_history",
        "description": (
            "Retrieve historical production and telemetry for the selected well over "
            "a time window (oil/water/gas allocation rates, fluid character, recent "
            "trend). Use for questions about how the well has produced or behaved "
            "over time, recent trends, or 'what has this well been doing'. The well "
            "is already set up for this session — supply only the time window."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Window start (ISO date YYYY-MM-DD), optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Window end (ISO date YYYY-MM-DD), optional.",
                    },
                },
                "required": [],
            }
        },
    }
}

_WATER_CUT_GOR_HISTORY_SPEC = {
    "toolSpec": {
        "name": "water_cut_gor_history",
        "description": (
            "Retrieve the selected well's water-cut and gas-oil-ratio (GOR) history "
            "over a time window — how the produced-fluid character has changed "
            "(watering up / drying out, GOR trend) alongside liquid rate. Use for "
            "questions about water cut, GOR, gas-oil ratio, fluid mix, or 'is this "
            "well watering up'. Note: water cut (not water rate) and GOR (gas-OIL "
            "ratio, not gas-liquid). The well is already set up for this session — "
            "supply only the time window."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Window start (ISO date YYYY-MM-DD), optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Window end (ISO date YYYY-MM-DD), optional.",
                    },
                },
                "required": [],
            }
        },
    }
}

_DELTA_P_FREQUENCY_SPEC = {
    "toolSpec": {
        "name": "delta_p_frequency",
        "description": (
            "Retrieve the selected well's pump differential pressure (ΔP across the "
            "pump, discharge minus intake) plotted against motor frequency over a time "
            "window — the pressure-lift response to operating frequency. Use for "
            "questions about pump ΔP, pressure rise/lift across the pump, head vs "
            "frequency, or 'how is the pump's differential pressure responding to "
            "frequency'. Note: ΔP_pump (discharge − intake), not stage/TDH ΔP, and "
            "intake (PIP) is measured-only. The well is already set up for this "
            "session — supply only the time window."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Window start (ISO date YYYY-MM-DD), optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Window end (ISO date YYYY-MM-DD), optional.",
                    },
                },
                "required": [],
            }
        },
    }
}

_DELTA_P_COMPOSITION_SPEC = {
    "toolSpec": {
        "name": "delta_p_composition",
        "description": (
            "Retrieve the selected well's pump pressure decomposition over a time "
            "window — how downhole discharge pressure is built from pump intake (PIP), "
            "the hydrostatic column, and tubing pressure, and the resulting ΔP across "
            "the pump. Use for questions about the pressure components, what makes up "
            "the pump's differential pressure, hydrostatic vs tubing contribution, or "
            "'break down the pump pressure'. Intake (PIP) is measured-only; depth/SG "
            "for the hydrostatic term are resolved with provenance. The well is already "
            "set up for this session — supply only the time window."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Window start (ISO date YYYY-MM-DD), optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Window end (ISO date YYYY-MM-DD), optional.",
                    },
                },
                "required": [],
            }
        },
    }
}

_CURVE_POSITION_SPEC = {
    "toolSpec": {
        "name": "curve_position",
        "description": (
            "Determine where the pump is operating on its performance curve right "
            "now — the ideal-curve overlay (single + multi-frequency, ΔP) plus the "
            "BEP position. Use for 'where am I on the curve', 'am I near best "
            "efficiency point', or operating-point-vs-pump-curve questions. "
            "(In production this needs a resolved pump connection; mocked in M1.)"
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "well_id": {
                        "type": "string",
                        "description": "Identifier of the well.",
                    },
                    "frequency_hz": {
                        "type": "number",
                        "description": "Operating frequency to evaluate, optional.",
                    },
                },
                "required": ["well_id"],
            }
        },
    }
}

# --- registry -----------------------------------------------------------------
# name -> {"spec": <toolSpec dict>, "fn": <callable>}. The engine builds toolConfig
# from the specs and dispatches tool_use by name to fn.

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "production_history": {"spec": _PRODUCTION_HISTORY_SPEC, "fn": production_history},
    "water_cut_gor_history": {
        "spec": _WATER_CUT_GOR_HISTORY_SPEC,
        "fn": water_cut_gor_history,
    },
    "delta_p_frequency": {"spec": _DELTA_P_FREQUENCY_SPEC, "fn": delta_p_frequency},
    "delta_p_composition": {
        "spec": _DELTA_P_COMPOSITION_SPEC,
        "fn": delta_p_composition,
    },
    "curve_position": {"spec": _CURVE_POSITION_SPEC, "fn": curve_position},
}


def build_tool_config(registry: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the Converse ``toolConfig`` from a registry."""
    registry = registry if registry is not None else TOOL_REGISTRY
    return {"tools": [entry["spec"] for entry in registry.values()]}
