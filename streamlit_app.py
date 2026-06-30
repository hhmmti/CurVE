"""CurVE M2 surface — Streamlit chat page over the shipped M2-core (FIRST KEITH DEMO).

This is a **thin surface**: it reuses the 2a shipped core verbatim and adds no
loop / tool / gate / session logic of its own.

  * setup     → ``run_tool_gate`` (front-load availability) + ``save_session``
  * chat      → ``run_curve_turn`` (the shipped hand-rolled loop) with the session
  * render    → the single-tool answer shape (CurVE-decisions §3 D8): one Plotly
                figure + KPI cards + one synthesizing narration paragraph + a
                trust-label badge
  * dev panel → the loop laid bare: tool trace, gate verdict, token/cost, timing,
                raw (model-facing) envelope — all from the SAME shipped loop output

Mirrors the deployed operator UX (CurVE-decisions §2 D6): front-loaded setup step
THEN chat. No pump pick (production_history is connection-free).

ASSUMPTION (flagged): the well dropdown is a small **configurable known-well list**
for the demo (``CURVE_KNOWN_WELLS`` env JSON, or the ``Manual entry`` option).
Production would populate it by querying the operator's own wells. Either way every
query filters by the selected ``organization_id`` (no cross-org data).

Run:
    aws sso login --profile roam-ai
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import streamlit as st

from cli import _estimate_cost_usd  # reuse the M1 cost estimator (no logic copied)
from curve import data, delta_p_inputs, ideal_catalog, recommendations, session, well_depth
from curve.engine import run_curve_turn
from curve.gate import run_tool_gate
from curve.tools import (
    NON_MODEL_RESULT_KEYS,
    probe_connection_coverage,
    probe_delta_p_readiness,
    probe_recommendation_readiness,
)

# The ΔP history tools front-loaded at setup (Estimated-label tools) + the Validated
# connection-free tools. Order drives the availability report rows.
_DELTA_P_TOOLS = ("delta_p_frequency", "delta_p_composition")
# The three recommendation-dependent tools (M3) — CurVE's second data path. They share
# the recommendation-absence hard block, front-loaded here per-well.
_REC_TOOLS = ("recommendation_comparison", "affinity_check", "energy_efficiency")

DEFAULT_PROFILE = os.environ.get("CURVE_AWS_PROFILE", "roam-ai")

# Configurable known-well list (demo). Override with CURVE_KNOWN_WELLS as a JSON list
# of {"label","organization_id","well_id"}. Empty by default so the operator either
# sets the env var or uses the Manual-entry option — no fake ids baked in.
def _load_known_wells() -> List[Dict[str, str]]:
    raw = os.environ.get("CURVE_KNOWN_WELLS")
    if not raw:
        return []
    try:
        wells = json.loads(raw)
        return [w for w in wells if w.get("organization_id") and w.get("well_id")]
    except (ValueError, TypeError):
        return []


KNOWN_WELLS = _load_known_wells()

_TRUST_BADGE = {
    "Validated": ("#1a7f37", "✅ Validated"),
    "Estimated": ("#9a6700", "🟡 Estimated"),
    "Proxy": ("#8250df", "🟣 Proxy"),
    "Research prototype": ("#cf222e", "🔬 Research prototype"),
}


# --- helpers (no core logic — composition + presentation only) ----------------


def _bind_session_store() -> None:
    """Point the session store at a dict inside st.session_state (storage-body swap).

    Only the backing mapping changes; ``load_session`` / ``save_session`` and the
    record shape are unchanged. Called every rerun so the store stays bound.
    """
    session.use_store(st.session_state.setdefault("_curve_session_store", {}))


def _set_aws_profile(profile: str) -> None:
    """Honor the chosen SSO profile for BOTH Bedrock and Athena via AWS_PROFILE.

    The wrapper and the data accessor both fall back to AWS_PROFILE when no explicit
    profile is passed, so setting it here routes the tool's internal Athena fetch and
    the Bedrock loop through the same profile (precedence: explicit → AWS_PROFILE →
    default chain).
    """
    if profile:
        os.environ["AWS_PROFILE"] = profile


def _coverage(telemetry_df, production_df) -> Dict[str, Any]:
    """The both-present observation_day range — the window production_history can serve.

    Intersection (max of mins, min of maxes) because the tool needs telemetry AND
    production. Fed into the setup context so the model keeps relative windows in
    range (the M2-demo date-anchor fix). observation_day is an ISO string, so
    min/max compare lexicographically.
    """

    def _range(df):
        if df is None or len(df) == 0 or "observation_day" not in df:
            return None, None
        s = df["observation_day"].astype(str)
        return s.min(), s.max()

    t_min, t_max = _range(telemetry_df)
    p_min, p_max = _range(production_df)
    mins = [d for d in (t_min, p_min) if d]
    maxs = [d for d in (t_max, p_max) if d]
    return {
        "min_day": max(mins) if len(mins) == 2 else (mins[0] if mins else None),
        "max_day": min(maxs) if len(maxs) == 2 else (maxs[0] if maxs else None),
    }


@st.cache_data(show_spinner=False)
def _load_ideal_catalog():
    """Re-read the ideal catalog once per session (cached across wells, same catalog).

    Written to re-read each session (build-plan M4 seam) so Track 2's auto-connection
    slots in with no rework. Obsolete rows are KEPT (CurVE deviation) — surfaced+flagged
    downstream, never dropped.
    """
    return ideal_catalog.fetch_ideal_catalog()


@st.cache_data(show_spinner=False)
def _availability_probe(organization_id: str, well_id: str) -> Dict[str, Any]:
    """Front-load gate: fetch once + run the SHIPPED gate to build the availability report.

    Same gate logic invoked at setup as before compute (CurVE-decisions §4 F1).
    Also records the available data range so the chat turn's relative-window math is
    anchored. Cached per (org, well) so it doesn't re-run on every Streamlit rerun.
    """
    telemetry_df, production_df = data.fetch_preprocessed_window(
        organization_id, well_id
    )
    # Real rrc depth read (the ported app query) — drives the depth-override pre-fill
    # and lets the report state which depth tier fired for this well.
    rrc_depth_ft = well_depth.fetch_well_depth_ft(well_id)
    # Front-load the ΔP tools' gate at the no-override baseline (rrc → default depth,
    # default SG), so the readiness surface shows the Estimated label + PIP coverage.
    delta_p_gates = {
        tool: probe_delta_p_readiness(
            tool, telemetry_df, production_df, well_id, resolved_inputs_ctx={}, rrc_depth_ft=rrc_depth_ft
        )
        for tool in _DELTA_P_TOOLS
    }
    # Front-load the recommendation tools' gate: fetch the latest recommendation ONCE
    # (CurVE's second data path, the Athena mirror) and run each tool's gate head so the
    # availability report shows recommendation-absence honestly per-well, BEFORE chat.
    latest_rec = recommendations.fetch_latest_recommendation(organization_id, well_id)
    rec_gates = {
        tool: probe_recommendation_readiness(
            tool, organization_id, well_id, resolved_inputs_ctx={},
            rrc_depth_ft=rrc_depth_ft, latest_row=latest_rec,
        )
        for tool in _REC_TOOLS
    }
    # M4 / 4a: front-load the pump-connection coverage report (the de-risking check).
    # Re-read the catalog, compute total fluid, BEP-narrow → candidate list. Reuses the
    # already-fetched telemetry/production frames (no double read). Never a gate.
    try:
        connection = probe_connection_coverage(
            organization_id, well_id, _load_ideal_catalog(),
            resolved_inputs_ctx={},
            telemetry_df=telemetry_df, production_df=production_df,
            rrc_depth_ft=rrc_depth_ft,
        )
    except Exception as exc:  # catalog read / schema mismatch — surface, don't crash setup
        connection = {"report": None, "error": str(exc), "n_candidates": 0}
    return {
        "gate": run_tool_gate("production_history", telemetry_df, production_df),
        # Second connection-free Validated tool — same data, same gate keys.
        "gate_wcg": run_tool_gate("water_cut_gor_history", telemetry_df, production_df),
        "coverage": _coverage(telemetry_df, production_df),
        "rrc_depth_ft": rrc_depth_ft,
        "delta_p": delta_p_gates,
        "recommendation": rec_gates,
        "recommendation_present": recommendations.has_recommendation(latest_rec),
        "connection": connection,
    }


def _trust_badge(label: Optional[str]) -> str:
    color, text = _TRUST_BADGE.get(label, ("#57606a", f"⚪ {label or 'Unknown'}"))
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:12px;font-size:0.85em;font-weight:600'>{text}</span>"
    )


def _render_kpis(values: Dict[str, Any]) -> None:
    """KPI cards from the tool's ``values`` (CurVE-decisions §3 D8 per-tool block)."""
    latest = (values or {}).get("latest") or {}
    trend = (values or {}).get("trend") or {}
    period = (values or {}).get("period") or {}

    c1, c2, c3, c4 = st.columns(4)
    oil = latest.get("oil_rate_bbl_day")
    change = trend.get("oil_rate_change_pct")
    c1.metric(
        "Oil rate (latest)",
        f"{oil:.0f} bbl/d" if oil is not None else "—",
        delta=f"{change:+.1f}% over window" if change is not None else None,
    )
    water = latest.get("water_rate_bbl_day")
    c2.metric("Water rate (latest)", f"{water:.0f} bbl/d" if water is not None else "—")
    wc = latest.get("water_cut")
    c3.metric("Water cut (latest)", f"{wc*100:.1f}%" if wc is not None else "—")
    gas = latest.get("gas_rate_mcf_day")
    c4.metric("Gas rate (latest)", f"{gas:.0f} mcf/d" if gas is not None else "—")

    if period:
        st.caption(
            f"Window: {period.get('start')} → {period.get('end')} "
            f"· {period.get('n_days')} days · "
            f"{period.get('n_telemetry_points')} telemetry points"
        )


def _render_wcg_kpis(values: Dict[str, Any]) -> None:
    """KPI cards for water_cut_gor_history (water cut + GOR + liquid rate)."""
    latest = (values or {}).get("latest") or {}
    trend = (values or {}).get("trend") or {}
    period = (values or {}).get("period") or {}

    c1, c2, c3, c4 = st.columns(4)
    wc = latest.get("water_cut")
    wc_change = trend.get("water_cut_change_pct")
    c1.metric(
        "Water cut (latest)",
        f"{wc*100:.1f}%" if wc is not None else "—",
        delta=f"{wc_change:+.1f}% over window" if wc_change is not None else None,
    )
    gor = latest.get("gor")
    gor_change = trend.get("gor_change_pct")
    c2.metric(
        "GOR (latest)",
        f"{gor:.0f} scf/bbl" if gor is not None else "—",
        delta=f"{gor_change:+.1f}% over window" if gor_change is not None else None,
    )
    liquid = latest.get("liquid_rate_bbl_day")
    c3.metric("Liquid rate (latest)", f"{liquid:.0f} bbl/d" if liquid is not None else "—")
    c4.metric("Trend", (trend.get("direction") or "—").replace("_", " "))

    if period:
        st.caption(
            f"Window: {period.get('start')} → {period.get('end')} "
            f"· {period.get('n_days')} days · "
            f"{period.get('n_telemetry_points')} telemetry points"
        )


def _render_dp_kpis(values: Dict[str, Any]) -> None:
    """KPI cards for the ΔP history tools — ΔP, intake, the depth used + its source,
    and PIP coverage (so the Estimated basis is visible alongside the numbers)."""
    latest = (values or {}).get("latest") or {}
    trend = (values or {}).get("trend") or {}
    period = (values or {}).get("period") or {}
    inputs = (values or {}).get("inputs") or {}

    c1, c2, c3, c4 = st.columns(4)
    dp = latest.get("delta_p_pump_psi")
    dp_change = trend.get("delta_p_pump_change_pct")
    c1.metric(
        "ΔP pump (latest)",
        f"{dp:.0f} psi" if dp is not None else "—",
        delta=f"{dp_change:+.1f}% over window" if dp_change is not None else None,
    )
    pip = latest.get("pump_intake_pressure_psi")
    c2.metric("Pump intake (latest)", f"{pip:.0f} psi" if pip is not None else "—")
    depth = inputs.get("depth_ft")
    c3.metric(
        "Depth used",
        f"{depth:,.0f} ft" if depth is not None else "—",
        help=f"source: {inputs.get('depth_source')} · SG source: {inputs.get('sg_source')}",
    )
    cov = period.get("pip_coverage") or {}
    c4.metric(
        "PIP coverage",
        f"{cov.get('rows_with_pip', 0)}/{cov.get('rows_total', 0)} rows",
    )

    if period:
        st.caption(
            f"Window: {period.get('start')} → {period.get('end')} "
            f"· {period.get('n_days')} days · "
            f"{period.get('n_telemetry_points')} PIP-present points · "
            f"depth {depth:,.0f} ft ({inputs.get('depth_source')})"
            if depth is not None
            else f"Window: {period.get('start')} → {period.get('end')}"
        )


def _render_recommendation_comparison_kpis(values: Dict[str, Any]) -> None:
    """KPI cards for recommendation_comparison — current vs recommended setpoint + deltas."""
    freq = (values or {}).get("motor_frequency_hz") or {}
    tp = (values or {}).get("tubing_pressure_psi") or {}
    liq = (values or {}).get("liquid_rate_bpd") or {}
    oil = (values or {}).get("oil_rate_bpd") or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Motor frequency",
        f"{freq.get('recommended'):.1f} Hz" if freq.get("recommended") is not None else "—",
        delta=f"{freq.get('delta'):+.1f} Hz" if freq.get("delta") is not None else None,
    )
    c2.metric(
        "Tubing pressure",
        f"{tp.get('recommended'):.0f} psi" if tp.get("recommended") is not None else "—",
        delta=f"{tp.get('delta'):+.0f} psi" if tp.get("delta") is not None else None,
    )
    c3.metric(
        "Liquid rate",
        f"{liq.get('recommended'):.0f} bbl/d" if liq.get("recommended") is not None else "—",
        delta=f"{liq.get('delta'):+.0f} bbl/d" if liq.get("delta") is not None else None,
    )
    c4.metric(
        "Oil rate",
        f"{oil.get('recommended'):.0f} bbl/d" if oil.get("recommended") is not None else "—",
        delta=f"{oil.get('delta'):+.0f} bbl/d" if oil.get("delta") is not None else None,
    )
    if values and values.get("recommendation_uuid"):
        st.caption(
            f"Recommendation `{values.get('recommendation_uuid')}` · goal: "
            f"{values.get('method')}"
        )


def _render_affinity_kpis(values: Dict[str, Any]) -> None:
    """KPI cards for affinity_check — speed ratio + per-check agreement."""
    flow = (values or {}).get("flow_check") or {}
    pressure = (values or {}).get("pressure_check") or {}

    c1, c2, c3, c4 = st.columns(4)
    sr = (values or {}).get("speed_ratio")
    c1.metric("Speed ratio (N₂/N₁)", f"{sr:.3f}" if sr is not None else "—")
    fd = (values or {}).get("frequency_delta_hz")
    c2.metric("Frequency Δ", f"{fd:+.2f} Hz" if fd is not None else "—")
    c3.metric("Flow agreement", flow.get("agreement") or "—")
    c4.metric("Overall", (values or {}).get("overall_agreement") or "—")
    if pressure.get("available"):
        st.caption(
            f"Pressure check: {pressure.get('agreement')} "
            f"(Δ {pressure.get('difference_pct')}%) · mode: {(values or {}).get('mode')}"
        )


def _render_energy_kpis(values: Dict[str, Any]) -> None:
    """KPI cards for energy_efficiency — efficiency + specific power + power source."""
    cur = (values or {}).get("current") or {}
    terms = (values or {}).get("term_provenance") or {}

    eff = cur.get("direct_power_efficiency_pct")
    eff = eff if eff is not None else cur.get("proxy_power_efficiency_pct")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pump efficiency", f"{eff:.1f}%" if eff is not None else "—")
    sp = cur.get("specific_power_kwh_per_liquid_bbl")
    c2.metric("Specific power", f"{sp:.2f} kWh/bbl" if sp is not None else "—")
    hp = cur.get("hydraulic_kw_estimate")
    c3.metric("Hydraulic power", f"{hp:.1f} kW" if hp is not None else "—")
    c4.metric("Power source", (cur.get("power_source") or "—"))
    st.caption(
        f"Term provenance — liquid: {terms.get('liquid')} · ΔP: {terms.get('delta_p')} "
        f"· power: {terms.get('power')} (overall = weakest)"
    )


# Physics tools whose envelopes render as figure + KPI block; name → KPI renderer.
_TOOL_KPI_RENDERERS = {
    "production_history": _render_kpis,
    "water_cut_gor_history": _render_wcg_kpis,
    "delta_p_frequency": _render_dp_kpis,
    "delta_p_composition": _render_dp_kpis,
    "recommendation_comparison": _render_recommendation_comparison_kpis,
    "affinity_check": _render_affinity_kpis,
    "energy_efficiency": _render_energy_kpis,
}


def _render_tool_envelope(name: str, envelope: Dict[str, Any]) -> None:
    """Render one tool envelope: badge + tool-specific KPI + figure (or a status warning)."""
    status = envelope.get("status")
    if status == "available":
        st.markdown(_trust_badge(envelope.get("trust_label")), unsafe_allow_html=True)
        kpi_renderer = _TOOL_KPI_RENDERERS.get(name)
        if kpi_renderer is not None:
            kpi_renderer(envelope.get("values"))
        figure = envelope.get("figure")
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)
    else:
        st.warning(
            f"`{name}` is **{status}** — "
            f"{', '.join(envelope.get('flags') or []) or 'no detail'}."
        )


def _render_answer(entry: Dict[str, Any], dev_mode: bool) -> None:
    """Render one assistant turn: per-tool badge + KPI + figure + narration (+ dev panel)."""
    # Render each physics tool the loop called (usually one — single-tool answer shape).
    for output in entry.get("tool_outputs") or []:
        name = output.get("name")
        envelope = output.get("result")
        if name in _TOOL_KPI_RENDERERS and isinstance(envelope, dict):
            _render_tool_envelope(name, envelope)

    # The one synthesizing narration paragraph (answer-level).
    st.markdown(entry.get("narration") or "_(no narration)_")

    st.caption(f"⏱ {entry.get('elapsed_s', 0):.1f}s this turn")

    if dev_mode:
        _render_dev_panel(entry)


def _render_dev_panel(entry: Dict[str, Any]) -> None:
    """Dev mode: expose the SHIPPED loop's own outputs (no recomputation)."""
    with st.expander("🔧 Dev — agentic loop", expanded=False):
        st.write(
            {
                "tool_trace": entry.get("tool_trace"),
                "stop_reason": entry.get("stop_reason"),
                "iterations": entry.get("iterations"),
                "elapsed_s": round(entry.get("elapsed_s", 0), 2),
            }
        )
        usage = entry.get("usage") or {}
        st.write(
            {
                "usage": usage,
                "estimated_cost_usd": round(_estimate_cost_usd(usage), 4),
            }
        )
        st.markdown("**Per-tool gate verdict + model-facing envelope** (figure stripped):")
        for output in entry.get("tool_outputs") or []:
            result = output.get("result")
            model_facing = (
                {k: v for k, v in result.items() if k not in NON_MODEL_RESULT_KEYS}
                if isinstance(result, dict)
                else result
            )
            st.write(f"`{output.get('name')}`")
            st.json(model_facing)


def _render_delta_p_input_overrides(
    record: Dict[str, Any], availability: Dict[str, Any]
) -> None:
    """Editable, pre-filled depth + SG fields that feed the ΔP tools (in scope, #5).

    The fields are pre-filled with EXACTLY what the tool would resolve without an
    override — the real rrc depth when present, else the CurVE 10,000 ft default, and
    the default SG endpoints. The operator accepts or overrides; a changed value is
    stored as an override on the session's ``resolved_inputs`` (which the engine injects
    into every ΔP tool call), changes the computed ΔP, and keeps the label Estimated.
    Accepting the pre-fill stores no override — the default/real value is used and
    surfaced here, never silently assumed. PIP is never shown as editable — it is
    measured-or-missing, not defaultable.
    """
    rrc_depth = availability.get("rrc_depth_ft")
    prefill = delta_p_inputs.default_prefill()
    # Pre-fill depth with the real rrc value when present, else the CurVE default.
    prefill_depth = float(rrc_depth) if rrc_depth else float(prefill["depth_ft"])
    depth_tier = "real rrc depth" if rrc_depth else f"CurVE default {prefill['depth_ft']:,.0f} ft"

    with st.expander("⚙️ ΔP physics inputs — depth & SG (editable, pre-filled)", expanded=False):
        st.caption(
            f"Pre-filled with the value the ΔP tools would use with no override "
            f"(**{depth_tier}**). Edit to override — a hand-typed value is honored, "
            f"keeps the trust label **Estimated**, and is reflected in the source flags. "
            f"PIP is measured-only and never defaulted."
        )
        c1, c2, c3 = st.columns(3)
        depth_val = c1.number_input(
            "Well depth (TVD, ft)",
            min_value=0.0,
            value=prefill_depth,
            step=100.0,
            key=f"dp_depth_{record['well_id']}",
        )
        sg_oil_val = c2.number_input(
            "SG oil",
            min_value=0.0,
            max_value=2.0,
            value=float(prefill["sg_oil"]),
            step=0.01,
            key=f"dp_sgoil_{record['well_id']}",
        )
        sg_water_val = c3.number_input(
            "SG water",
            min_value=0.0,
            max_value=2.0,
            value=float(prefill["sg_water"]),
            step=0.01,
            key=f"dp_sgwater_{record['well_id']}",
        )

        # Only a CHANGE from the pre-fill is an override; otherwise let the tool resolve
        # real-rrc → default (so accepting a real rrc depth stays Validated-eligible for
        # that input, and accepting the default stays a surfaced default).
        overrides: Dict[str, Any] = {}
        if abs(depth_val - prefill_depth) > 1e-9:
            overrides["depth_override"] = depth_val
        if abs(sg_oil_val - float(prefill["sg_oil"])) > 1e-9:
            overrides["sg_oil_override"] = sg_oil_val
        if abs(sg_water_val - float(prefill["sg_water"])) > 1e-9:
            overrides["sg_water_override"] = sg_water_val

        # Persist onto the session so the engine injects it into each ΔP tool call.
        record["resolved_inputs"] = overrides
        session.save_session(record)

        # Show the provenance the next ΔP answer will carry.
        resolved = delta_p_inputs.resolve_from_context(
            overrides, float(rrc_depth) if rrc_depth else None
        )
        st.markdown(
            _trust_badge(resolved.trust_label)
            + f"  &nbsp; depth **{resolved.depth_ft:,.0f} ft** "
            f"(`{resolved.depth_source}`) · SG {resolved.sg_oil}/{resolved.sg_water} "
            f"(`{resolved.sg_source}`) · flags: `{resolved.flags}`",
            unsafe_allow_html=True,
        )


def _render_pump_pick(record: Dict[str, Any], availability: Dict[str, Any]) -> None:
    """Manual pump-pick UX (M4 / 4a) — VE proposes BEP-narrowed candidates; operator picks.

    The §1 connection ladder, manual rung: candidates come from the BEP-narrowed catalog
    (the coverage report front-loaded at setup); the operator selects one. The pick is a
    setup-injected value stored on the session's ``pump`` field (``set_pump_on_session``),
    NEVER a model-facing tool argument. No pick = an honest blocked state for 4b's
    ``curve_position`` — never a silent default pump. Obsolete candidates are offered
    (flagged), not hidden.
    """
    connection = availability.get("connection") or {}
    report = connection.get("report")
    with st.expander("🔌 Pump connection — manual pick (BEP-narrowed candidates)", expanded=True):
        if connection.get("error"):
            st.warning(f"Catalog/coverage unavailable: {connection['error']}")
            return
        if not report:
            st.info("No connection coverage computed for this well.")
            return

        tf = report.get("total_fluid_bpd")
        st.caption(
            (f"Total fluid (median liquid rate): **{tf:,.0f} bpd** · " if tf is not None
             else "Total fluid: **—** (telemetry absent) · ")
            + f"BEP tolerance **±{report.get('bep_tolerance', 0)*100:.0f}%** · "
            f"**{report.get('n_candidates', 0)}** selectable candidate(s) "
            f"(of {report.get('n_bep_compatible', 0)} BEP-compatible; "
            f"{report.get('n_excluded_missing_coeffs', 0)} excluded for missing curve coeffs)"
        )

        candidates = report.get("candidates") or []
        if not candidates:
            st.warning(
                "No selectable candidates for this well's total fluid — no pump can be "
                "picked, so curve-position questions will block honestly (4b). Widen the "
                "BEP tolerance or verify the well's liquid rate."
            )
            return

        # Candidate labels: esp_model (display) + BEP, obsolete flagged. Value = pump_id (key).
        def _label(c: Dict[str, Any]) -> str:
            bep = f"{c['bep_bpd']:.0f} bpd" if c.get("bep_bpd") is not None else "BEP —"
            tag = "  ⚠ obsolete" if c.get("is_obsolete") else ""
            return f"{c['esp_model']} · {c['manufacturer']}/{c['series']} · {bep}{tag}"

        labels = ["— No pump selected —"] + [_label(c) for c in candidates]
        pump_ids = [None] + [c["pump_id"] for c in candidates]
        # Preserve a prior pick across reruns.
        current_pid = (record.get("pump") or {}).get("pump_id")
        default_idx = pump_ids.index(current_pid) if current_pid in pump_ids else 0
        choice_idx = st.selectbox(
            "Installed pump (manual pick)",
            range(len(labels)),
            index=default_idx,
            format_func=lambda i: labels[i],
            key=f"pump_pick_{record['well_id']}",
        )

        picked_pid = pump_ids[choice_idx]
        if picked_pid is None:
            ideal_catalog.set_pump_on_session(record, None)
            st.info("No pump picked — connection unresolved (curve_position would block).")
            return

        # Build the canonical pick from the re-read catalog (single source of pick shape).
        pick = ideal_catalog.make_pump_pick(
            _load_ideal_catalog(), picked_pid,
            bep_tolerance=report.get("bep_tolerance"),
        )
        ideal_catalog.set_pump_on_session(record, pick)
        obs = " ⚠ **obsolete** (operator-confirmed)" if pick.get("is_obsolete") else ""
        st.markdown(
            f"Connected pump: **{pick['esp_model']}** "
            f"(`pump_id={pick['pump_id']}`, {pick['manufacturer']}/{pick['series']}) · "
            f"BEP {pick['bep_bpd']:.0f} bpd · trust would be **Estimated (catalog)**{obs}"
        )


# --- page ---------------------------------------------------------------------


def render_page() -> None:
    st.set_page_config(page_title="CurVE — Physics Validation", page_icon="🛢️", layout="wide")
    _bind_session_store()

    st.title("🛢️ CurVE — Virtual Engineer physics validation")
    st.caption(
        "M3 demo · production history + water-cut/GOR history, real telemetry + "
        "production, live trust label"
    )

    # --- sidebar: setup step (well select → availability) ---------------------
    with st.sidebar:
        st.header("Setup")
        profile = st.text_input("AWS SSO profile", value=DEFAULT_PROFILE)
        _set_aws_profile(profile)
        dev_mode = st.checkbox("Developer mode (expose the loop)", value=True)

        options = [w["label"] for w in KNOWN_WELLS] + ["Manual entry…"]
        choice = st.selectbox("Select a well", options, index=0)

        if choice == "Manual entry…":
            organization_id = st.text_input("organization_id")
            well_id = st.text_input("well_id")
        else:
            well = next(w for w in KNOWN_WELLS if w["label"] == choice)
            organization_id, well_id = well["organization_id"], well["well_id"]
            st.text(f"org:  {organization_id}\nwell: {well_id}")

        start_setup = st.button("Run setup / availability", type="primary")

    # --- run setup ------------------------------------------------------------
    if start_setup:
        if not (organization_id and well_id):
            st.sidebar.error("Both organization_id and well_id are required.")
        else:
            with st.spinner("Front-loading availability (Athena + gate)…"):
                try:
                    probe = _availability_probe(organization_id, well_id)
                except Exception as exc:  # surface auth/data errors honestly
                    st.sidebar.error(f"Setup failed: {exc}")
                    probe = None
            if probe is not None:
                session_id = f"st-{organization_id}-{well_id}"
                record = session.save_session(
                    session.new_session_record(
                        session_id=session_id,
                        organization_id=organization_id,
                        well_id=well_id,
                        # resolved_inputs starts empty → the ΔP tools resolve depth/SG
                        # by real-rrc → default. The operator's edits below populate the
                        # override keys, which the engine injects into each tool call.
                        resolved_inputs={},
                        availability={
                            "production_history": probe["gate"],
                            "water_cut_gor_history": probe["gate_wcg"],
                            "coverage": probe["coverage"],
                            "rrc_depth_ft": probe.get("rrc_depth_ft"),
                            "delta_p_frequency": probe["delta_p"]["delta_p_frequency"]["gate"],
                            "delta_p_composition": probe["delta_p"]["delta_p_composition"]["gate"],
                            "delta_p_meta": {
                                t: {
                                    "coverage": probe["delta_p"][t]["coverage"],
                                    "resolved": probe["delta_p"][t]["resolved"],
                                }
                                for t in _DELTA_P_TOOLS
                            },
                            # The three recommendation tools' gate heads (incl. the
                            # recommendation-absence block when no rec exists).
                            **{t: probe["recommendation"][t]["gate"] for t in _REC_TOOLS},
                            "recommendation_present": probe.get("recommendation_present"),
                            # M4 / 4a: the pump-connection coverage report (candidate
                            # list); the operator's pick is stored separately on the
                            # record's `pump` field via set_pump_on_session.
                            "connection": probe.get("connection"),
                        },
                        # No pump picked yet — an honest "no connection" state (4b's
                        # curve_position blocks on it). The pick UI below sets it.
                        pump=None,
                    )
                )
                st.session_state["active_session_id"] = session_id
                st.session_state["chat"] = []

    # --- availability report --------------------------------------------------
    active_id = st.session_state.get("active_session_id")
    record = session.load_session(active_id) if active_id else None

    if record is None:
        st.info("Select a well in the sidebar and run setup to begin.")
        return

    availability = record.get("availability") or {}
    gate = availability.get("production_history", {})
    coverage = availability.get("coverage") or {}
    telemetry_available = gate.get("status") == "available"
    # The recommendation tools (second data path) are usable whenever a recommendation
    # exists — independent of telemetry presence. Chat proceeds if EITHER path is live.
    rec_available = availability.get("recommendation_present", False)
    available = telemetry_available or rec_available
    with st.container():
        st.subheader(f"Well {record['well_id']} — availability")
        for tool_name in (
            "production_history",
            "water_cut_gor_history",
            *_DELTA_P_TOOLS,
            *_REC_TOOLS,
        ):
            tool_gate = availability.get(tool_name, {})
            cols = st.columns([1, 3])
            cols[0].markdown(
                _trust_badge(tool_gate.get("trust_label")), unsafe_allow_html=True
            )
            cols[1].markdown(
                f"**{tool_name}** → `{tool_gate.get('status')}`"
                + (f" · flags: {tool_gate.get('flags')}" if tool_gate.get("flags") else "")
            )
        # M4 / 4a: the connection (curve_position) row — driven by the candidate count,
        # not a gate. Reflects whether a pump *can* be connected for this well.
        connection = availability.get("connection") or {}
        conn_report = connection.get("report") or {}
        n_cand = conn_report.get("n_candidates", 0) if not connection.get("error") else 0
        cols = st.columns([1, 3])
        cols[0].markdown(_trust_badge("Estimated"), unsafe_allow_html=True)
        if connection.get("error"):
            conn_msg = f"`blocked` · catalog unavailable: {connection['error']}"
        elif n_cand:
            conn_msg = f"`manual pick` · {n_cand} BEP-narrowed candidate(s) — pick below"
        else:
            conn_msg = "`blocked` · no candidate pump for this well's total fluid"
        cols[1].markdown(f"**pump_connection** → {conn_msg}")
        if coverage.get("min_day") and coverage.get("max_day"):
            st.caption(
                f"Data available: {coverage['min_day']} → {coverage['max_day']}"
            )

    # --- pump connection: manual pick (M4 / 4a) -------------------------------
    _render_pump_pick(record, availability)

    # --- editable ΔP physics inputs (depth + SG overrides) --------------------
    _render_delta_p_input_overrides(record, availability)
    if not telemetry_available:
        st.warning(
            "Telemetry + production not both present for this well — the "
            "telemetry-history tools are unavailable."
            + (
                " A recommendation IS available, so the recommendation tools "
                "(recommendation_comparison / affinity_check / energy_efficiency) "
                "can still answer."
                if rec_available
                else " No recommendation is available either — pick another well."
            )
        )
    if not available:
        return

    # --- chat -----------------------------------------------------------------
    st.divider()
    for entry in st.session_state.get("chat", []):
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.markdown(entry["text"])
            else:
                _render_answer(entry, dev_mode)

    question = st.chat_input(
        "Ask about this well's production, water cut/GOR, ΔP, or the ML "
        "recommendation (comparison, affinity check, energy efficiency)…"
    )
    if question:
        st.session_state["chat"].append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Running the CurVE loop…"):
                started = time.time()
                result = run_curve_turn(
                    question, session=record, profile_name=profile
                )
                elapsed = time.time() - started
            # Rendering is driven by tool_outputs — each physics tool the loop called.
            entry = {
                "role": "assistant",
                "narration": result.get("text"),
                "tool_trace": result.get("tool_trace"),
                "tool_outputs": result.get("tool_outputs"),
                "stop_reason": result.get("stop_reason"),
                "iterations": result.get("iterations"),
                "usage": result.get("usage"),
                "elapsed_s": elapsed,
            }
            st.session_state["chat"].append(entry)
            _render_answer(entry, dev_mode)


if __name__ == "__main__":
    render_page()
