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

Well selection (M5 step 3): the setup sidebar offers **dependent org → well
dropdowns** enumerated from the VE well-configuration table
(``roam_prd_ddb.default.esp_well_configuration_v2``, via
:mod:`curve.well_catalog`) — picking an org filters its wells. The enumeration is
``@st.cache_data``-cached (slow-changing; TTL + a manual refresh button). A
``Manual entry…`` option is retained so an off-catalog well can still be run, and an
enumeration failure falls back to manual entry with a visible notice (never a crash).
This is enumeration/UX only: the chosen org/well feed the same setup path as typed
entry, and every downstream query still filters by the selected ``organization_id``
(no cross-org data).

Run:
    aws sso login --profile roam-ai
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from curve import (
    config,
    data,
    delta_p_inputs,
    ideal_catalog,
    recommendations,
    session,
    well_catalog,
    well_depth,
)
from curve.cost import estimate_cost_usd  # shared cost estimator (no logic copied)
from curve.engine import run_curve_turn
from curve.gate import run_tool_gate
from curve.sql_tool import SESSION_SQL_RESULT_KEY, SESSION_SQL_RESULTS_KEY
from curve.tools import (
    NON_MODEL_RESULT_KEYS,
    SQL_TOOL_NAME,
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

DEFAULT_PROFILE = config.AWS_PROFILE

# Manual-entry sentinel — the retained free-text fallback for off-catalog wells (and
# the graceful fallback when enumeration returns nothing / fails).
MANUAL_ENTRY = "Manual entry…"

_TRUST_BADGE = {
    "Validated": ("#1a7f37", "✅ Validated"),
    "Estimated": ("#9a6700", "🟡 Estimated"),
    "Proxy": ("#8250df", "🟣 Proxy"),
    "Research prototype": ("#cf222e", "🔬 Research prototype"),
}

# --- /sql surface state (M4a) -------------------------------------------------
# M3 stashes the FULL ExecuteResult on the session record. Streamlit reruns the whole
# script on every widget interaction (including a download click), so the full result
# must (a) live in st.session_state to survive the rerun and (b) be keyed PER TURN —
# and, within a turn, one payload PER EXECUTION, since the model can call sql_query
# more than once. These two keys are new and additive.
SQL_RESULTS_STATE_KEY = "_curve_sql_results"  # {turn_id: [payload, …] in exec order}
SQL_TURN_SEQ_KEY = "_curve_sql_turn_seq"  # monotonic per-message id source


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


@st.cache_data(ttl=3600, show_spinner="Enumerating wells…")
def _enumerate_org_wells(profile: str) -> Dict[str, List[str]]:
    """Cached org→wells enumeration for the setup dropdowns (M5 step 3).

    Reads distinct (org, well) from ``roam_prd_ddb.default.esp_well_configuration_v2``
    via :mod:`curve.well_catalog` (the existing awswrangler/Athena read path). Keyed on
    the AWS profile so switching profiles re-queries; a 1-hour TTL plus the sidebar's
    "Refresh well list" button (``_enumerate_org_wells.clear()``) keep it from going
    permanently stale. Slow-changing config table → must not re-query every rerun.
    """
    return well_catalog.fetch_org_well_map(profile_name=profile or None)


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
    # M4 / 4b: representative telemetry motor frequency — the Hz the pump is actually
    # running at. Seeds (only) the curve_position operating-frequency field in the
    # pump-pick step; the operator can override it. NOT line frequency, NOT a default
    # stage count — a physically-real reading, surfaced for the operator to accept/edit.
    _freq = (
        pd.to_numeric(telemetry_df.get("motor_frequency_hz"), errors="coerce").dropna()
        if telemetry_df is not None
        else pd.Series(dtype=float)
    )
    telemetry_hz = float(_freq.median()) if not _freq.empty else None
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
        # Seed for the curve_position operating-frequency field (pump-pick step).
        "telemetry_motor_frequency_hz": telemetry_hz,
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

    # delta_p_frequency only: frequency-normalized ΔP drift (affinity-referenced).
    affinity_norm = (values or {}).get("affinity_normalized")
    if affinity_norm:
        d1, d2, d3, _d4 = st.columns(4)
        adj = affinity_norm.get("affinity_adjusted_delta_p_pct")
        d1.metric(
            "ΔP drift vs affinity",
            f"{adj:+.1f}%" if adj is not None else "—",
            help="observed ΔP change minus the (N₂/N₁)² affinity-expected change",
        )
        exp = affinity_norm.get("affinity_expected_delta_p_change_pct")
        d2.metric("Affinity-expected ΔP", f"{exp:+.1f}%" if exp is not None else "—")
        obs = affinity_norm.get("observed_delta_p_change_pct")
        d3.metric("Observed ΔP change", f"{obs:+.1f}%" if obs is not None else "—")
        if affinity_norm.get("reason"):
            st.caption(f"Affinity-normalized ΔP not computed: {affinity_norm.get('reason')}")

    # delta_p_composition only: friction/%-split + ΔP_pump/PIP drawdown ratio.
    composition = (values or {}).get("composition")
    if composition:
        pct = composition.get("composition_pct") or {}
        e1, e2, e3, e4 = st.columns(4)
        ratio = composition.get("delta_p_intake_ratio")
        e1.metric(
            "ΔP/PIP (drawdown)",
            f"{ratio:.2f}" if ratio is not None else "—",
            help="ΔP_pump ÷ pump-intake pressure — drawdown severity",
        )
        e2.metric(
            "Hydrostatic share",
            f"{pct.get('hydrostatic'):.0f}%" if pct.get("hydrostatic") is not None else "—",
        )
        e3.metric(
            "Backpressure share",
            f"{pct.get('backpressure'):.0f}%" if pct.get("backpressure") is not None else "—",
        )
        e4.metric(
            "Friction share (resid.)",
            f"{pct.get('friction'):.0f}%" if pct.get("friction") is not None else "—",
        )
        if not composition.get("decomposable", True) and composition.get("reason"):
            st.caption(f"ΔP composition not decomposable: {composition.get('reason')}")


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
        # Render every figure the tool carries, in order. curve_position carries a
        # ``figures`` list (single-freq overlay, then affinity family); the other tools
        # carry a singular ``figure``. Support both — a blocked/not-ready tool has
        # neither, so nothing (no placeholder) renders for it.
        figures = envelope.get("figures")
        if figures:
            for figure in figures:
                if figure is not None:
                    st.plotly_chart(figure, use_container_width=True)
        else:
            figure = envelope.get("figure")
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True)
    else:
        st.warning(
            f"`{name}` is **{status}** — "
            f"{', '.join(envelope.get('flags') or []) or 'no detail'}."
        )


# --- /sql result: stash → per-turn session state → render (M4a) ---------------


def take_sql_stashes(record: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """POP every full-result stash this turn produced, in execution order.

    Popping (not reading) is deliberate, and BOTH M3 keys are cleared: a /sql turn that
    FAILS the generation cap writes nothing, and a leftover stash would otherwise let
    the failed turn serve the previous turn's rows. Clearing before each turn and
    popping after means "what is in the stash" is always "what THIS turn produced, or
    nothing".

    Returns a list because one turn can execute sql_query more than once (the engine
    forces the tool, then `auto` lets the model call it again to refine). The legacy
    single slot is drained too so it cannot survive into a later turn.
    """
    if not isinstance(record, dict):
        return []
    record.pop(SESSION_SQL_RESULT_KEY, None)  # legacy single slot — drain, don't use
    return record.pop(SESSION_SQL_RESULTS_KEY, None) or []


def build_sql_download(
    stash: Optional[Dict[str, Any]], well_id: str, turn_id: int, call_index: int = 0
) -> Optional[Dict[str, Any]]:
    """Encode ONE stashed FULL ExecuteResult into a CSV download payload — ONCE.

    Encoding happens here (at turn completion), not at render time, so the bytes are a
    plain value in session state: the download click reruns the script and serves those
    bytes directly. The query is NEVER re-executed for the CSV — the Athena
    ``query_execution_id`` carried alongside is the evidence (it is the id of the single
    M3 execution and never changes across download clicks).

    ``guarded_sql`` rides along so the render path can bind a payload to the block that
    actually ran it, rather than trusting list position (see :func:`sql_payload_for`).
    """
    exec_result = (stash or {}).get("execute_result")
    dataframe = getattr(exec_result, "dataframe", None)
    if dataframe is None:
        return None
    return {
        "csv": dataframe.to_csv(index=False).encode("utf-8"),
        "row_count": getattr(exec_result, "row_count", len(dataframe)),
        "query_execution_id": getattr(exec_result, "query_execution_id", None),
        "data_scanned_bytes": getattr(exec_result, "data_scanned_bytes", 0) or 0,
        "guarded_sql": (stash or {}).get("guarded_sql"),
        "file_name": f"curve_sql_{well_id or 'well'}_turn{turn_id}_{call_index + 1}.csv",
    }


def build_sql_downloads(
    stashes: List[Dict[str, Any]], well_id: str, turn_id: int
) -> List[Dict[str, Any]]:
    """One download payload per execution in this turn, in execution order."""
    payloads = [
        build_sql_download(stash, well_id, turn_id, index)
        for index, stash in enumerate(stashes)
    ]
    return [p for p in payloads if p is not None]


def sql_payload_for(
    payloads: List[Dict[str, Any]], executed_sql: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Bind a rendered block to the payload whose SQL that block actually executed.

    Matching on the executed SQL rather than list position is what makes a mixed turn
    safe: a call that FAILS the generation cap still appends an envelope to
    ``tool_outputs`` but stashes nothing, so envelope index N and stash index N drift
    apart. Position-matching would then hand a block another query's rows — the very
    bug this fix exists to close. Two identical executions in one turn match either
    entry, which is harmless: same SQL, same data.
    """
    if not executed_sql:
        return None
    for payload in payloads:
        if payload.get("guarded_sql") == executed_sql:
            return payload
    return None


def sql_row_caption(row_count: int, shown: int) -> str:
    """Honest row accounting under the inline table ("47 matched, showing 5")."""
    if not row_count:
        return "0 rows matched."
    if row_count <= shown:
        return f"{row_count} row(s) matched — all shown."
    return f"{row_count} rows matched · showing the first {shown}."


def _render_sql_result(
    envelope: Dict[str, Any], turn_id: Optional[int], call_index: int = 0
) -> None:
    """Render one ``sql_query`` envelope — B7 (top-5 + CSV) + C10 (collapsed SQL).

    Deliberately NOT a physics answer block: no trust badge, no ``st.metric`` KPI cards.
    This is retrieval of stored columns, and mixing it with the labeled physics surface
    would let a viewer read it as validated physics. The transparency mechanism here is
    the **executed** SQL — the guarded/injected statement, showing the org/well scope,
    date window and row cap the customer actually got — not the pre-guard generated SQL.
    """
    with st.container(border=True):
        st.caption(
            "🔎 **Data retrieval** — stored columns returned by a scoped SQL query. "
            "Not a physics validation; no trust label applies."
        )

        # Honest failure (B6 cap exceeded): reason + last attempted SQL, and NOTHING
        # that looks like a result — no table, no download, no empty success.
        if envelope.get("error"):
            st.error(
                f"**Query not produced.** {envelope['error']}\n\n"
                f"Last failure reason: `{envelope.get('last_reason') or 'unknown'}`"
            )
            with st.expander("Show query", expanded=False):
                st.caption("Last attempted SQL (rejected — never executed).")
                st.code(envelope.get("last_sql") or "-- no SQL was produced", language="sql")
            return

        rows = envelope.get("sample_rows") or []
        row_count = int(envelope.get("row_count") or 0)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("The query ran successfully and matched 0 rows.")
        st.caption(sql_row_caption(row_count, len(rows)))

        # C10: default-collapsed, and it holds the EXECUTED sql (``sql``), not
        # ``generated_sql`` — the injected scoping is the point of showing it.
        with st.expander("Show query", expanded=False):
            st.caption(
                "Executed SQL — organization/well scope, date window and row cap are "
                "injected by the guard before execution."
            )
            st.code(envelope.get("sql") or "", language="sql")

        # B7: CSV of the FULL result, served from THIS BLOCK's own stored bytes. The
        # widget key carries the call index because one turn can render several SQL
        # blocks, and a repeated key is a hard DuplicateWidgetID crash.
        payload = sql_payload_for(
            (st.session_state.get(SQL_RESULTS_STATE_KEY) or {}).get(turn_id) or [],
            envelope.get("sql"),
        )
        if payload is not None:
            st.download_button(
                "⬇ Download full result (CSV)",
                data=payload["csv"],
                file_name=payload["file_name"],
                mime="text/csv",
                key=f"sql_download_{turn_id}_{call_index}",
            )
            st.caption(
                f"{payload['row_count']} row(s) · served from this turn's stored "
                f"result — downloading does not re-run the query · Athena query "
                f"`{payload['query_execution_id']}` · "
                f"{payload['data_scanned_bytes']:,} bytes scanned."
            )


def _render_answer(entry: Dict[str, Any], dev_mode: bool) -> None:
    """Render one assistant turn: per-tool badge + KPI + figure + narration (+ dev panel)."""
    # Render each physics tool the loop called, IN TOOL-TRACE ORDER (tool_outputs is
    # appended by the engine in call order). Every tool with an envelope is rendered —
    # NOT only those in _TOOL_KPI_RENDERERS: curve_position carries a figure but no KPI
    # block, and gating on the KPI map silently dropped its overlay. _render_tool_envelope
    # renders the KPI only when a renderer exists, always renders an available tool's
    # figure, and draws no chart/placeholder for a blocked/not-ready tool (its status is
    # surfaced as text + in the narration).
    # One turn can call sql_query more than once, so SQL blocks are numbered within the
    # turn — that index makes each block's download widget key unique.
    sql_call_index = 0
    for output in entry.get("tool_outputs") or []:
        name = output.get("name")
        envelope = output.get("result")
        if not isinstance(envelope, dict):
            continue
        # /sql (M4a): the retrieval tool has its own, deliberately non-physics block —
        # it carries no gate status/trust label, so the physics renderer would report it
        # as a status-less "blocked" warning. Every other tool is untouched.
        if name == SQL_TOOL_NAME:
            _render_sql_result(envelope, entry.get("sql_turn_id"), sql_call_index)
            sql_call_index += 1
        else:
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
                "estimated_cost_usd": round(estimate_cost_usd(usage), 4),
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
        # This handler OWNS only the depth/SG override keys — rebuild that slice (so a
        # revert-to-prefill clears the override) while PRESERVING keys owned by other
        # setup steps (the pump-pick step's `stages` / `operating_frequency_hz`, which
        # curve_position reads). A blanket overwrite here would wipe them.
        resolved = dict(record.get("resolved_inputs") or {})
        for _k in ("depth_override", "sg_oil_override", "sg_water_override"):
            resolved.pop(_k, None)
        resolved.update(overrides)
        record["resolved_inputs"] = resolved
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

        # --- curve_position scaling inputs (M4 / 4b) --------------------------
        # Co-located with the pick: the pump's identity + its stage count + its
        # operating frequency are one physical act. Both ride ``resolved_inputs``
        # under the CANONICAL keys curve_position reads — ``stages`` and
        # ``operating_frequency_hz`` (NOT the ``frequency_hz`` alias, NOT
        # ``session['pump']``, NOT a model argument) — the same setup-injection path
        # as depth/SG. Guardrail 4: stages (section count) ≠ operating frequency
        # (the pump's running Hz, seeded from telemetry — not line frequency).
        st.markdown("**Curve scaling** — stage count & operating frequency (feeds `curve_position`)")
        sc1, sc2 = st.columns(2)
        # stages: NO default — blank until entered. A pre-filled stage count is the
        # fabrication the block exists to prevent (it silently corrupts curve scaling),
        # so absent must stay reachable → the honest ``stages_absent`` block.
        stages_val = sc1.number_input(
            "Pump stages",
            min_value=1,
            value=None,
            step=1,
            format="%d",
            placeholder="enter stage count",
            key=f"pump_stages_{record['well_id']}",
            help="Number of pump stages/sections — no honest default; the per-stage "
            "curve can't be scaled to the well without it. Distinct from the running Hz.",
        )
        # operating_frequency_hz: seeded from the telemetry frequency the pump is
        # actually running at (the physically-correct scaling target), overridable.
        hz_seed = availability.get("telemetry_motor_frequency_hz")
        hz_val = sc2.number_input(
            "Operating frequency (Hz)",
            min_value=0.0,
            value=float(hz_seed) if hz_seed else None,
            step=0.5,
            placeholder="enter operating Hz",
            key=f"pump_opfreq_{record['well_id']}",
            help="The Hz the pump is running at (seeded from telemetry "
            "motor_frequency_hz) — the ideal curve is scaled to this. Not line frequency.",
        )

        # Persist the two CANONICAL keys, preserving keys owned by other setup steps
        # (depth/SG overrides). Blank stages → drop the key so ``stages_absent`` fires.
        resolved = dict(record.get("resolved_inputs") or {})
        if stages_val is not None and int(stages_val) >= 1:
            resolved["stages"] = int(stages_val)
        else:
            resolved.pop("stages", None)
        if hz_val is not None and float(hz_val) > 0:
            resolved["operating_frequency_hz"] = float(hz_val)
        else:
            resolved.pop("operating_frequency_hz", None)
        record["resolved_inputs"] = resolved
        session.save_session(record)

        # Validation — WARN only (a real reading may sit near the edge); never hard-block.
        if stages_val is None:
            st.warning(
                "Enter the pump stage count — `curve_position` blocks "
                "(`stages_absent`) until it's set."
            )
        if hz_val is not None and not (40.0 <= float(hz_val) <= 80.0):
            st.warning(
                f"Operating frequency {float(hz_val):.1f} Hz is outside the typical ESP "
                "range (~40–80 Hz) — double-check the value (not hard-blocked)."
            )
        _seed_note = f" · seeded from telemetry **{hz_seed:.1f} Hz**" if hz_seed else ""
        st.caption(
            "`curve_position` will scale the ideal curve to "
            + (f"**{int(stages_val)} stages**" if stages_val is not None else "**— stages**")
            + " at "
            + (f"**{float(hz_val):.1f} Hz**" if hz_val is not None else "**— Hz**")
            + _seed_note
            + f" · keys → `resolved_inputs['stages']`, `resolved_inputs['operating_frequency_hz']`"
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
        # Dev panel default comes from config (CURVE_DEV_PANEL): OFF for the company
        # copy, on when the flag is set. Still toggleable per-session via the checkbox.
        dev_mode = st.checkbox(
            "Developer mode (expose the loop)", value=config.DEV_PANEL_DEFAULT
        )

        # Dependent org → well dropdowns, enumerated (cached) from the VE
        # well-configuration table. Enumeration failure or an empty result falls back
        # to manual entry with a visible notice — the sidebar never crashes.
        org_well_map: Dict[str, List[str]] = {}
        enum_error: Optional[str] = None
        try:
            org_well_map = _enumerate_org_wells(profile)
        except Exception as exc:  # surface auth/data errors, don't crash the sidebar
            enum_error = str(exc)

        if st.button("↻ Refresh well list"):
            _enumerate_org_wells.clear()
            st.rerun()

        if enum_error:
            st.warning(
                f"Could not load the well list ({enum_error}). Falling back to manual entry."
            )
            org_choice = MANUAL_ENTRY
        elif not org_well_map:
            st.info("No wells enumerated from the catalog. Falling back to manual entry.")
            org_choice = MANUAL_ENTRY
        else:
            org_options = sorted(org_well_map) + [MANUAL_ENTRY]
            org_choice = st.selectbox("Organization", org_options, index=0)

        if org_choice == MANUAL_ENTRY:
            organization_id = st.text_input("organization_id")
            well_id = st.text_input("well_id")
        else:
            organization_id = org_choice
            wells = org_well_map.get(org_choice, [])
            if wells:
                well_id = st.selectbox("Well", wells, index=0)
            else:
                well_id = ""
                st.info("No wells for this organization.")

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
                            # M4 / 4b: seed for the operating-frequency field (pump-pick).
                            "telemetry_motor_frequency_hz": probe.get(
                                "telemetry_motor_frequency_hz"
                            ),
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
        "recommendation (comparison, affinity check, energy efficiency)… "
        "— or prefix /sql for a direct data lookup"
    )
    if question:
        st.session_state["chat"].append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            # /sql (M4a): clear M3's full-result stashes BEFORE the turn, so whatever is
            # there afterwards belongs to THIS turn (a cap-exceeded turn writes nothing
            # and must not inherit the previous query's rows).
            take_sql_stashes(record)
            turn_id = st.session_state.get(SQL_TURN_SEQ_KEY, 0) + 1
            st.session_state[SQL_TURN_SEQ_KEY] = turn_id
            with st.spinner("Running the CurVE loop…"):
                started = time.time()
                result = run_curve_turn(
                    question, session=record, profile_name=profile
                )
                elapsed = time.time() - started
            # /sql (M4a): move this turn's full results out of the session-record stash
            # and into per-turn session state — ONE payload per execution. Keying by
            # turn_id fixes the cross-turn collision (message 1 keeps serving message 1's
            # rows); carrying a list fixes the intra-turn one (a turn that runs sql_query
            # twice keeps both results instead of the second clobbering the first).
            # Encoded once here; a download click just replays the bytes.
            sql_downloads = build_sql_downloads(
                take_sql_stashes(record), record.get("well_id", ""), turn_id
            )
            if sql_downloads:
                st.session_state.setdefault(SQL_RESULTS_STATE_KEY, {})[
                    turn_id
                ] = sql_downloads
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
                # Per-message handle into SQL_RESULTS_STATE_KEY (None on non-/sql turns).
                "sql_turn_id": turn_id,
            }
            st.session_state["chat"].append(entry)
            _render_answer(entry, dev_mode)


if __name__ == "__main__":
    render_page()
