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
from curve import data, session
from curve.engine import run_curve_turn
from curve.gate import run_tool_gate
from curve.tools import NON_MODEL_RESULT_KEYS

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
def _availability_probe(organization_id: str, well_id: str) -> Dict[str, Any]:
    """Front-load gate: fetch once + run the SHIPPED gate to build the availability report.

    Same gate logic invoked at setup as before compute (CurVE-decisions §4 F1).
    Also records the available data range so the chat turn's relative-window math is
    anchored. Cached per (org, well) so it doesn't re-run on every Streamlit rerun.
    """
    telemetry_df, production_df = data.fetch_preprocessed_window(
        organization_id, well_id
    )
    return {
        "gate": run_tool_gate("production_history", telemetry_df, production_df),
        "coverage": _coverage(telemetry_df, production_df),
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


def _render_answer(entry: Dict[str, Any], dev_mode: bool) -> None:
    """Render one assistant turn: badge + KPI + figure + narration (+ dev panel)."""
    envelope = entry.get("primary_envelope")

    if envelope is not None:
        trust = envelope.get("trust_label")
        status = envelope.get("status")
        if status == "available":
            st.markdown(_trust_badge(trust), unsafe_allow_html=True)
            _render_kpis(envelope.get("values"))
            figure = envelope.get("figure")
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True)
        else:
            st.warning(
                f"`production_history` is **{status}** — "
                f"{', '.join(envelope.get('flags') or []) or 'no detail'}."
            )

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


# --- page ---------------------------------------------------------------------


def render_page() -> None:
    st.set_page_config(page_title="CurVE — Physics Validation", page_icon="🛢️", layout="wide")
    _bind_session_store()

    st.title("🛢️ CurVE — Virtual Engineer physics validation")
    st.caption("M2 demo · production history, real telemetry + production, live trust label")

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
                        availability={
                            "production_history": probe["gate"],
                            "coverage": probe["coverage"],
                        },
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

    gate = (record.get("availability") or {}).get("production_history", {})
    coverage = (record.get("availability") or {}).get("coverage") or {}
    available = gate.get("status") == "available"
    with st.container():
        st.subheader(f"Well {record['well_id']} — availability")
        cols = st.columns([1, 3])
        cols[0].markdown(_trust_badge(gate.get("trust_label")), unsafe_allow_html=True)
        cols[1].markdown(
            f"**production_history** → `{gate.get('status')}`"
            + (f" · flags: {gate.get('flags')}" if gate.get("flags") else "")
        )
        if coverage.get("min_day") and coverage.get("max_day"):
            st.caption(
                f"Data available: {coverage['min_day']} → {coverage['max_day']}"
            )
    if not available:
        st.warning(
            "Telemetry + production not both present for this well — "
            "`production_history` is unavailable. Pick another well."
        )
        return

    # --- chat -----------------------------------------------------------------
    st.divider()
    for entry in st.session_state.get("chat", []):
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.markdown(entry["text"])
            else:
                _render_answer(entry, dev_mode)

    question = st.chat_input("Ask about this well's production history…")
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
            # Pull the production_history envelope (single-tool shape) for rendering.
            primary = None
            for output in result.get("tool_outputs", []):
                if output.get("name") == "production_history":
                    primary = output.get("result")
                    break
            entry = {
                "role": "assistant",
                "narration": result.get("text"),
                "primary_envelope": primary,
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
