"""M3 routing re-test (live Bedrock model routing; data isolated to local fixtures).

Runs the three Requirement-#6 cases through the SHIPPED loop with the FULL connection-
free roster registered. The MODEL does the real routing; the data layer is monkeypatched
to local fixtures so we test routing (and the absence-block narration) without depending
on a specific live well's data. Run from the repo root with AWS creds:

    AWS_PROFILE=roam-ai python routing_retest.py
"""

import json

import pandas as pd

from curve import data, ideal_catalog, recommendations, session, well_depth
from curve.engine import run_curve_turn

ORG, WELL = "org-demo", "W-12"

_CURRENT = "{motor_frequency_hz=58.0, tubing_pressure_psi=240.0, production={oil=300.0, water=100.0, gas=450.0}}"
_MODEL = "{max_oil={motor_frequency_hz=62.0, tubing_pressure_psi=260.0, production={oil=340.0, water=110.0, gas=480.0}}}"
_SUMMARY = json.dumps({
    "pump_intake_pressure_psi_1d_avg": 800.0, "tubing_pressure_psi_1d_avg": 240.0,
    "alloc_oil_vol_1d_ago": 300.0, "alloc_water_vol_1d_ago": 100.0,
    "motor_amps_1d_avg": 40.0, "motor_volts_1d_avg": 2300.0,
})


def _rec_row():
    return {
        "organization_id": ORG, "well_id": WELL, "uuid": "rec-1",
        "timestamp": "2026-06-01", "current_setpoint": _CURRENT,
        "model_setpoint_recommendations": _MODEL, "summary_data_json": _SUMMARY,
    }


def _telemetry():
    days = pd.date_range("2026-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "organization_id": ORG, "well_id": WELL,
        "observation_day": days.strftime("%Y-%m-%d"),
        "motor_frequency_hz": [55.0] * 30, "tubing_pressure_psi": 250.0,
        "pump_intake_pressure_psi": 800.0, "motor_amps": 40.0, "motor_volts": 2300.0,
    })


def _production():
    days = pd.date_range("2026-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "organization_id": ORG, "well_id": WELL,
        "observation_day": days.strftime("%Y-%m-%d"),
        "alloc_oil_vol": [300.0] * 30, "alloc_water_vol": [100.0] * 30,
        "alloc_gas_vol": [450.0] * 30,
    })


def _ideal_catalog() -> pd.DataFrame:
    """One BEP-compatible (median liquid = 400 bpd) selectable ChampionX pump."""
    row = {
        "pump_id": "CX-DEMO", "esp_model": "P6-400", "manufacturer": "ChampionX",
        "series": "400", "bep_bpd": 400.0, "min_recommended_bpd": 250.0,
        "max_recommended_bpd": 600.0, "max_plotted_bpd": 800.0, "is_obsolete": "false",
    }
    row.update({"ideal_head_c1": 60.0, "ideal_head_c2": -0.04, "ideal_head_c3": 0.0,
                "ideal_head_c4": 0.0, "ideal_head_c5": 0.0, "ideal_head_c6": 0.0})
    row.update({f"ideal_power_c{i}": (10.0 if i == 1 else 0.0) for i in range(1, 7)})
    return pd.DataFrame([row])


def _isolate_data(rec_present: bool):
    data.fetch_preprocessed_window = lambda *a, **k: (_telemetry(), _production())
    well_depth.fetch_well_depth_ft = lambda *a, **k: 9100.0
    recommendations.fetch_latest_recommendation = lambda *a, **k: (_rec_row() if rec_present else None)
    ideal_catalog.fetch_ideal_catalog = lambda *a, **k: _ideal_catalog()


def _run(label, question, rec_present, pump=None, resolved_inputs=None):
    _isolate_data(rec_present)
    rec = session.new_session_record(
        "sess", ORG, WELL, resolved_inputs=resolved_inputs or {}, pump=pump
    )
    result = run_curve_turn(question, session=rec, profile_name="roam-ai")
    print(f"\n=== {label} ===")
    print(f"Q: {question}")
    print(f"tool_trace : {result['tool_trace']}")
    statuses = [
        f"{o['name']}={o['result'].get('status') if isinstance(o['result'], dict) else '?'}"
        for o in result.get("tool_outputs", [])
    ]
    print(f"tool status: {statuses}")
    print(f"answer     : {result['text'][:400]}")
    return result


if __name__ == "__main__":
    # 1) true-neither — equipment/nameplate; no tool covers it → should fire NONE.
    _run(
        "true-neither (equipment/nameplate)",
        "What is the motor nameplate horsepower and the pump model installed on this well?",
        rec_present=True,
    )
    # 2) Scenario-3 — recommendation absent; should route to a rec tool and hit the
    #    absence-block cleanly, narrating that no recommendation is available (no synth).
    _run(
        "Scenario-3 (frequency-change recommendation, NO rec on record)",
        "What frequency change do you recommend for this well?",
        rec_present=False,
    )
    # 3) adjacent-physics — affinity is now real; confirm no misroute among
    #    production / water-cut / affinity.
    for q in (
        "How has this well been producing recently?",
        "Is this well watering up — how have water cut and GOR changed?",
        "Does the recommended frequency change obey the pump affinity laws?",
    ):
        _run("adjacent-physics", q, rec_present=True)

    # 4) M4 (4b) — pump-curve / variance-from-design questions must route to
    #    curve_position. A pump is picked + stage count / Hz injected on the session, so
    #    the tool runs end-to-end and returns an Estimated overlay (catalog ⊓ ΔP-tier).
    _pump = ideal_catalog.make_pump_pick(_ideal_catalog(), "CX-DEMO", bep_tolerance=0.25)
    _scaling = {"stages": 175, "operating_frequency_hz": 58.0}
    for q in (
        "Where is this well operating on its pump curve right now?",
        "How far off design is this pump — what's the variance from the ideal curve?",
        "Is the pump running near its best efficiency point?",
    ):
        _run("M4 curve_position", q, rec_present=True, pump=_pump, resolved_inputs=_scaling)
