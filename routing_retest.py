"""M3 routing re-test (live Bedrock model routing; data isolated to local fixtures).

Runs the three Requirement-#6 cases through the SHIPPED loop with the FULL connection-
free roster registered. The MODEL does the real routing; the data layer is monkeypatched
to local fixtures so we test routing (and the absence-block narration) without depending
on a specific live well's data. Run from the repo root with AWS creds:

    AWS_PROFILE=roam-ai python routing_retest.py
"""

import json

import pandas as pd

from curve import data, recommendations, session, well_depth
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


def _isolate_data(rec_present: bool):
    data.fetch_preprocessed_window = lambda *a, **k: (_telemetry(), _production())
    well_depth.fetch_well_depth_ft = lambda *a, **k: 9100.0
    recommendations.fetch_latest_recommendation = lambda *a, **k: (_rec_row() if rec_present else None)


def _run(label, question, rec_present):
    _isolate_data(rec_present)
    rec = session.new_session_record("sess", ORG, WELL, resolved_inputs={})
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
