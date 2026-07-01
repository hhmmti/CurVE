"""CurVE M4 / 4b — the ``curve_position`` tool, run with NO AWS credentials.

Covered:
  * The two HARD BLOCKS fire honestly off the injected pick state: no pump picked →
    blocked ``pump_connection_unresolved``; picked pump with null curve coeffs →
    blocked ``pump_curve_coeffs_absent`` (read, not re-derived).
  * ΔP-missing (zero PIP coverage) surfaces as the inherited not-ready, not a new block.
  * Stage count + Hz are sourced from ``resolved_inputs`` (NOT well_configuration);
    absent → not-ready, never a silent default.
  * Happy path on a valid pick returns the position ``values`` + a figure, with trust =
    Estimated ⊓ ΔP-tier (weakest-wins) — never Validated for a catalog overlay.
  * Obsolete pick → the obsolete flag is carried while the curve + position still return
    with an Estimated label (no block, no downgrade).
  * The pick / stages / Hz are NOT in the tool inputSchema; the figure is stripped from
    the model-facing result.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curve import curve_position as cp_layer  # noqa: E402
from curve import data, ideal_catalog, tools, well_depth  # noqa: E402

# Demo-well org mapping (corrected): HACKBERRY SPRINGS 3BH → usedc. Org is setup-injected
# — never hardcode permian_resources in a fixture.
ORG = "usedc"
WELL = "HACKBERRY SPRINGS 3BH"


# --- fixtures -----------------------------------------------------------------


def _coeffs(prefix: str) -> dict:
    # Linear-ish head that decreases with flow; positive at the operating flow band.
    if prefix == "head":
        return {"ideal_head_c1": 60.0, "ideal_head_c2": -0.02,
                "ideal_head_c3": 0.0, "ideal_head_c4": 0.0,
                "ideal_head_c5": 0.0, "ideal_head_c6": 0.0}
    return {f"ideal_{prefix}_c1": 10.0, "ideal_power_c2": 0.01,
            "ideal_power_c3": 0.0, "ideal_power_c4": 0.0,
            "ideal_power_c5": 0.0, "ideal_power_c6": 0.0}


def _catalog() -> pd.DataFrame:
    base = {
        "manufacturer": "ChampionX", "series": "400",
        "bep_bpd": 1000.0, "min_recommended_bpd": 800.0, "max_recommended_bpd": 1300.0,
        "max_plotted_bpd": 1600.0,
    }
    rows = []
    # CX-A: valid, selectable, not obsolete.
    a = {"pump_id": "CX-A", "esp_model": "P6-1000", "is_obsolete": "false", **base}
    a.update(_coeffs("head")); a.update(_coeffs("power")); rows.append(a)
    # CX-OBS: valid + selectable but OBSOLETE.
    o = {"pump_id": "CX-OBS", "esp_model": "P6-1000-OBS", "is_obsolete": "true", **base}
    o.update(_coeffs("head")); o.update(_coeffs("power")); rows.append(o)
    return pd.DataFrame(rows)


def _telemetry(pip=800.0) -> pd.DataFrame:
    days = pd.date_range("2026-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "organization_id": ORG, "well_id": WELL,
        "observation_day": days.strftime("%Y-%m-%d"),
        "motor_frequency_hz": [58.0] * 30, "tubing_pressure_psi": 250.0,
        "pump_intake_pressure_psi": [pip] * 30, "motor_amps": 40.0, "motor_volts": 2300.0,
    })


def _production() -> pd.DataFrame:
    days = pd.date_range("2026-01-01", periods=30, freq="D")
    # liquid = oil + water = 700 + 300 = 1000 bpd (median) → near BEP 1000.
    return pd.DataFrame({
        "organization_id": ORG, "well_id": WELL,
        "observation_day": days.strftime("%Y-%m-%d"),
        "alloc_oil_vol": [700.0] * 30, "alloc_water_vol": [300.0] * 30,
        "alloc_gas_vol": [450.0] * 30,
    })


_RESOLVED = {"stages": 175, "operating_frequency_hz": 58.0}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Monkeypatch every AWS read so the tool runs cred-free on mocked rows."""
    monkeypatch.setattr(ideal_catalog, "fetch_ideal_catalog", lambda *a, **k: _catalog())
    monkeypatch.setattr(data, "fetch_preprocessed_window",
                        lambda *a, **k: (_telemetry(), _production()))
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: 9100.0)


def _pick(catalog, pump_id):
    return ideal_catalog.make_pump_pick(catalog, pump_id, bep_tolerance=0.25)


def _input(pump, resolved=None):
    return {
        "organization_id": ORG, "well_id": WELL,
        "pump": pump, "resolved_inputs": resolved if resolved is not None else dict(_RESOLVED),
    }


# --- hard blocks (read off the injected pick, not re-derived) ------------------


def test_block_no_pump_picked():
    env = tools.curve_position(_input(pump=None))
    assert env["status"] == "blocked"
    assert any("pump_connection_unresolved" in f for f in env["flags"])
    assert env["trust_label"] is None and env["figure"] is None


def test_block_pick_with_null_curve_coeffs():
    # A pick flagged not-selectable (null coeffs) blocks with the coverage-gap reason.
    pick = _pick(_catalog(), "CX-A")
    pick["selectable"] = False
    env = tools.curve_position(_input(pick))
    assert env["status"] == "blocked"
    assert any("pump_curve_coeffs_absent" in f for f in env["flags"])


# --- scaling inputs are setup-injected, never defaulted -----------------------


def test_block_when_stages_or_hz_absent():
    env = tools.curve_position(_input(_pick(_catalog(), "CX-A"), resolved={}))
    assert env["status"] == "blocked"
    assert "stages_absent" in env["flags"] and "operating_frequency_absent" in env["flags"]


def test_scaling_inputs_read_from_resolved_inputs_not_config():
    s = cp_layer.resolve_scaling_inputs({"stages": 200, "operating_frequency_hz": 55.0})
    assert s.ready and s.stages == 200 and s.frequency_hz == 55.0
    assert cp_layer.resolve_scaling_inputs({"frequency_hz": 50.0}).stages is None  # alias ok


# --- inherited ΔP not-ready (zero PIP coverage) -------------------------------


def test_zero_pip_coverage_is_inherited_not_ready(monkeypatch):
    monkeypatch.setattr(data, "fetch_preprocessed_window",
                        lambda *a, **k: (_telemetry(pip=0.0), _production()))
    env = tools.curve_position(_input(_pick(_catalog(), "CX-A")))
    assert env["status"] == "blocked"
    assert env["flags"] == ["pip_coverage_zero"]  # the ΔP layer's block, not a new one


# --- happy path ---------------------------------------------------------------


def test_happy_path_returns_position_and_figures():
    env = tools.curve_position(_input(_pick(_catalog(), "CX-A")))
    # curve_position carries a FIGURES LIST (overlay, family) — not the singular figure
    # slot the other tools use. Model-facing parity still holds (all figure keys stripped).
    assert set(env) == {"status", "values", "trust_label", "flags", "figures"}
    assert env["status"] == "available"
    # Estimated ⊓ ΔP-tier (depth from rrc but SG defaulted → Estimated; catalog model
    # is always Estimated) → overall Estimated, never Validated for a catalog overlay.
    assert env["trust_label"] == "Estimated"
    # Two figures, in order: single-frequency overlay, then the affinity family sweep.
    assert isinstance(env["figures"], list) and len(env["figures"]) == 2
    assert all(f is not None for f in env["figures"])

    v = env["values"]
    op = v["operating_point"]
    assert op["flow_bpd"] == 1000.0          # liquid/total fluid (oil+water), NOT oil 700
    assert op["delta_p_pump_psi"] > 0
    # Variance from design + BEP position are present (the headline fields).
    assert "variance_from_design" in v and v["variance_from_design"]["delta_p_psi"] is not None
    assert v["bep"]["bep_bpd"] == 1000.0
    assert v["bep"]["pct_of_bep"] == 100.0   # 1000 / 1000
    assert v["bep"]["in_recommended_window"] is True   # 1000 in [800, 1300]
    assert v["scaling"]["stages"] == 175 and v["scaling"]["operating_frequency_hz"] == 58.0


def test_obsolete_pick_flagged_not_blocked_not_downgraded():
    env = tools.curve_position(_input(_pick(_catalog(), "CX-OBS")))
    assert env["status"] == "available"          # NOT blocked on obsolete
    assert env["trust_label"] == "Estimated"     # NOT downgraded on obsolete
    assert "picked_pump_obsolete" in env["flags"]
    assert env["values"]["pump"]["is_obsolete"] is True


# --- spec hygiene: pick / stages / Hz are not model-facing args ---------------


def test_curve_position_spec_exposes_no_model_args():
    schema = tools._CURVE_POSITION_SPEC["toolSpec"]["inputSchema"]["json"]
    assert schema["properties"] == {} and schema["required"] == []
