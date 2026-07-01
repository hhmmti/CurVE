"""CurVE operating-point-on-the-curve layer — connection ⊓ ΔP (M4 / 4b).

This is the layer that turns the 4a pump connection + the M3 ΔP inputs into the
**operating point on the well-scaled ideal curve** — the "where am I on the pump
curve, and how far off design am I" that ``curve_position`` answers. It is the FIRST
surface that is BOTH connection-dependent (needs the picked pump's ideal curve) AND
ΔP-input-dependent (the operating point's head comes from the M3 ``delta_p_inputs``
layer). It is pure orchestration over the **vendored** ``compute`` — it does NOT
re-author the polynomial eval, the per-stage→well scaling, or the BEP diagnostic.

WHERE THE CONSTRUCTION CAME FROM (resolved from the week-8 physics app, not invented)
------------------------------------------------------------------------------------
The real-vs-ideal overlay is the app's *Single Frequency* tab + the *BEP / Operating
Range* tab. Ported verbatim:

  * Curve reconstruction + per-stage→well scaling —
    ``compute.ideal_curve_overlay.build_ideal_curve_for_frequency(pump_row,
    frequency_hz, stages, sg_for_dp)``. The head poly is evaluated at 60 Hz then
    scaled by the affinity laws AND the stage count: ``head = head_60 *
    (f/60)**2 * stages``; flow ``q = q0 * (f/60)``; power ``bhp_60 * (f/60)**3 *
    stages``. So the reconstructed per-stage curve becomes the WELL-scaled curve only
    after multiplying by ``stages`` — both ``stages`` and the operating ``frequency_hz``
    are needed, and in v1 BOTH are setup-injected manual inputs on ``resolved_inputs``
    (NOT read from ``well_configuration`` — that is v2).

  * Operating flow Q — ``median(liquid_rate_bbl_day)`` (the LIQUID/total-fluid rate,
    oil + water; NOT the oil rate — guardrail 4). Same figure 4a BEP-narrows on
    (:func:`curve.ideal_catalog.resolve_total_fluid_bpd`), reused here so the operating
    point and the candidate narrowing agree.

  * Operating head — ΔP_pump from the M3 ``delta_p_inputs`` layer (depth/SG/PIP).
    The representative operating ΔP is ``median(delta_p_pump_psi)`` over the
    PIP-present rows (measured-or-missing; PIP is never proxied). Converted to head
    with the mixture SG via ``compute.physics_common.calc_head_ft_from_pressure_psi``.

  * BEP position — ``compute.ideal_curve_overlay.compute_bep_position_diagnostic``
    (the app's *BEP / Operating Range* compute): %-distance from BEP, the Near/
    Acceptable/Far band, and the Inside/Below/Above recommended-window status. We have
    a single operating point (no recommendation here — ``curve_position`` is connection
    ⊓ ΔP, not recommendation-dependent), so the operating Q is passed as both the
    "current" and "recommended" inputs and only the current point is surfaced.

GUARDRAIL-4 PINS (do not conflate)
----------------------------------
  * operating flow = ``liquid_rate_bbl_day`` (total fluid), NOT oil rate.
  * the reconstructed curve + operating ΔP are WELL-total (per-stage × ``stages``),
    NOT per-stage head.
  * ``bep_bpd`` (best-efficiency flow) is DISTINCT from
    ``min/max_recommended_bpd`` (the recommended operating window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compute import ideal_curve_overlay
from compute.physics_common import calc_head_ft_from_pressure_psi

# Keys the operator's setup injects onto ``resolved_inputs`` for the curve scaling.
# Both are MANUAL v1 inputs (NOT well_configuration). ``operating_frequency_hz`` has a
# ``frequency_hz`` alias so an older setup key still resolves.
STAGES_KEY = "stages"
FREQUENCY_HZ_KEYS = ("operating_frequency_hz", "frequency_hz")

# The affinity frequency set for the family sweep — ported VERBATIM from the app's
# ``ideal_curve_service.build_ideal_payload`` (its ``family_freqs``). Not invented; this
# is the exact set the week-8 *Multi Frequency* tab fans across. The family sweeps
# FREQUENCY only — it never re-fabricates the stage count (guardrail 4: stages ≠ Hz).
FAMILY_FREQUENCIES = [45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]


def _num(value: Any) -> Optional[float]:
    """Parse to a finite float, else None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


@dataclass(frozen=True)
class ScalingInputs:
    """The setup-injected per-stage→well scaling inputs + their resolution flags.

    ``stages`` and ``frequency_hz`` are both manual v1 inputs riding ``resolved_inputs``
    (alongside depth/SG, the pump pick, and ``bep_tolerance``). When either is absent we
    do NOT assume a default — the tool surfaces a not-ready (you cannot scale a per-stage
    curve to the well without the stage count, nor place it without the operating Hz).
    """

    stages: Optional[int]
    frequency_hz: Optional[float]
    flags: List[str]

    @property
    def ready(self) -> bool:
        return self.stages is not None and self.frequency_hz is not None

    def as_values(self) -> Dict[str, Any]:
        return {
            "stages": self.stages,
            "operating_frequency_hz": self.frequency_hz,
            "source": "setup_injected_manual",  # v1: manual; NOT well_configuration (v2)
        }


def resolve_scaling_inputs(resolved_inputs: Optional[Dict[str, Any]]) -> ScalingInputs:
    """Read stage count + operating Hz off the setup-injected ``resolved_inputs``.

    Same path depth/SG (``delta_p_inputs``), the pump pick, and ``bep_tolerance`` ride —
    NOT a model-facing Converse arg, NOT ``well_configuration``. Missing/invalid → None
    (a not-ready surfaced honestly, never a silent default stage count or frequency).
    """
    ctx = resolved_inputs or {}
    stages_f = _num(ctx.get(STAGES_KEY))
    stages = int(stages_f) if (stages_f is not None and stages_f >= 1) else None

    freq = None
    for key in FREQUENCY_HZ_KEYS:
        freq = _num(ctx.get(key))
        if freq is not None and freq > 0:
            break
        freq = None

    flags: List[str] = []
    if stages is None:
        flags.append("stages_absent")
    if freq is None:
        flags.append("operating_frequency_absent")
    return ScalingInputs(stages=stages, frequency_hz=freq, flags=flags)


def lookup_pump_row(catalog: pd.DataFrame, pump_id: str) -> Optional[pd.Series]:
    """The full catalog row (with curve coeffs) for the picked ``pump_id``, or None.

    The injected ``session['pump']`` carries only the SHAPED pick (id/label/flow/flags),
    not the polynomial coeffs — the curve reconstruction needs the row, so we re-read it
    from the catalog by the pick's ``pump_id`` key. None when the key is absent (the tool
    turns that into an error envelope; never a substituted pump).
    """
    if catalog is None or "pump_id" not in getattr(catalog, "columns", []):
        return None
    match = catalog[catalog["pump_id"].astype(str).str.strip() == str(pump_id).strip()]
    return None if match.empty else match.iloc[0]


def representative_sg(present_rows: pd.DataFrame) -> float:
    """Representative mixture SG for the ΔP↔head conversion (median ``sg_mixture``).

    The vendored preprocessed compute writes a per-row ``sg_mixture`` (from water cut);
    we take its median over the PIP-present rows so the curve's ΔP axis and the operating
    point's head conversion use the SAME SG. Falls back to 1.0 (water) when absent.
    """
    if present_rows is None or "sg_mixture" not in getattr(present_rows, "columns", []):
        return 1.0
    series = pd.to_numeric(present_rows["sg_mixture"], errors="coerce").dropna()
    sg = float(series.median()) if not series.empty else 1.0
    return sg if sg > 0 else 1.0


def build_operating_point(
    analyzed: pd.DataFrame,
    operating_flow_bpd: Optional[float],
    present_rows: pd.DataFrame,
    sg_for_dp: float,
) -> Optional[Dict[str, Any]]:
    """The well's representative operating point: (flow, ΔP_pump, head).

    flow = the 4a total-fluid figure (median liquid rate, passed in so the operating
    point and the BEP narrowing agree); ΔP = median ``delta_p_pump_psi`` over the
    PIP-present rows (measured-or-missing); head = that ΔP converted with the mixture SG.
    Returns None when either flow or ΔP can't be formed (→ inherited not-ready).
    """
    if operating_flow_bpd is None or operating_flow_bpd <= 0:
        return None
    if present_rows is None or len(present_rows) == 0:
        return None
    dp = pd.to_numeric(present_rows.get("delta_p_pump_psi"), errors="coerce").dropna()
    if dp.empty:
        return None
    operating_dp = float(dp.median())
    if operating_dp <= 0:
        return None
    operating_head = calc_head_ft_from_pressure_psi(operating_dp, sg=sg_for_dp)
    return {
        "flow_bpd": float(operating_flow_bpd),       # total fluid (liquid), NOT oil
        "delta_p_pump_psi": operating_dp,            # well-total ΔP, NOT per-stage
        "head_ft": float(operating_head) if operating_head is not None else None,
    }


def evaluate_position(
    pump_row: pd.Series,
    pump_pick: Dict[str, Any],
    scaling: ScalingInputs,
    operating_point: Dict[str, Any],
    sg_for_dp: float,
) -> Tuple[pd.DataFrame, Dict[float, pd.DataFrame], Dict[str, Any]]:
    """Reconstruct the well-scaled curve(s) + compute the position values at the op point.

    Returns ``(curve_df, family_curves, position_values)``. ``curve_df`` is the vendored
    well-scaled ideal curve at the operating Hz (flow_bpd / head_ft / delta_p_psi /
    bhp_hp); ``family_curves`` is the same reconstruction across :data:`FAMILY_FREQUENCIES`
    (the affinity fan) — both for the UI figures only. ``position_values`` is the proposed
    model-facing field set (see module docstring).
    """
    q = operating_point["flow_bpd"]
    op_dp = operating_point["delta_p_pump_psi"]
    op_head = operating_point["head_ft"]

    # WELL-scaled curve via the vendored per-stage→well scaling (affinity × stages).
    curve = ideal_curve_overlay.build_ideal_curve_for_frequency(
        pump_row,
        frequency_hz=scaling.frequency_hz,
        stages=scaling.stages,
        sg_for_dp=sg_for_dp,
    )

    # The affinity family — the SAME stage count + SG, swept across the app's frequency
    # set (frequency only; stages is not re-fabricated). Vendored reconstruction, reused.
    family_curves = ideal_curve_overlay.build_multi_frequency_curves(
        pump_row,
        stages=scaling.stages,
        frequencies=FAMILY_FREQUENCIES,
        sg_for_dp=sg_for_dp,
    )

    # Ideal curve value AT the operating flow (the design target at this Q). Interpolate
    # the vendored curve grid — no hand-rolled poly eval.
    xp = curve["flow_bpd"].to_numpy(dtype=float)
    ideal_dp_at_q = float(np.interp(q, xp, curve["delta_p_psi"].to_numpy(dtype=float)))
    ideal_head_at_q = float(np.interp(q, xp, curve["head_ft"].to_numpy(dtype=float)))

    # Variance from design — how far the measured operating point sits off the ideal
    # curve at the same flow (the headline Donny asked for). Reported in BOTH ΔP and head.
    var_dp = op_dp - ideal_dp_at_q
    var_head = (op_head - ideal_head_at_q) if op_head is not None else None
    var_pct = round(var_dp / ideal_dp_at_q * 100.0, 1) if ideal_dp_at_q else None

    # BEP position — the app's BEP/Operating-Range compute. Single operating point (no
    # recommendation), so Q is passed as both current+recommended; surface only current.
    bep = ideal_curve_overlay.compute_bep_position_diagnostic(
        current_flow_bpd=q,
        recommended_flow_bpd=q,
        bep_bpd=pump_pick.get("bep_bpd"),
        min_recommended_bpd=pump_pick.get("min_recommended_bpd"),
        max_recommended_bpd=pump_pick.get("max_recommended_bpd"),
        pump_label=pump_pick.get("esp_model"),
        pump_source=pump_pick.get("source"),
    )
    cur = bep.get("current") or {}
    bep_bpd = _num(pump_pick.get("bep_bpd"))
    pct_of_bep = round(q / bep_bpd * 100.0, 1) if bep_bpd else None
    range_status = cur.get("range_status")

    position = {
        "pump": {
            "pump_id": pump_pick.get("pump_id"),
            "esp_model": pump_pick.get("esp_model"),       # display label ≠ pump_id key
            "manufacturer": pump_pick.get("manufacturer"),
            "series": pump_pick.get("series"),
            "is_obsolete": bool(pump_pick.get("is_obsolete")),
        },
        "scaling": scaling.as_values(),
        "operating_point": {
            "flow_bpd": round(q, 1),                       # liquid/total fluid, NOT oil
            "delta_p_pump_psi": round(op_dp, 1),           # well-total, NOT per-stage
            "head_ft": round(op_head, 1) if op_head is not None else None,
        },
        "ideal_at_operating_flow": {
            "delta_p_psi": round(ideal_dp_at_q, 1),
            "head_ft": round(ideal_head_at_q, 1),
        },
        "variance_from_design": {
            "delta_p_psi": round(var_dp, 1),
            "head_ft": round(var_head, 1) if var_head is not None else None,
            "pct": var_pct,
        },
        "bep": {
            "bep_bpd": bep_bpd,                            # best-efficiency flow …
            "pct_of_bep": pct_of_bep,
            "distance_from_bep_pct": _round(cur.get("distance_from_bep_pct")),
            "bep_position_label": cur.get("bep_position_label"),  # Near/Acceptable/Far
            "min_recommended_bpd": _num(pump_pick.get("min_recommended_bpd")),  # … ≠ window
            "max_recommended_bpd": _num(pump_pick.get("max_recommended_bpd")),
            "range_status": range_status,                 # Inside/Below/Above window
            "in_recommended_window": range_status == "Inside recommended range",
        },
    }
    return curve, family_curves, position


def _round(value: Any, ndigits: int = 1) -> Optional[float]:
    f = _num(value)
    return round(f, ndigits) if f is not None else None
