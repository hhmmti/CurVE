"""CurVE shared ΔP input-resolution layer — depth + SG + PIP, with provenance.

Both ΔP history tools (``delta_p_frequency``, ``delta_p_composition``) sit on the
**same** ``delta_p_pump`` preprocessed compute and resolve the **same** three inputs.
This is the one layer that resolves them, by precedence, and emits the per-input
source flags + the trust-label decision that the gate threads to the surface. It is
the first place CurVE carries the **Estimated** label end-to-end, so the resolution
rule lives here, once, not duplicated per tool.

PRECEDENCE (prompt #4 core contract):
  * Depth (TVD): real from rrc → operator override → CurVE default (10,000 ft).
    The readiness surface pre-fills with the *no-override* resolution (rrc real if
    present, else the default); when the operator changes that pre-fill, the typed
    value is honored and supersedes — a hand-typed depth is a knowledgeable estimate,
    so the source becomes ``user_supplied`` (Estimated), never Validated.
  * SG: derived from measured per-well composition where available → operator
    override → defaults (0.85 oil / 1.00 water). The mixture SG itself is computed
    downstream from water cut by the vendored compute; this layer resolves the oil /
    water *endpoints* it needs. (Preprocessed telemetry carries no per-well SG, so in
    practice SG defaults unless the operator overrides — the ``measured`` tier exists
    but rarely fires on this data.)
  * PIP: measured-or-missing — NEVER defaulted, NEVER proxied. PIP is not resolved
    here; the vendored service nulls non-positive intake and ΔP is NaN on PIP-absent
    rows. Coverage is *read* from the computed frame by :func:`pip_coverage` (the #2
    intake-state truth — read, not re-derived; ``check_pump_intake_fallback_needed``
    is deliberately NOT called).

TRUST RULE (the precedent for every later Estimated tool):
  all inputs measured/real → ``Validated``; any input from a **default OR an operator
  override** → ``Estimated``. Missing PIP coverage / a missing x-axis input is NOT
  Estimated — that is not-ready/blocked, decided by the gate, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from compute.physics_common import DEFAULT_SG_OIL, DEFAULT_SG_WATER

# CurVE default depth — overrides the app's 5,000 ft (physics_input_contracts_v1.md
# "CurVE v1 deltas"). Labeled Estimated whenever it (or an override) is used.
CURVE_DEFAULT_DEPTH_FT: float = 10_000.0


@dataclass(frozen=True)
class DeltaPInputs:
    """Resolved ΔP inputs + provenance, ready to feed the vendored compute.

    ``flags`` names the source of every defaulted/overridden/real input unambiguously
    (``depth_from_rrc`` / ``depth_user_supplied`` / ``depth_defaulted``; same shape for
    SG). ``trust_label`` is ``Validated`` only when every input is measured/real.
    """

    depth_ft: float
    sg_oil: float
    sg_water: float
    depth_source: str  # "rrc" | "user_supplied" | "defaulted"
    sg_source: str  # "measured" | "user_supplied" | "defaulted"
    trust_label: str  # "Validated" | "Estimated"
    flags: List[str] = field(default_factory=list)

    def as_values(self) -> Dict[str, Any]:
        """Provenance block for the tool's model-facing ``values`` (for narration)."""
        return {
            "depth_ft": self.depth_ft,
            "depth_source": self.depth_source,
            "sg_oil": self.sg_oil,
            "sg_water": self.sg_water,
            "sg_source": self.sg_source,
        }


def _coerce_float(value: Any) -> Optional[float]:
    """Parse a possible override/real value to a positive float, else None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f if f > 0 else None


def resolve_delta_p_inputs(
    *,
    rrc_depth_ft: Optional[float] = None,
    depth_override: Optional[float] = None,
    sg_oil_measured: Optional[float] = None,
    sg_water_measured: Optional[float] = None,
    sg_oil_override: Optional[float] = None,
    sg_water_override: Optional[float] = None,
) -> DeltaPInputs:
    """Resolve depth + SG by precedence and return values + flags + trust label.

    Args mirror the precedence inputs: the real rrc depth (``None`` when no record),
    the operator overrides (setup-injected, not model args), and any measured per-well
    SG endpoints. Overrides supersede when supplied; otherwise real wins; otherwise the
    CurVE default. See module docstring for the full rule.
    """
    flags: List[str] = []

    # -- depth (TVD): override → rrc real → CurVE default ----------------------
    depth_ovr = _coerce_float(depth_override)
    rrc = _coerce_float(rrc_depth_ft)
    if depth_ovr is not None:
        depth_ft, depth_source = depth_ovr, "user_supplied"
        flags.append("depth_user_supplied")
    elif rrc is not None:
        depth_ft, depth_source = rrc, "rrc"
        flags.append("depth_from_rrc")
    else:
        depth_ft, depth_source = CURVE_DEFAULT_DEPTH_FT, "defaulted"
        flags.append("depth_defaulted")

    # -- SG endpoints: override → measured → defaults --------------------------
    sg_oil_ovr = _coerce_float(sg_oil_override)
    sg_water_ovr = _coerce_float(sg_water_override)
    sg_oil_meas = _coerce_float(sg_oil_measured)
    sg_water_meas = _coerce_float(sg_water_measured)

    if sg_oil_ovr is not None or sg_water_ovr is not None:
        sg_oil = sg_oil_ovr if sg_oil_ovr is not None else (sg_oil_meas or DEFAULT_SG_OIL)
        sg_water = (
            sg_water_ovr if sg_water_ovr is not None else (sg_water_meas or DEFAULT_SG_WATER)
        )
        sg_source = "user_supplied"
        flags.append("sg_user_supplied")
    elif sg_oil_meas is not None and sg_water_meas is not None:
        sg_oil, sg_water, sg_source = sg_oil_meas, sg_water_meas, "measured"
        flags.append("sg_from_measured")
    else:
        sg_oil, sg_water, sg_source = DEFAULT_SG_OIL, DEFAULT_SG_WATER, "defaulted"
        flags.append("sg_defaulted")

    # -- trust: Validated only when EVERY input is measured/real ---------------
    all_real = depth_source == "rrc" and sg_source == "measured"
    trust_label = "Validated" if all_real else "Estimated"

    return DeltaPInputs(
        depth_ft=depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
        depth_source=depth_source,
        sg_source=sg_source,
        trust_label=trust_label,
        flags=flags,
    )


def resolve_from_context(resolved_inputs: Optional[Dict[str, Any]], rrc_depth_ft: Optional[float]) -> DeltaPInputs:
    """Resolve from the engine-injected ``resolved_inputs`` context + a real rrc depth.

    ``resolved_inputs`` is the session record's operator-controlled override block
    (depth/SG overrides), injected per turn like org/well — NOT model-supplied. Keys
    read: ``depth_override``, ``sg_oil_override``, ``sg_water_override``,
    ``sg_oil_measured``, ``sg_water_measured``. Missing keys fall through to real/default.
    """
    ctx = resolved_inputs or {}
    return resolve_delta_p_inputs(
        rrc_depth_ft=rrc_depth_ft,
        depth_override=ctx.get("depth_override"),
        sg_oil_override=ctx.get("sg_oil_override"),
        sg_water_override=ctx.get("sg_water_override"),
        sg_oil_measured=ctx.get("sg_oil_measured"),
        sg_water_measured=ctx.get("sg_water_measured"),
    )


def default_prefill() -> Dict[str, Any]:
    """The values the readiness surface pre-fills when there is no real/override yet.

    Exactly what the tool would default to (CurVE default depth + default SG), so the
    operator sees and can accept-or-edit the real fallback rather than a silent
    assumption. (When a real rrc depth exists, the surface pre-fills that instead — see
    the Streamlit setup step.)
    """
    return {
        "depth_ft": CURVE_DEFAULT_DEPTH_FT,
        "sg_oil": DEFAULT_SG_OIL,
        "sg_water": DEFAULT_SG_WATER,
    }


# --- PIP coverage (read the #2 intake-state truth; do NOT re-derive) ----------


def pip_coverage(analyzed_df: pd.DataFrame) -> Dict[str, Any]:
    """Read PIP coverage from the computed frame — measured-or-missing already applied.

    The vendored ``run_preprocessed_analysis`` nulls non-positive intake and yields NaN
    ΔP on PIP-absent rows. We *read* that state (``pump_intake_pressure_psi`` present
    AND a finite ΔP) rather than re-deriving the missing rule. Returns the present /
    absent counts, the coverage fraction, and the boolean ``zero`` (hard-block) and
    ``partial`` (coverage flag) signals.
    """
    n_total = int(len(analyzed_df)) if analyzed_df is not None else 0
    if n_total == 0:
        return {"n_total": 0, "n_present": 0, "n_absent": 0, "fraction": 0.0, "zero": True, "partial": False}

    intake = pd.to_numeric(analyzed_df.get("pump_intake_pressure_psi"), errors="coerce")
    dp = pd.to_numeric(analyzed_df.get("delta_p_pump_psi"), errors="coerce")
    present_mask = intake.notna() & (intake > 0) & dp.notna()
    n_present = int(present_mask.sum())
    n_absent = n_total - n_present
    return {
        "n_total": n_total,
        "n_present": n_present,
        "n_absent": n_absent,
        "fraction": round(n_present / n_total, 4) if n_total else 0.0,
        "zero": n_present == 0,
        "partial": 0 < n_present < n_total,
        "present_mask": present_mask,
    }


def pip_present_rows(analyzed_df: pd.DataFrame, coverage: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Return only the PIP-present rows (the rows ΔP is computed on)."""
    cov = coverage or pip_coverage(analyzed_df)
    mask = cov.get("present_mask")
    if mask is None:
        return analyzed_df.iloc[0:0]
    return analyzed_df[mask]
