"""Shared physics constants and pure helper functions for ESP analysis.

This module is the single source of truth for constants and scalar formulas
that are used across compute/preprocessed_calcs.py, compute/core_calcs.py,
compute/ml_recommendation_calcs.py, and compute/ideal_curve_overlay.py.

Rules:
- No Streamlit imports.
- No data loading.
- No plotting.
- No session state.
- All functions are pure (no side effects, deterministic output).

Unit conventions used throughout:
- Pressure   : psi
- Head       : ft
- Flow rate  : bbl/day
- Depth      : ft
- Power      : HP (horsepower) or kW where stated
- SG         : dimensionless specific gravity (water = 1.0)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

# Pressure gradient of water: 1 psi per 2.31 ft of head (at SG=1.0)
PSI_PER_FT_PER_SG: float = 0.433
"""Hydrostatic gradient coefficient: psi = PSI_PER_FT_PER_SG * SG * depth_ft"""

FT_PER_PSI_WATER: float = 2.31
"""Head conversion for water (SG=1.0): ft = psi * FT_PER_PSI_WATER"""

# ---------------------------------------------------------------------------
# Default fluid property assumptions
# ---------------------------------------------------------------------------

DEFAULT_SG_OIL: float = 0.85
"""Default oil specific gravity. Source: canonical app defaults (WELL_DEPTH_FT comment block)."""

DEFAULT_SG_WATER: float = 1.00
"""Default water specific gravity (fresh/formation water approximation)."""

# ---------------------------------------------------------------------------
# Default well geometry
# ---------------------------------------------------------------------------

DEFAULT_WELL_DEPTH_FT: float = 5000.0
"""Default well depth when no well-config depth is available. Hardcoded fallback."""

# ---------------------------------------------------------------------------
# Power conversion
# ---------------------------------------------------------------------------

WATTS_PER_HP: float = 745.7
"""Conversion factor: 1 HP = 745.7 W."""

DEFAULT_POWER_FACTOR: float = 0.90
"""Default three-phase motor power factor used when actual PF is unknown."""

# ---------------------------------------------------------------------------
# Hydraulic / efficiency denominators (keep as named constants to avoid
# bare literals in compute modules)
# ---------------------------------------------------------------------------

HHP_DENOMINATOR: float = 58765.7
"""Denominator for hydraulic horsepower proxy: HHP = (flow_bpd * delta_p_psi) / HHP_DENOMINATOR."""

PUMP_EFF_DENOMINATOR: float = 135773.0
"""Denominator for ideal pump efficiency ratio: eff = (q * head) / (bhp * PUMP_EFF_DENOMINATOR)."""


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def calc_liquid_rate(oil: float, water: float) -> float:
    """Compute total liquid rate from oil and water volumes.

    Formula:
        liquid_rate = oil + water

    Units:
        oil   : bbl/day
        water : bbl/day
        return: bbl/day

    Notes:
        - NaN inputs propagate to NaN output.
        - Negative inputs are not clipped; callers are responsible for cleaning data.
    """
    return oil + water


def calc_gor(gas: float, oil: float) -> float:
    """Compute gas-oil ratio.

    Formula:
        GOR = gas / oil   (oil > 0)
        GOR = NaN         (oil == 0 or oil is NaN)

    Units:
        gas   : Mscf/day or scf/day (caller-defined; ratio is unit-consistent)
        oil   : bbl/day
        return: scf/bbl (same unit as gas/oil)

    Notes:
        - Returns NaN when oil is zero to avoid division by zero.
    """
    try:
        if oil > 0:
            return float(gas) / float(oil)
    except (TypeError, ValueError):
        pass
    return float("nan")


def calc_water_cut(water: float, liquid_rate: float) -> float:
    """Compute water-cut fraction.

    Formula:
        water_cut = water / liquid_rate   (liquid_rate > 0)
        water_cut = NaN                   (liquid_rate == 0 or NaN)

    Units:
        water        : bbl/day
        liquid_rate  : bbl/day
        return       : dimensionless fraction in [0, 1]

    Notes:
        - Output is clipped to [0, 1].
        - Returns NaN when liquid_rate is zero.
    """
    try:
        if liquid_rate > 0:
            return float(np.clip(float(water) / float(liquid_rate), 0.0, 1.0))
    except (TypeError, ValueError):
        pass
    return float("nan")


def calc_mixture_sg(
    water_cut: float,
    sg_oil: float = DEFAULT_SG_OIL,
    sg_water: float = DEFAULT_SG_WATER,
) -> float:
    """Compute fluid mixture specific gravity from water cut.

    Formula:
        sg_mixture = sg_oil * (1 - water_cut) + sg_water * water_cut

    Units:
        water_cut : dimensionless fraction [0, 1]
        sg_oil    : dimensionless
        sg_water  : dimensionless
        return    : dimensionless

    Assumptions:
        - Linear (volumetric) mixing rule.
        - Defaults: SG_OIL = 0.85, SG_WATER = 1.00.

    Notes:
        - Returns sg_oil when water_cut is NaN (fallback matching preprocessed_calcs behavior).
    """
    try:
        wc = float(water_cut)
        if not np.isfinite(wc):
            return float(sg_oil)
        return float(sg_oil) * (1.0 - wc) + float(sg_water) * wc
    except (TypeError, ValueError):
        return float(sg_oil)


def calc_hydrostatic_pressure_psi(sg: float, depth_ft: float) -> float:
    """Compute hydrostatic pressure from fluid column.

    Formula:
        delta_p_hyd = PSI_PER_FT_PER_SG * sg * depth_ft
                    = 0.433 * sg * depth_ft

    Units:
        sg       : dimensionless specific gravity
        depth_ft : ft (measured depth from surface to pump)
        return   : psi

    Assumptions:
        - Uses canonical constant 0.433 psi/ft per unit SG.
        - depth_ft defaults to DEFAULT_WELL_DEPTH_FT (5000 ft) when not provided by caller.

    Notes:
        - Returns NaN when sg or depth_ft is NaN / non-finite.
    """
    try:
        sg_f = float(sg)
        d_f = float(depth_ft)
        if not (np.isfinite(sg_f) and np.isfinite(d_f)):
            return float("nan")
        return PSI_PER_FT_PER_SG * sg_f * d_f
    except (TypeError, ValueError):
        return float("nan")


def calc_discharge_pressure_downhole_psi(
    tubing_pressure_psi: float,
    hydrostatic_psi: float,
) -> float:
    """Compute estimated downhole discharge pressure at pump outlet.

    Formula:
        p_dis_downhole = tubing_pressure_psi + hydrostatic_psi

    Units:
        tubing_pressure_psi : psi (surface tubing head pressure)
        hydrostatic_psi     : psi (from calc_hydrostatic_pressure_psi)
        return              : psi

    Notes:
        - NaN inputs propagate to NaN output.
    """
    try:
        return float(tubing_pressure_psi) + float(hydrostatic_psi)
    except (TypeError, ValueError):
        return float("nan")


def calc_pump_delta_p_psi(
    discharge_downhole_psi: float,
    intake_pressure_psi: float,
) -> float:
    """Compute pressure differential across the pump.

    Formula:
        delta_p_pump = discharge_downhole_psi - intake_pressure_psi

    Units:
        discharge_downhole_psi : psi
        intake_pressure_psi    : psi
        return                 : psi

    Notes:
        - Returns NaN when intake_pressure_psi is missing (no fallback here;
          callers handle the intake fallback before calling this function).
    """
    try:
        return float(discharge_downhole_psi) - float(intake_pressure_psi)
    except (TypeError, ValueError):
        return float("nan")


def calc_head_ft_from_pressure_psi(pressure_psi: float, sg: float = 1.0) -> float:
    """Convert pressure in psi to equivalent fluid column head in feet.

    Formula:
        head_ft = pressure_psi * FT_PER_PSI_WATER / sg
                = pressure_psi * 2.31 / sg

    Units:
        pressure_psi : psi
        sg           : dimensionless specific gravity (default 1.0 = water)
        return       : ft

    Notes:
        - Returns NaN when sg is zero or non-finite.
    """
    try:
        sg_f = float(sg)
        if not np.isfinite(sg_f) or sg_f == 0.0:
            return float("nan")
        return float(pressure_psi) * FT_PER_PSI_WATER / sg_f
    except (TypeError, ValueError):
        return float("nan")


def calc_pressure_psi_from_head_ft(head_ft: float, sg: float = 1.0) -> float:
    """Convert fluid column head in feet to pressure in psi.

    Formula:
        pressure_psi = (head_ft / FT_PER_PSI_WATER) * sg
                     = (head_ft / 2.31) * sg

    Units:
        head_ft : ft
        sg      : dimensionless specific gravity (default 1.0 = water)
        return  : psi

    Notes:
        - This is the inverse of calc_head_ft_from_pressure_psi.
        - Used for converting ideal pump head output to delta-P in psi.
    """
    try:
        sg_f = float(sg)
        if not np.isfinite(sg_f):
            return float("nan")

        head = np.asarray(head_ft, dtype=float)
        pressure = (head / FT_PER_PSI_WATER) * sg_f

        if np.ndim(pressure) == 0:
            return float(pressure)
        return pressure
    except (TypeError, ValueError):
        return float("nan")
