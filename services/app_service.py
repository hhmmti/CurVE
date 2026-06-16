"""Thin service functions for current app UI flows."""

from typing import Dict, Optional, Tuple

import pandas as pd

from data import VitalEnergyDB
from services.extension_hooks import AnalysisExtensionHooks
from services.pipeline_service import run_full_pipeline


def build_vital_energy_client() -> VitalEnergyDB:
    """Create a Vital Energy data client with default configuration."""
    return VitalEnergyDB()


def connect_vital_energy(db: VitalEnergyDB) -> Tuple[bool, str]:
    """Connect an existing Vital Energy client."""
    return db.connect()


def load_available_wells(db: VitalEnergyDB) -> pd.DataFrame:
    """Load wells that have overlapping flowmeter and telemetry data."""
    return db.get_overlapping_wells()


def load_well_coverage(db: VitalEnergyDB, well_id: str) -> Dict:
    """Load timestamp coverage summary for a specific well."""
    return db.get_well_coverage(well_id)


def load_well_timeseries(db: VitalEnergyDB, well_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw flowmeter and telemetry frames for a specific well."""
    return db.fetch_well_data(well_id)


def run_well_analysis(
    db: VitalEnergyDB,
    well_id: str,
    well_depth_ft: float,
    sg_oil: float,
    sg_water: float,
    extension_hooks: Optional[AnalysisExtensionHooks] = None,
) -> Dict:
    """Fetch well data and run the full analysis workflow."""
    if extension_hooks and extension_hooks.preprocessed_loader:
        df_flow, df_telem = extension_hooks.preprocessed_loader(well_id)
    else:
        df_flow, df_telem = load_well_timeseries(db, well_id)

    return run_full_pipeline(
        df_flow,
        df_telem,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
        well_id=well_id,
        extension_hooks=extension_hooks,
    )
