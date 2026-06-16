"""Service-layer orchestration for ESP analysis workflows."""

from typing import Dict, Optional

import pandas as pd

from compute.core_calcs import (
    SG_OIL,
    SG_WATER,
    WELL_DEPTH_FT,
    clean_flow_data,
    create_segments,
    engineer_features,
    flow_quality_audit,
    merge_flow_telemetry,
    treat_outliers,
)
from plotting.curves import generate_pump_curves, generate_temporal_plots_after_merge
from services.extension_hooks import AnalysisExtensionHooks


def run_full_pipeline(
    df_flow: pd.DataFrame,
    df_telem: pd.DataFrame,
    well_depth_ft: float = WELL_DEPTH_FT,
    sg_oil: float = SG_OIL,
    sg_water: float = SG_WATER,
    well_id: Optional[str] = None,
    extension_hooks: Optional[AnalysisExtensionHooks] = None,
) -> Dict:
    """Run complete analysis workflow and return UI-ready payloads."""
    results = {}

    if extension_hooks and extension_hooks.schema_mapper:
        mapped_flow, mapped_telem = extension_hooks.schema_mapper(df_flow, df_telem)
    else:
        mapped_flow, mapped_telem = df_flow, df_telem

    merged_data = merge_flow_telemetry(mapped_flow, mapped_telem)
    results["merged_data"] = merged_data
    results["merge_stats"] = {
        "total_rows": len(merged_data),
        "flowmeter_rows": len(mapped_flow),
        "telemetry_rows": len(mapped_telem),
    }

    temporal_plots, temporal_notes = generate_temporal_plots_after_merge(
        merged_data,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
        resample_hours=6,
    )
    results["temporal_plots"] = temporal_plots
    results["temporal_notes"] = temporal_notes

    results["quality_audit"] = flow_quality_audit(merged_data)

    cleaned_data = clean_flow_data(merged_data)
    results["cleaned_data"] = cleaned_data
    results["clean_stats"] = {
        "rows_before": len(merged_data),
        "rows_after": len(cleaned_data),
        "rows_dropped": len(merged_data) - len(cleaned_data),
    }

    featured_data = engineer_features(
        cleaned_data,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
    )
    results["featured_data"] = featured_data

    seg_df, segment_summary = create_segments(featured_data)
    results["seg_df"] = seg_df
    results["segment_summary"] = segment_summary

    outlier_df, outlier_summary = treat_outliers(
        seg_df,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
    )
    results["outlier_df"] = outlier_df
    results["outlier_summary"] = outlier_summary

    results["analysis_inputs"] = {
        "well_depth_ft": well_depth_ft,
        "sg_oil": sg_oil,
        "sg_water": sg_water,
    }

    try:
        fig = generate_pump_curves(outlier_df)
        results["pump_curve_fig"] = fig
    except Exception as e:
        results["pump_curve_error"] = str(e)

    # Optional extension outputs for future checkpoints.
    if extension_hooks and extension_hooks.ml_recommendation_loader and well_id:
        try:
            results["ml_recommendations"] = extension_hooks.ml_recommendation_loader(well_id)
        except Exception as e:
            results["ml_recommendations_error"] = str(e)

    if extension_hooks and extension_hooks.ideal_overlay_builder:
        try:
            results["ideal_overlay"] = extension_hooks.ideal_overlay_builder(results)
        except Exception as e:
            results["ideal_overlay_error"] = str(e)

    return results
