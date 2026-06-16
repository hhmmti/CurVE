"""Compute/core layer package for pure engineering logic."""

from .core_calcs import (
	MIN_SEGMENT_ROWS,
	WELL_DEPTH_FT,
	SG_OIL,
	SG_WATER,
	clean_flow_data,
	create_segments,
	detect_amp_column,
	detect_frequency_column,
	detect_volt_column,
	engineer_features,
	flow_quality_audit,
	merge_flow_telemetry,
	treat_outliers,
)
from .ml_recommendation_calcs import (
	build_curve_payload,
	build_summary_table,
	extract_compare_row,
	parse_setpoint_like_map,
)

__all__ = [
	"MIN_SEGMENT_ROWS",
	"WELL_DEPTH_FT",
	"SG_OIL",
	"SG_WATER",
	"clean_flow_data",
	"create_segments",
	"detect_amp_column",
	"detect_frequency_column",
	"detect_volt_column",
	"engineer_features",
	"flow_quality_audit",
	"merge_flow_telemetry",
	"parse_setpoint_like_map",
	"treat_outliers",
	"extract_compare_row",
	"build_summary_table",
	"build_curve_payload",
]
