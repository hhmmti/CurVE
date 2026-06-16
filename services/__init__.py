"""Service layer package for workflow orchestration.

NOTE: Vital Energy services have been removed from the active app layer.
Legacy Vital service functions remain in app_service.py for reference only.
"""

from .extension_hooks import AnalysisExtensionHooks
from .pipeline_service import run_full_pipeline
from .ml_recommendation_service import (
	build_analysis_from_latest_row,
	select_pump_row,
)
from .data_availability_gate import (
	run_data_availability_gate,
	build_readiness_dataframe,
	summarize_readiness,
)

__all__ = [
	"AnalysisExtensionHooks",
	"select_pump_row",
	"run_full_pipeline",
	"build_analysis_from_latest_row",
	"run_data_availability_gate",
	"build_readiness_dataframe",
	"summarize_readiness",
]
