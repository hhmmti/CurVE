"""Service layer for CurVE — the two workflow modules CurVE consumes.

Only the clean, physics/pandas-backed modules CurVE actually imports live here:
  * ``preprocessed_pipeline_service`` — the join + engineered-dataframe pipeline
  * ``data_availability_gate``        — the availability gate the per-tool gate adapts

Both import nothing but ``compute`` / numpy / pandas. The other service modules from
the original app (``ml_recommendation_service``, ``ideal_curve_service``,
``pipeline_service``, ``app_service``, ``extension_hooks``) reach into a ``data/``
layer that is not part of the CurVE slice and are intentionally not carried over.
"""
