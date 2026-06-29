"""CurVE recommendation access — the Athena-mirror read of the VE's setpoint
recommendations, plus the VE-parity parse of the rec-item blobs (M3).

CurVE's **second data path** (after preprocessed telemetry, ``curve.data``). Per
[[CurVE-decisions]] §9 (source-of-truth parity), recommendations are read from
CurVE's **Athena mirror** of the VE's DDB ``esp_setpoint_recommendations_v2`` — a
**verified mirror** (Clause A: same source-of-truth, different transport). So this
path is **live, no DDB port, no fixtures, no divergence flag**.

WHERE THE READ CAME FROM (resolved during execution, per Requirement #1)
-----------------------------------------------------------------------
The app's recommendation access is ``app/data/ml_recommendation_db.py`` →
``MLRecommendationDB.get_latest_recommendation_row()``. It is the known-good reader
of the same mirror table the VE's DDB replicates:

    catalog  : roam_prd_ddb            (DynamoDB via the Athena connector)
    database : default
    table    : esp_setpoint_recommendations_v2
    columns  : organization_id, well_id, uuid, timestamp,
               current_setpoint, model_setpoint_recommendations,
               summary_data_json
    query    : SELECT … ORDER BY timestamp DESC LIMIT 1   (latest per org/well)

This module ports **that exact read** into ``playground`` as its own small
awswrangler query (the ``curve.data`` access pattern — NOT a copy of the app's data
layer wholesale, and NOT physics). The blobs carried confirm the schema CurVE needs:
``uuid`` (traceability), ``current_setpoint`` + ``model_setpoint_recommendations``
(the operating-point blobs), and ``summary_data_json`` (the measured current-state
snapshot — power/intake/allocation).

VE-PARITY PARSE (Requirement #1 / Do-NOT: never synthesize a ``summary_data_json``)
-----------------------------------------------------------------------------------
The VE (``actions.py::explain_model_recommendation``) consumes the rec item by:
  * ``current_setpoint`` / ``model_setpoint_recommendations`` — read as maps, the
    recommended branch keyed by the **goal function**
    (``actions.py`` L3473: ``model_recs.get(goal_function) or
    optimal_setpoint_recommendation``);
  * ``summary_data_json`` — ``json.loads(...summary_data_json, "{}")`` (L913).

We **imitate** that parse, reusing the app's pure parsers
(``compute.ml_recommendation_calcs.parse_setpoint_like_map`` for the setpoint maps
and ``parse_json_summary_data`` for the JSON summary — the latter is exactly the VE's
``json.loads`` with NaN-tolerance). We never re-author / synthesize the summary; we
consume and parse the VE's copy off the rec item.

The Athena mirror carries ``model_setpoint_recommendations`` (the goal-keyed map);
the recommended operating point is its ``[goal]`` branch — matching the app's
``extract_compare_row(optimal_text=model_setpoint_recommendations, method=goal)``.
The VE's secondary ``optimal_setpoint_recommendation`` fallback is a DDB-native field
the app's proven mirror query does not select; we follow the app's proven projection
and flag this in the change report rather than reference an unverified column.

Credentials: same precedence as :mod:`curve.data` / :mod:`curve.well_depth`
(explicit ``profile_name`` → ``AWS_PROFILE`` env → default chain). Region
``us-east-1``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import boto3

# Default recommendation method/goal branch — matches the app's default
# (``build_analysis_from_latest_row(method="max_oil")``). The VE keys the recommended
# setpoint on the goal function; "max_oil" is CurVE's default goal until a session
# goal is selected.
DEFAULT_RECOMMENDATION_METHOD = "max_oil"

# Athena mirror of the VE's DDB recommendations (ported from the app's data layer).
REC_CATALOG = "roam_prd_ddb"  # DynamoDB via the Athena connector
REC_DATABASE = "default"
REC_TABLE = "esp_setpoint_recommendations_v2"
REC_ATHENA_S3_OUTPUT = "s3://esp-athena-results-v2-411237692998/"
REC_REGION = "us-east-1"

# 3-phase power, matching the VE's _compute_motor_power_kwh_per_day (PF fixed at 0.85,
# √3 ≈ 1.7321) — see actions.py L3269. We mirror the VE's constants exactly so the
# amp×volt proxy power term reproduces the VE's number (kW; the VE's ×24 gives kWh/day).
VE_POWER_FACTOR = 0.85
VE_SQRT3 = 1.7321


def _quote(value: str) -> str:
    """Single-quote a SQL literal, escaping embedded quotes (same as curve.data)."""
    return "'" + str(value).replace("'", "''") + "'"


def _build_session(
    profile_name: Optional[str] = None, region_name: str = REC_REGION
) -> boto3.Session:
    """Profile precedence identical to curve.data: explicit → AWS_PROFILE → chain."""
    if profile_name:
        return boto3.Session(profile_name=profile_name, region_name=region_name)
    if os.environ.get("AWS_PROFILE"):
        return boto3.Session(
            profile_name=os.environ["AWS_PROFILE"], region_name=region_name
        )
    return boto3.Session(region_name=region_name)  # role / env / default chain


def build_latest_recommendation_query(organization_id: str, well_id: str) -> str:
    """Build the latest-recommendation query for one well (pure string — testable).

    Mirrors the app's ``get_latest_recommendation_row`` projection: identity + uuid +
    timestamp + the three setpoint blobs, newest first. Cross-org guard: both keys are
    mandatory and ANDed — never a broad scan.
    """
    if not organization_id or not well_id:
        raise ValueError(
            "Both organization_id and well_id are required (cross-org "
            "confidentiality); refusing to query recommendations without both."
        )
    return (
        "SELECT organization_id, well_id, uuid, timestamp, "
        "current_setpoint, model_setpoint_recommendations, "
        "summary_data_json "
        f"FROM {REC_DATABASE}.{REC_TABLE} "
        f"WHERE organization_id = {_quote(organization_id)} "
        f"AND well_id = {_quote(well_id)} "
        "ORDER BY timestamp DESC LIMIT 1"
    )


def fetch_latest_recommendation(
    organization_id: str,
    well_id: str,
    *,
    profile_name: Optional[str] = None,
    region_name: str = REC_REGION,
    s3_output: str = REC_ATHENA_S3_OUTPUT,
    session: Optional[boto3.Session] = None,
) -> Optional[Dict[str, Any]]:
    """Return the latest recommendation row dict for ``(org, well)``, or ``None``.

    ``None`` means **no recommendation exists** for this well/session — the caller's
    recommendation-absence hard block fires (``not-ready / recommendation_absent``).
    The recommendation is NEVER fabricated when absent.

    Kept as a free function (not just a method) so cred-free tests can monkeypatch
    ``curve.recommendations.fetch_latest_recommendation`` with a fixture / ``None`` and
    never touch AWS.
    """
    # Imported lazily so importing this module (and cred-free tests that monkeypatch
    # this fetch) does not require awswrangler at import time.
    import awswrangler as wr

    sess = session or _build_session(profile_name=profile_name, region_name=region_name)
    df = wr.athena.read_sql_query(
        sql=build_latest_recommendation_query(organization_id, well_id),
        database=REC_DATABASE,
        data_source=REC_CATALOG,
        boto3_session=sess,
        ctas_approach=False,
        s3_output=s3_output,
    )
    if df is None or df.empty:
        return None
    return df.iloc[0].to_dict()


# --- VE-parity parse of the rec-item blobs ------------------------------------


def has_recommendation(latest_row: Optional[Dict[str, Any]]) -> bool:
    """True when a usable recommendation is present (both setpoint blobs non-empty).

    Mirrors the app's guard (``build_analysis_from_latest_row``: raises when
    ``model_setpoint_recommendations`` or ``current_setpoint`` is missing) — but here
    a miss is the **recommendation-absence** condition, not an exception.
    """
    if not latest_row:
        return False
    return bool(latest_row.get("model_setpoint_recommendations")) and bool(
        latest_row.get("current_setpoint")
    )


def summary_blob(latest_row: Dict[str, Any]) -> Any:
    """Return the raw ``summary_data_json`` blob off the rec item (VE's key).

    The VE reads ``recommendation_item.get("summary_data_json")``; the app aliases it
    to ``json_summary_data`` in its query. We accept either spelling but never
    synthesize one — the blob is consumed as-is and parsed by the compute's
    ``parse_json_summary_data`` (the VE's ``json.loads`` with NaN-tolerance).
    """
    if "summary_data_json" in latest_row:
        return latest_row.get("summary_data_json")
    return latest_row.get("json_summary_data")


def extract_operating_point(
    latest_row: Dict[str, Any], method: str = DEFAULT_RECOMMENDATION_METHOD
) -> Dict[str, Any]:
    """Parse current-vs-recommended operating point from the rec blobs (VE-parity).

    Reuses the app's pure parsers (``compute.ml_recommendation_calcs``):
    ``extract_compare_row`` parses ``current_setpoint`` (current) +
    ``model_setpoint_recommendations[method]`` (recommended) into the compare row.
    No physics here — faithful extraction relative to the payload (Validated-rel-payload).
    """
    from compute import ml_recommendation_calcs

    return ml_recommendation_calcs.extract_compare_row(
        optimal_text=latest_row.get("model_setpoint_recommendations"),
        current_text=latest_row.get("current_setpoint"),
        method=method,
    )


def ve_power_kw_from_amps_volts(
    motor_amps: Optional[float], motor_volts: Optional[float]
) -> Optional[float]:
    """3-phase power (kW) from measured amps × volts — the VE's method (PF 0.85).

    Mirrors ``actions.py::_compute_motor_power_kwh_per_day`` exactly:
        kW = amps × volts × √3 × 0.85 / 1000
    (the VE then ×24 for kWh/day; we keep kW for the efficiency term). Returns ``None``
    when either input is missing or non-positive — never a fabricated power.

    This is the **amp×volt proxy** power path → labeled **Proxy** by the caller (CurVE
    v1's first Proxy label). A direct measured ``motor_power_kw`` channel is preferred
    and labeled Validated; this proxy is the fallback.
    """
    try:
        amps = float(motor_amps)
        volts = float(motor_volts)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(amps) and math.isfinite(volts)):
        return None
    if amps <= 0 or volts <= 0:
        return None
    return amps * volts * VE_SQRT3 * VE_POWER_FACTOR / 1000.0
