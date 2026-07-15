"""CurVE runtime configuration — every deployment pin, read from an env var.

Single source of truth for the values that used to be hardcoded across the tree:
AWS profile / region, the Bedrock model id, the Athena catalog.database.table
bindings, the Athena results bucket, the BEP-tolerance default, and the dev-panel
flag. Each value is read from an environment variable with a **default equal to the
prior hardcoded literal** — so with no env set, behavior is byte-for-byte identical
to before this module existed. This is purely "make the current values overridable";
it does NOT re-point anything at a different source.

Precedence: an OS environment variable wins; otherwise a KEY=VALUE line in an
optional ``.env`` file at the repo root; otherwise the built-in default below. The
``.env`` loader is dependency-free and a no-op when no ``.env`` exists (so the
shipped company copy, which gitignores ``.env``, uses the defaults). See
``.env.example`` for the documented variable list.
"""

from __future__ import annotations

import os

# --- optional, dependency-free .env loader ------------------------------------


def _load_dotenv() -> None:
    """Populate os.environ from a repo-root ``.env`` (real env vars win; no override).

    Minimal KEY=VALUE parser — no interpolation, no export keyword handling beyond a
    leading ``export``. Silently does nothing when the file is absent or unreadable,
    so importing this module never fails and never changes behavior without a .env.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


# --- typed readers ------------------------------------------------------------


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- AWS / Bedrock ------------------------------------------------------------

AWS_PROFILE = _get("CURVE_AWS_PROFILE", "roam-ai")
AWS_REGION = _get("CURVE_AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = _get("CURVE_BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")

# Shared Athena query-results bucket (all data modules used the same literal).
ATHENA_S3_OUTPUT = _get(
    "CURVE_ATHENA_S3_OUTPUT", "s3://esp-athena-results-v2-411237692998/"
)

# --- Athena bindings: preprocessed telemetry + production (curve/data.py) ------

PREPROCESSED_CATALOG = _get("CURVE_PREPROCESSED_CATALOG", "roam_prd_products")
PREPROCESSED_DATABASE = _get("CURVE_PREPROCESSED_DATABASE", "esp_optimization_v2")
TELEMETRY_TABLE = _get("CURVE_TELEMETRY_TABLE", "esp_telemetry_preprocessed")
PRODUCTION_TABLE = _get("CURVE_PRODUCTION_TABLE", "esp_production_preprocessed")

# --- Athena bindings: ML setpoint recommendations (curve/recommendations.py) ---

REC_CATALOG = _get("CURVE_REC_CATALOG", "roam_prd_ddb")
REC_DATABASE = _get("CURVE_REC_DATABASE", "default")
REC_TABLE = _get("CURVE_REC_TABLE", "esp_setpoint_recommendations_v2")

# --- Athena bindings: well configuration (curve/well_catalog.py) ---------------

WELLCFG_CATALOG = _get("CURVE_WELLCFG_CATALOG", "roam_prd_ddb")
WELLCFG_DATABASE = _get("CURVE_WELLCFG_DATABASE", "default")
WELLCFG_TABLE = _get("CURVE_WELLCFG_TABLE", "esp_well_configuration_v2")

# --- Athena bindings: RRC well depth (curve/well_depth.py) ---------------------

WELL_DEPTH_CATALOG = _get("CURVE_WELL_DEPTH_CATALOG", "roam_dev_products")
WELL_DEPTH_DATABASE = _get("CURVE_WELL_DEPTH_DATABASE", "well_depth_dev")
WELL_DEPTH_TABLE = _get("CURVE_WELL_DEPTH_TABLE", "rrc_well_depth")

# --- Athena bindings: ideal pump catalog (curve/ideal_catalog.py) --------------

IDEAL_CATALOG_CATALOG = _get("CURVE_IDEAL_CATALOG_CATALOG", "roam_dev_products")
IDEAL_CATALOG_DATABASE = _get("CURVE_IDEAL_CATALOG_DATABASE", "esp_ideal_pump_dev")
IDEAL_CATALOG_TABLE = _get("CURVE_IDEAL_CATALOG_TABLE", "ideal_pump_library_v1")

# --- physics defaults ---------------------------------------------------------

# BEP-tolerance slider default (0.25 = ±25%), the app's narrow-catalog window.
DEFAULT_BEP_TOLERANCE = _get_float("CURVE_DEFAULT_BEP_TOLERANCE", 0.25)

# --- sql_query guard (M2) pins ------------------------------------------------
# The single hard row cap the guard forces onto every generated query. Absent LIMIT
# is injected at this value; a model LIMIT above it is REJECTED (never clamped).
SQL_QUERY_LIMIT_CAP = _get_int("CURVE_SQL_QUERY_LIMIT_CAP", 10000)
# Client-side wall-clock budget for execute(); on breach the query is cancelled with
# stop_query_execution and the call aborts (fail-closed — never an unbounded wait).
SQL_QUERY_TIMEOUT_SECONDS = _get_float("CURVE_SQL_QUERY_TIMEOUT_SECONDS", 60.0)

# --- UI flags -----------------------------------------------------------------

# Streamlit developer panel default. OFF for the company copy (was hardcoded ON in
# the sidebar checkbox); flip on with CURVE_DEV_PANEL=true to expose the loop.
DEV_PANEL_DEFAULT = _get_bool("CURVE_DEV_PANEL", False)
