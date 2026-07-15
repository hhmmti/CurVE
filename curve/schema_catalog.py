"""CurVE ``sql_query`` schema catalog — the model's entire view of the allowlist.

M1 deliverable for the v1.1 ``sql_query`` tool (NL→SQL over Athena). A structured,
versioned, per-table description of the **five** allowlist tables, plus
:func:`render_schema_prompt` which emits the curated schema-as-text block that M3
injects verbatim into the generation prompt.

This module is **pure data + rendering** — no query execution, no physics, no AWS.
Catalog / database / table / region are read from :mod:`curve.config` (the single
source of truth for deployment pins); nothing here re-hardcodes a name config owns.

Provenance of the contents:
  * Column names + types were re-derived **live** via Athena ``DESCRIBE`` on
    2026-07-14 (SSO profile ``roam-ai``), cross-checked against the M1a gap report.
  * The three per-table flags — partition columns, scoping column, catalog type —
    and the mandatory-predicate sets are what M2 consumes to build safe predicates.
  * Curation decisions ratified 2026-07-14 are applied verbatim (see the per-table
    ``gotchas`` and the notes below): ``rrc_well_depth."API Depth"`` is feet/TVD and
    nullable (returned as ``NULL`` — no default-depth fallback lives here);
    ``ideal_pump_library_v1.shaft_type`` is **dropped** (mislabeled numeric duplicate
    of ``shaft_area``); pump rating columns are **units-unknown**; ``is_obsolete`` is
    surfaced, never dropped.

Scoping is **heterogeneous** — do NOT assume a well filter everywhere. Each table
carries ``scoping`` ∈ {node_id | well_id | org-only | none} and an explicit
``mandatory_predicates`` list so M2 can answer "which predicates are required for
table X" without guessing. ``ideal_pump_library_v1`` is an unscoped reference table
with **no** ``organization_id``/``well_id`` column at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from curve import config

SCHEMA_CATALOG_VERSION = "1.1.0-m1"
SCHEMA_DERIVED_LIVE = "2026-07-14"  # date of the live DESCRIBE pass backing this file


# --- controlled vocabularies --------------------------------------------------

# catalog-type flag (confirmed live from athena.get_data_catalog, not guessed)
CATALOG_GLUE = "Glue"          # Glue-native, partition-aware
CATALOG_FEDERATED = "federated"  # Athena federated connector (here: DynamoDB/Lambda)

# scoping column vocabulary — the confidentiality/cost key each table filters on
SCOPING_NODE_ID = "node_id"
SCOPING_WELL_ID = "well_id"
SCOPING_ORG_ONLY = "org-only"
SCOPING_NONE = "none"  # reference table — no org/well column exists
SCOPING_VALUES = frozenset(
    {SCOPING_NODE_ID, SCOPING_WELL_ID, SCOPING_ORG_ONLY, SCOPING_NONE}
)


# --- data model ---------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One physical column. ``gloss`` is set only where non-obvious (coded / unit-
    ambiguous / struct). ``partition`` marks a Glue partition key. ``quoted`` marks an
    identifier that MUST be double-quoted in SQL (space or reserved char)."""

    name: str
    type: str
    gloss: str = ""
    partition: bool = False
    quoted: bool = False

    def sql_ref(self) -> str:
        return f'"{self.name}"' if self.quoted else self.name


@dataclass(frozen=True)
class TableSchema:
    """A single allowlist table: identity (from config pins), purpose/grain, full
    column list, and the three mandatory flags + predicate metadata M2 consumes."""

    key: str
    catalog: str
    database: str
    table: str
    purpose: str
    grain: str
    catalog_type: str            # CATALOG_GLUE | CATALOG_FEDERATED
    columns: List[Column]
    partition_columns: List[str]
    scoping: str                 # one of SCOPING_VALUES
    mandatory_predicates: List[str]  # cols M2 MUST pin; [] for unscoped reference
    join_keys: List[str]
    gotchas: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.scoping in SCOPING_VALUES, f"bad scoping: {self.scoping}"
        assert self.catalog_type in (CATALOG_GLUE, CATALOG_FEDERATED)

    @property
    def fqn(self) -> str:
        """Fully-qualified ``catalog.database.table`` for the SQL FROM clause."""
        return f"{self.catalog}.{self.database}.{self.table}"


# --- the five allowlist tables ------------------------------------------------
# Identity fields reference config.py pins; column lists are the live DESCRIBE.

_TELEMETRY = TableSchema(
    key="telemetry",
    catalog=config.PREPROCESSED_CATALOG,
    database=config.PREPROCESSED_DATABASE,
    table=config.TELEMETRY_TABLE,
    purpose="Preprocessed ESP sensor telemetry (15-min resampled).",
    grain="one row per (organization_id, well_id, timestamp) within observation_day.",
    catalog_type=CATALOG_GLUE,
    columns=[
        Column("timestamp", "timestamp", "reading time within the partition day"),
        Column("bus_dc_volts", "double"),
        Column("casing_pressure_psi", "double"),
        Column("esp_downhole_amps", "double"),
        Column("esp_downhole_volts", "double"),
        Column("motor_amps", "double"),
        Column("motor_frequency_hz", "double", "measured motor frequency (control var)"),
        Column("motor_frequency_setpoint_hz", "double", "commanded frequency setpoint"),
        Column("motor_overload_amps", "double"),
        Column("motor_power_kw", "double", "present here despite Feb-doc claim it is not"),
        Column("motor_speed_rpm", "double"),
        Column("motor_temperature_f", "double"),
        Column("motor_torque_nm", "double"),
        Column("motor_underload_amps", "double"),
        Column("motor_vibration_x_g", "double"),
        Column("motor_vibration_y_g", "double"),
        Column("motor_volts", "double"),
        Column("pump_intake_pressure_psi", "double", "observed intake pressure (not settable)"),
        Column("pump_intake_temperature_f", "double"),
        Column("tubing_pressure_psi", "double", "surface tubing pressure (control var)"),
        Column("flowmeter_gas_vol", "double", "flowmeter volume; mcf"),
        Column("flowmeter_oil_vol", "double", "flowmeter volume; bbl"),
        Column("flowmeter_water_vol", "double", "flowmeter volume; bbl"),
        Column("organization_id", "string", "org scope", partition=True),
        Column("well_id", "string", "well scope", partition=True),
        Column("observation_day", "string", "partition date 'YYYY-MM-DD'", partition=True),
    ],
    partition_columns=["organization_id", "well_id", "observation_day"],
    scoping=SCOPING_WELL_ID,
    mandatory_predicates=["organization_id", "well_id"],
    join_keys=["organization_id", "well_id", "observation_day"],
    gotchas=[
        "Partitioned on organization_id, well_id, observation_day — ALWAYS filter "
        "organization_id AND well_id (both mandatory: cross-org guard + cost).",
        "Columns are RAW (motor_frequency_hz, not *_1h_avg). The Feb catalog's "
        "aggregated *_1h_avg/_7d_avg feature columns do NOT exist here.",
        "Scoping is well_id — there is NO node_id column (Feb doc says node_id).",
    ],
)

_PRODUCTION = TableSchema(
    key="production",
    catalog=config.PREPROCESSED_CATALOG,
    database=config.PREPROCESSED_DATABASE,
    table=config.PRODUCTION_TABLE,
    purpose="Preprocessed daily allocated production volumes.",
    grain="one row per (organization_id, well_id, observation_day).",
    catalog_type=CATALOG_GLUE,
    columns=[
        Column("timestamp", "timestamp", "production time within the partition day"),
        Column("alloc_gas_vol", "double", "allocated daily gas volume; mcf"),
        Column("alloc_oil_vol", "double", "allocated daily oil volume; bbl"),
        Column("alloc_water_vol", "double", "allocated daily water volume; bbl"),
        Column("organization_id", "string", "org scope", partition=True),
        Column("well_id", "string", "well scope", partition=True),
        Column("observation_day", "string", "partition date 'YYYY-MM-DD'", partition=True),
    ],
    partition_columns=["organization_id", "well_id", "observation_day"],
    scoping=SCOPING_WELL_ID,
    mandatory_predicates=["organization_id", "well_id"],
    join_keys=["organization_id", "well_id", "observation_day"],
    gotchas=[
        "Partitioned + org+well mandatory (same as telemetry). Join to telemetry on "
        "organization_id + well_id + observation_day.",
        "Lean table: only the three alloc_*_vol volumes. The Feb catalog's "
        "'production_esp_data' (well_age, theo_*, gor_14day_avg, node_id, …) is a "
        "different, richer object that does NOT exist here.",
    ],
)

_RECOMMENDATIONS = TableSchema(
    key="recommendations",
    catalog=config.REC_CATALOG,
    database=config.REC_DATABASE,
    table=config.REC_TABLE,
    purpose="Latest ML ESP setpoint recommendations (DynamoDB via Athena connector).",
    grain="one row per (organization_id, well_id, timestamp); newest = latest rec.",
    catalog_type=CATALOG_FEDERATED,
    columns=[
        Column("model_version", "varchar"),
        Column("last_datapoint_time_timestamp", "varchar"),
        Column("organization_id", "varchar", "org scope"),
        Column(
            "model_setpoint_recommendations",
            "struct",
            "per-goal recs: max_oil/total_economics -> {tubing_pressure_psi, "
            "motor_frequency_hz, production{gas,water,oil}} (decimals)",
        ),
        Column("status", "varchar"),
        Column("timestamp", "varchar", "recommendation time; latest via ORDER BY DESC LIMIT 1"),
        Column("last_datapoint_time", "varchar"),
        Column("total_economics_weights_json", "varchar", "JSON blob"),
        Column("telemetry_boundaries_json", "varchar", "JSON blob"),
        Column("well_id", "varchar", "well scope (DDB hash key)"),
        Column("summary_data_json", "varchar", "JSON blob: telemetry snapshot + forecasts"),
        Column("organization_id_well_id", "varchar", "composite org+well key (DDB)"),
        Column(
            "optimal_setpoint_recommendation",
            "struct",
            "chosen setpoint per goal (same struct shape as model_setpoint_recommendations)",
        ),
        Column("goal_function", "varchar"),
        Column("uuid", "varchar"),
        Column("data_quality_score", "decimal(38,9)"),
        Column("recommendation_type", "varchar"),
        Column("pipeline_execution_id", "varchar"),
        Column("max_step_sizes", "struct", "max per-step change {tubing_pressure_psi, motor_frequency_hz}"),
        Column("current_setpoint", "struct", "current operating setpoint (same struct shape)"),
        Column("telemetry_quality_issues", "array<struct>", "sensor boundary violations"),
        Column("out_of_bounds_reasons", "array<varchar>"),
    ],
    partition_columns=[],  # federated DynamoDB — no Hive partitions
    scoping=SCOPING_WELL_ID,
    mandatory_predicates=["organization_id", "well_id"],
    join_keys=["organization_id", "well_id"],
    gotchas=[
        "Federated (DynamoDB via Lambda connector): NO partitions. A missing "
        "org+well filter is a full-table scan — always filter organization_id AND well_id.",
        "Latest recommendation = ORDER BY timestamp DESC LIMIT 1 (timestamp is a string).",
        "Setpoint struct fields are tubing_pressure_psi / motor_frequency_hz (NOT the "
        "Feb doc's recommended_tubing_pressure_psi / recommended_frequency_hz).",
        "Numeric leaves are decimal(38,9); struct/array columns need dot/UNNEST access.",
    ],
)

_WELL_DEPTH = TableSchema(
    key="well_depth",
    catalog=config.WELL_DEPTH_CATALOG,
    database=config.WELL_DEPTH_DATABASE,
    table=config.WELL_DEPTH_TABLE,
    purpose="RRC-sourced well depth for depth enrichment (dev catalog).",
    grain="one row per well (keyed by well_id / API identifier).",
    catalog_type=CATALOG_GLUE,
    columns=[
        Column("organization_id", "string", "org slug (e.g. 'greenlakeenergy')"),
        Column("well_id", "string", "well / API identifier — the scope + join key"),
        Column("operator name", "string", "operator; identifier has a space", quoted=True),
        Column("lease name", "string", "lease; identifier has a space", quoted=True),
        Column("well no", "string", "well number suffix; identifier has a space", quoted=True),
        Column(
            "api depth",
            "double",
            "well depth in FEET, TVD (ratified); nullable -> return NULL as-is "
            "(no default-depth fallback here); identifier has a space",
            quoted=True,
        ),
    ],
    partition_columns=[],
    scoping=SCOPING_WELL_ID,
    mandatory_predicates=["well_id"],
    join_keys=["well_id"],
    gotchas=[
        'Spaced identifiers must be double-quoted: "operator name", "lease name", '
        '"well no", "api depth". (Code queries "API Depth" — Athena is case-insensitive.)',
        '"api depth" is FEET/TVD and NULLABLE — some wells have no depth; the SQL tool '
        "returns NULL, it does NOT substitute a default (that is a physics-tool concern).",
        "Scoped by well_id only (no node_id, no org filter required). Dev catalog "
        "(roam_dev_products) — prod-swap is flagged landing debt, not resolved here.",
        "org slug ('greenlakeenergy') vs preprocessed organization_id join-compatibility "
        "is UNVERIFIED — do not assert an org join across tables.",
    ],
)

_IDEAL_CATALOG = TableSchema(
    key="ideal_catalog",
    catalog=config.IDEAL_CATALOG_CATALOG,
    database=config.IDEAL_CATALOG_DATABASE,
    table=config.IDEAL_CATALOG_TABLE,
    purpose="Ideal pump-curve reference library (multi-manufacturer; dev catalog).",
    grain="one row per pump model (pump_id); a reference table, not well-scoped.",
    catalog_type=CATALOG_GLUE,
    columns=[
        Column("pump_id", "string", "primary key; pump pick filters on this in-memory"),
        Column("manufacturer", "string", "multi-manufacturer (e.g. Borets-Weatherford, ChampionX)"),
        Column("series", "string"),
        Column("esp_model", "string", "display model label (distinct from pump_id key)"),
        Column("min_recommended_bpd", "double", "recommended operating band, low (BPD)"),
        Column("bep_bpd", "double", "best-efficiency flow (BPD) — distinct from the band"),
        Column("max_recommended_bpd", "double", "recommended operating band, high (BPD)"),
        Column("max_plotted_bpd", "double"),
        Column("ideal_head_c1", "double", "head-vs-flow polynomial coefficient"),
        Column("ideal_head_c2", "double"),
        Column("ideal_head_c3", "double"),
        Column("ideal_head_c4", "double"),
        Column("ideal_head_c5", "double"),
        Column("ideal_head_c6", "double"),
        Column("ideal_power_c1", "double", "power-vs-flow polynomial coefficient"),
        Column("ideal_power_c2", "double"),
        Column("ideal_power_c3", "double"),
        Column("ideal_power_c4", "double"),
        Column("ideal_power_c5", "double"),
        Column("ideal_power_c6", "double"),
        Column("vis_valid_min_cp", "double", "viscosity-correction valid range, low (cP)"),
        Column("vis_valid_max_cp", "double", "viscosity-correction valid range, high (cP)"),
        Column("vis_capacity_c1", "double", "viscosity capacity-correction coefficient"),
        Column("vis_capacity_c2", "double"),
        Column("vis_capacity_c3", "double"),
        Column("vis_capacity_c4", "double"),
        Column("vis_head_c1", "double", "viscosity head-correction coefficient"),
        Column("vis_head_c2", "double"),
        Column("vis_head_c3", "double"),
        Column("vis_head_c4", "double"),
        Column("vis_power_c1", "double", "viscosity power-correction coefficient"),
        Column("vis_power_c2", "double"),
        Column("vis_power_c3", "double"),
        Column("vis_power_c4", "double"),
        Column("nominal_casing_size", "string", "nominal casing size (inches)"),
        Column("std_shaft_rating", "string", "rating; UNITS UNKNOWN (do not assume HP/lbf)"),
        Column("hs_shaft_rating", "string", "rating; UNITS UNKNOWN"),
        Column("std_housing_rating", "string", "rating; UNITS UNKNOWN"),
        Column("hs_housing_rating", "string", "rating; UNITS UNKNOWN"),
        Column("comp_thrust_rating", "string", "rating; UNITS UNKNOWN"),
        Column("shaft_area", "double", "shaft cross-section area; unit unverified"),
        Column("is_obsolete", "boolean", "surfaced, never dropped (~209/699 rows are true)"),
        Column("user_champx_detail", "string", "low-coverage free-text detail (mostly null)"),
    ],
    partition_columns=[],
    scoping=SCOPING_NONE,
    mandatory_predicates=[],  # unscoped reference — no org/well predicate exists or applies
    join_keys=["pump_id"],
    gotchas=[
        "Reference table: NO organization_id and NO well_id column — never add an "
        "org/well predicate; narrow by pump_id / esp_model / series instead.",
        "shaft_type is intentionally DROPPED from this catalog (a mislabeled numeric "
        "duplicate of shaft_area) — do not re-add it.",
        "Rating columns (std/hs shaft, std/hs housing, comp_thrust) are strings with "
        "UNKNOWN units — do not assert HP/lbf in generated SQL or narration.",
        "Multi-manufacturer (not ChampionX-only). Dev catalog (roam_dev_products).",
    ],
)


SCHEMA_TABLES: List[TableSchema] = [
    _TELEMETRY,
    _PRODUCTION,
    _RECOMMENDATIONS,
    _WELL_DEPTH,
    _IDEAL_CATALOG,
]

SCHEMA_CATALOG: Dict[str, TableSchema] = {t.key: t for t in SCHEMA_TABLES}


# --- prompt rendering ---------------------------------------------------------


def _render_table(t: TableSchema) -> str:
    part = "partitioned" if t.partition_columns else "unpartitioned"
    lines: List[str] = []
    lines.append(f"TABLE: {t.fqn}  [{t.catalog_type}, {part}]")
    lines.append(f"  Purpose: {t.purpose}")
    lines.append(f"  Grain: {t.grain}")
    lines.append("  Columns:")
    for c in t.columns:
        tag = " [partition]" if c.partition else ""
        gloss = f"  — {c.gloss}" if c.gloss else ""
        lines.append(f"    {c.sql_ref()} ({c.type}){tag}{gloss}")
    lines.append(
        "  Partitions: " + (", ".join(t.partition_columns) if t.partition_columns else "none")
    )
    if t.mandatory_predicates:
        lines.append(
            f"  Scoping: {t.scoping} — MANDATORY filter: "
            + " AND ".join(t.mandatory_predicates)
        )
    else:
        lines.append(f"  Scoping: {t.scoping} — no mandatory filter (reference table)")
    lines.append("  Join keys: " + ", ".join(t.join_keys))
    if t.gotchas:
        lines.append("  Gotchas:")
        for g in t.gotchas:
            lines.append(f"    - {g}")
    return "\n".join(lines)


def render_schema_prompt() -> str:
    """Return the curated schema as a single readable text block for the NL→SQL prompt.

    This is the model's ENTIRE schema view: the five allowlist tables, each with
    columns (+ glosses only where non-obvious), partition/scoping/catalog-type flags,
    join keys, and gotchas. Injected verbatim by M3 into the generation prompt.
    """
    header = (
        f"CurVE SQL schema catalog v{SCHEMA_CATALOG_VERSION} "
        f"(live-derived {SCHEMA_DERIVED_LIVE}). "
        "Query ONLY these five tables. Scoping is per-table and heterogeneous — do NOT "
        "assume a well filter everywhere; honor each table's MANDATORY filter exactly."
    )
    blocks = [header] + [_render_table(t) for t in SCHEMA_TABLES]
    return "\n\n".join(blocks) + "\n"


if __name__ == "__main__":  # manual inspection: python -m curve.schema_catalog
    print(render_schema_prompt())
