"""CurVE connection-resolution layer — ideal-catalog fetch + BEP-narrow + pump pick (M4 / 4a).

This is the FIRST connection-dependent surface: the well ↔ installed-pump ↔
ideal-curve linkage. v1 resolves that connection by **manual pump pick** (CurVE-
decisions §1 connection ladder: auto → manual → blocked; v1 = manual). This module is
the *plumbing only* — fetch the catalog, BEP-narrow the candidates by the well's total
fluid, shape the chosen pump as a setup-injected value, and emit a per-well coverage
report. It does NOT build the ``curve_position`` tool (that is 4b) and exposes nothing
to the model: the pump pick rides the same setup-injection path as org/well + the
depth/SG overrides (the M3 override mechanism), never a Converse tool argument.

WHERE THE CATALOG READ + BEP FORMULA CAME FROM (resolved per Requirement #1)
---------------------------------------------------------------------------
The ideal-curve overlay is **CurVE-owned** (CurVE-decisions §9 — the VE does not do
it), so we read the catalog the way the week-8 physics app does. Resolved from the
app, NOT guessed:

  * Catalog read — ``app/data/ideal_pump_catalog.py::load_ideal_catalog`` queries::

        catalog  : roam_dev_products
        database : esp_ideal_pump_dev
        table    : ideal_pump_library_v1            (NOT esp_optimization_v2)
        query    : SELECT * FROM esp_ideal_pump_dev.ideal_pump_library_v1

  * BEP narrowing — ``app/services/ideal_curve_service.py::narrow_catalog`` is a
    **two-stage AND** hard filter (NOT a BEP±window alone)::

        stage 1 (flow-range containment): min_recommended_bpd <= rate <= max_recommended_bpd
        stage 2 (BEP proximity):          bep_bpd*(1-tol) <= rate <= bep_bpd*(1+tol)
        candidates = catalog[stage1 & stage2]

    ``tol`` (BEP tolerance) defaults to **0.25** and is the app's slider value
    (``preprocessed_page.py`` / ``ml_recommendation_page.py``: 5–50%, default 25%).
    We port BOTH stages verbatim. :func:`narrow_candidates` reproduces it exactly.

  * Total fluid (the BEP narrowing input) — the app's preprocessed overlay uses
    ``median(analyzed_df["liquid_rate_bbl_day"])`` (``preprocessed_page.py`` L652).
    Total fluid is the LIQUID rate (oil + water), NOT the oil rate (guardrail 4).
    :func:`resolve_total_fluid_bpd` reads exactly that.

TWO INTENTIONAL DEVIATIONS FROM THE APP (flagged, per guardrails 3 + 11)
-----------------------------------------------------------------------
  1. The app's ``load_ideal_catalog`` **drops** ``is_obsolete`` rows for its end-user
     selectors. CurVE must **surface** obsolete rows with a flag and NOT auto-exclude
     them — an operator may run an obsolete pump (guardrail 3). So this fetch keeps
     obsolete rows; :func:`annotate_candidates` flags them instead of dropping.
  2. Rows with null/incomplete ``ideal_head_c*`` / ``ideal_power_c*`` are **not
     selectable** (you cannot reconstruct a curve from missing coeffs). They are
     excluded from the selectable candidate list but counted in the coverage report,
     never silently dropped (guardrail 3).

RE-READ EACH SESSION (build-plan M4 seam)
-----------------------------------------
The fetch is written to re-read the Athena catalog each session (no module-level
cache here; the caller decides caching). When Track 2 lands ``pump_config``, covered
wells flip manual → auto with no v1 rework (CurVE-decisions §5 B3).

Credentials: same precedence as :mod:`curve.data` / :mod:`curve.recommendations`
(explicit ``profile_name`` → ``AWS_PROFILE`` env → default chain). Region
``us-east-1``. The fetch is a free function so cred-free tests monkeypatch it with
mocked rows and never touch AWS.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd

from . import config
from . import session as session_mod

# Resolved ideal-catalog location (ported from the app's data layer — NOT guessed).
# Sourced from curve.config (env-overridable; defaults equal the prior literals).
IDEAL_CATALOG_CATALOG = config.IDEAL_CATALOG_CATALOG
IDEAL_CATALOG_DATABASE = config.IDEAL_CATALOG_DATABASE
IDEAL_CATALOG_TABLE = config.IDEAL_CATALOG_TABLE
IDEAL_CATALOG_S3_OUTPUT = config.ATHENA_S3_OUTPUT
IDEAL_CATALOG_REGION = config.AWS_REGION

# The app's BEP-tolerance slider default (0.25 = ±25%). Plumbed as a setup-injected
# parameter so the Streamlit slider maps straight onto it (see :func:`bep_tolerance_from_context`).
DEFAULT_BEP_TOLERANCE = config.DEFAULT_BEP_TOLERANCE

# Curve-reconstruction coeffs (vendored ``compute.ideal_curve_overlay`` reads
# ``ideal_head_c1..c6`` + ``ideal_power_c1..c6``). All 12 must be finite for a row to
# be SELECTABLE — 4b reconstructs the H-Q / power curves from these.
HEAD_COEFF_COLS: List[str] = [f"ideal_head_c{i}" for i in range(1, 7)]
POWER_COEFF_COLS: List[str] = [f"ideal_power_c{i}" for i in range(1, 7)]
CURVE_COEFF_COLS: List[str] = HEAD_COEFF_COLS + POWER_COEFF_COLS

# The flow-character columns the BEP narrowing reads. ``bep_bpd`` (the best-efficiency
# flow) is DISTINCT from ``min/max_recommended_bpd`` (the recommended operating band) —
# the two-stage filter uses all three; do not collapse them (guardrail 4).
BEP_BPD_COL = "bep_bpd"
MIN_RECOMMENDED_BPD_COL = "min_recommended_bpd"
MAX_RECOMMENDED_BPD_COL = "max_recommended_bpd"
BEP_RANGE_COLS: List[str] = [BEP_BPD_COL, MIN_RECOMMENDED_BPD_COL, MAX_RECOMMENDED_BPD_COL]

# Identity columns. ``pump_id`` is the KEY (the pick rides on it); ``esp_model`` is the
# DISPLAY label — do not conflate them (guardrail 4).
PUMP_ID_COL = "pump_id"
ESP_MODEL_COL = "esp_model"
IDENTITY_COLS: List[str] = [PUMP_ID_COL, "manufacturer", "series", ESP_MODEL_COL]

# The columns a fetched catalog must carry for this layer to function. If any are
# absent, the schema doesn't line up with the app's read → resolve-or-stop (guardrail
# 11): we raise rather than guess a substitute column.
REQUIRED_COLUMNS: List[str] = [PUMP_ID_COL, ESP_MODEL_COL, *BEP_RANGE_COLS, *CURVE_COEFF_COLS]


# --- credentials (lazy; building a query needs no creds) ----------------------


def _build_session(
    profile_name: Optional[str] = None, region_name: str = IDEAL_CATALOG_REGION
) -> boto3.Session:
    """Profile precedence identical to curve.data: explicit → AWS_PROFILE → chain."""
    if profile_name:
        return boto3.Session(profile_name=profile_name, region_name=region_name)
    if os.environ.get("AWS_PROFILE"):
        return boto3.Session(
            profile_name=os.environ["AWS_PROFILE"], region_name=region_name
        )
    return boto3.Session(region_name=region_name)  # role / env / default chain


def build_catalog_query() -> str:
    """The ideal-catalog query (pure string — testable without AWS).

    Mirrors the app's ``load_ideal_catalog`` projection: a full read of the canonical
    library (no obsolete-drop here — that is an app-side selector concern CurVE
    deliberately reverses; see module docstring deviation #1).
    """
    return f"SELECT * FROM {IDEAL_CATALOG_DATABASE}.{IDEAL_CATALOG_TABLE}"


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Resolve-or-stop (guardrail 11): the fetched schema must carry what we read."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Ideal catalog schema does not line up with the app's read — missing "
            f"columns: {', '.join(missing)}. Refusing to guess substitutes; resolve "
            "the catalog/schema against app/data/ideal_pump_catalog.py and retry."
        )


def fetch_ideal_catalog(
    *,
    profile_name: Optional[str] = None,
    region_name: str = IDEAL_CATALOG_REGION,
    s3_output: str = IDEAL_CATALOG_S3_OUTPUT,
    session: Optional[boto3.Session] = None,
) -> pd.DataFrame:
    """Read the full ideal-pump catalog from Athena (re-read each session).

    Returns the catalog as a DataFrame with identity fields stripped to strings. Unlike
    the app's loader this does NOT drop obsolete rows (deviation #1) — selectability +
    obsolete flagging happen downstream in :func:`annotate_candidates`. Raises if the
    schema doesn't carry the columns this layer reads (resolve-or-stop, guardrail 11).

    Kept as a free function so cred-free tests monkeypatch
    ``curve.ideal_catalog.fetch_ideal_catalog`` with mocked rows and never touch AWS.
    """
    # Imported lazily so importing this module (and cred-free tests that monkeypatch
    # this fetch) does not require awswrangler at import time.
    import awswrangler as wr

    sess = session or _build_session(profile_name=profile_name, region_name=region_name)
    df = wr.athena.read_sql_query(
        sql=build_catalog_query(),
        database=IDEAL_CATALOG_DATABASE,
        data_source=IDEAL_CATALOG_CATALOG,
        boto3_session=sess,
        ctas_approach=False,
        s3_output=s3_output,
    )
    _validate_required_columns(df)
    # Normalize identity fields used by the pick selectors (mirrors the app's strip).
    for col in IDENTITY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)


# --- total fluid (the BEP-narrowing input) ------------------------------------


def resolve_total_fluid_bpd(analyzed_df: Optional[pd.DataFrame]) -> Optional[float]:
    """Representative total fluid (LIQUID rate, bpd) for BEP narrowing.

    Reads ``median(liquid_rate_bbl_day)`` off the vendored-analysis frame — exactly the
    app's preprocessed overlay (``preprocessed_page.py`` L652). Total fluid is oil +
    water (the liquid rate), NOT the oil rate (guardrail 4). Returns ``None`` when the
    column is absent or has no positive values (the narrowing then yields no candidates,
    surfaced honestly — never a fabricated rate).
    """
    if analyzed_df is None or "liquid_rate_bbl_day" not in getattr(analyzed_df, "columns", []):
        return None
    series = pd.to_numeric(analyzed_df["liquid_rate_bbl_day"], errors="coerce").dropna()
    if series.empty:
        return None
    median = float(series.median())
    return median if median > 0 else None


# --- BEP narrowing (ported verbatim from the app's narrow_catalog) ------------


def narrow_candidates(
    catalog: pd.DataFrame,
    total_fluid_bpd: Optional[float],
    bep_tolerance: float = DEFAULT_BEP_TOLERANCE,
) -> pd.DataFrame:
    """BEP-narrow the catalog by the well's total fluid — the app's exact two-stage AND.

    Ported verbatim from ``ideal_curve_service.narrow_catalog``::

        stage 1 (flow-range containment): min_recommended_bpd <= rate <= max_recommended_bpd
        stage 2 (BEP proximity):          bep_bpd*(1-tol) <= rate <= bep_bpd*(1+tol)
        return catalog[stage1 & stage2]

    A non-positive / unparseable rate returns an empty frame (no candidates), matching
    the app. NaN in any of the three flow columns makes that row's comparisons False, so
    it drops out — same as the app. This narrows by FLOW only; selectability (curve
    coeffs) and obsolete flagging are layered on by :func:`annotate_candidates`.
    """
    rate = pd.to_numeric(pd.Series([total_fluid_bpd]), errors="coerce").iloc[0]
    if pd.isna(rate) or rate <= 0:
        return catalog.iloc[0:0].copy()

    min_col = pd.to_numeric(catalog[MIN_RECOMMENDED_BPD_COL], errors="coerce")
    max_col = pd.to_numeric(catalog[MAX_RECOMMENDED_BPD_COL], errors="coerce")
    bep_col = pd.to_numeric(catalog[BEP_BPD_COL], errors="coerce")

    stage1 = (min_col <= rate) & (rate <= max_col)
    stage2 = (bep_col * (1 - bep_tolerance) <= rate) & (rate <= bep_col * (1 + bep_tolerance))
    return catalog[stage1 & stage2].copy()


# --- selectability + obsolete (the CurVE deviations) --------------------------


def selectable_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series — True where every curve coeff is present and finite.

    A row is selectable only when all 12 ``ideal_head_c*`` / ``ideal_power_c*`` coeffs
    parse to finite numbers (4b reconstructs the curve from them). Missing/incomplete
    coeffs → not selectable (guardrail 3); such rows are excluded from candidates but
    counted in the coverage report, never silently dropped.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype=bool)
    mask = pd.Series(True, index=df.index)
    for col in CURVE_COEFF_COLS:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= pd.to_numeric(df[col], errors="coerce").notna()
    return mask


def obsolete_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series — True where ``is_obsolete`` reads truthy (true/1/yes).

    Surfaced, NOT dropped (guardrail 3 / deviation #1). When the column is absent every
    row is treated as not-obsolete (the app's catalog may omit it on some snapshots).
    """
    if df is None or len(df) == 0 or "is_obsolete" not in df.columns:
        return pd.Series(False, index=getattr(df, "index", None))
    flags = df["is_obsolete"].astype(str).str.strip().str.lower()
    return flags.isin(["true", "1", "yes"])


def _num(value: Any) -> Optional[float]:
    """Parse to a finite float, else None (JSON-friendly for the report / pick)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def annotate_candidates(narrowed: pd.DataFrame) -> List[Dict[str, Any]]:
    """Shape BEP-narrowed rows into candidate dicts with selectable + obsolete flags.

    Each candidate carries the identity (``pump_id`` key + ``esp_model`` display, kept
    distinct), the flow character (``bep_bpd`` vs ``min/max_recommended_bpd``, kept
    distinct), ``selectable`` (curve coeffs present), and ``is_obsolete`` (surfaced, not
    dropped). Non-selectable and obsolete rows are BOTH included here — the coverage
    report and the pick logic decide what to do with the flags, this layer only honestly
    annotates them.
    """
    if narrowed is None or len(narrowed) == 0:
        return []
    sel = selectable_mask(narrowed)
    obs = obsolete_mask(narrowed)
    candidates: List[Dict[str, Any]] = []
    for idx, row in narrowed.iterrows():
        candidates.append(
            {
                "pump_id": str(row.get(PUMP_ID_COL, "")),
                "esp_model": str(row.get(ESP_MODEL_COL, "")),
                "manufacturer": str(row.get("manufacturer", "")),
                "series": str(row.get("series", "")),
                "bep_bpd": _num(row.get(BEP_BPD_COL)),
                "min_recommended_bpd": _num(row.get(MIN_RECOMMENDED_BPD_COL)),
                "max_recommended_bpd": _num(row.get(MAX_RECOMMENDED_BPD_COL)),
                "selectable": bool(sel.loc[idx]),
                "is_obsolete": bool(obs.loc[idx]),
            }
        )
    return candidates


def selectable_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The subset an operator may actually pick — selectable rows (obsolete included)."""
    return [c for c in candidates if c.get("selectable")]


# --- per-well coverage report (the de-risking check, not a gate) --------------


def build_coverage_report(
    catalog: pd.DataFrame,
    total_fluid_bpd: Optional[float],
    well_id: str,
    bep_tolerance: float = DEFAULT_BEP_TOLERANCE,
) -> Dict[str, Any]:
    """Per-well coverage report: total fluid, candidate count, candidate list.

    The de-risking check the build plan asks for — NOT a gate (it never blocks). Reports
    the well's total fluid, the BEP tolerance used, how many catalog rows are
    BEP-compatible, how many of those are selectable (the candidate list the operator
    picks from), and how many were surfaced obsolete or excluded for missing coeffs.
    """
    narrowed = narrow_candidates(catalog, total_fluid_bpd, bep_tolerance)
    candidates = annotate_candidates(narrowed)
    selectable = selectable_candidates(candidates)
    n_obsolete = sum(1 for c in selectable if c.get("is_obsolete"))
    n_excluded_missing_coeffs = sum(1 for c in candidates if not c.get("selectable"))
    return {
        "well_id": well_id,
        "total_fluid_bpd": _num(total_fluid_bpd),
        "bep_tolerance": float(bep_tolerance),
        "n_catalog": int(len(catalog)) if catalog is not None else 0,
        "n_bep_compatible": len(candidates),
        "n_candidates": len(selectable),
        "n_excluded_missing_coeffs": n_excluded_missing_coeffs,
        "n_obsolete_surfaced": n_obsolete,
        "candidates": selectable,
        # The full BEP-compatible set incl. the not-selectable rows, so the report can
        # show WHY a row was excluded rather than hiding it (guardrail 3 honesty).
        "bep_compatible": candidates,
    }


# --- the pump pick: a setup-injected value (rides the M3 override path) --------


def make_pump_pick(
    catalog: pd.DataFrame,
    pump_id: str,
    *,
    bep_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Shape the operator's manual pump pick into the canonical setup-injected value.

    Keyed by ``pump_id`` (the KEY); carries ``esp_model`` (the DISPLAY label) and the
    flow character alongside — kept distinct (guardrail 4). ``source`` is always
    ``manual`` in v1 (the §1 ladder's manual rung); Track 2's auto-connection will set a
    different source with zero change here. ``selectable`` / ``is_obsolete`` are carried
    so a downstream block-when-unresolved (4b) can refuse a non-selectable pump and a
    narration can disclose an obsolete one — this layer flags, it does not reconcile.

    Raises ``KeyError`` when ``pump_id`` is not in the catalog (no silent default pump —
    no pick is an honest blocked state downstream, never a guess).
    """
    if catalog is None or PUMP_ID_COL not in getattr(catalog, "columns", []):
        raise KeyError(f"catalog carries no '{PUMP_ID_COL}' column; cannot resolve pick.")
    match = catalog[catalog[PUMP_ID_COL].astype(str).str.strip() == str(pump_id).strip()]
    if match.empty:
        raise KeyError(f"pump_id not found in catalog: {pump_id!r}")
    row = match.iloc[0]
    selectable = bool(selectable_mask(match).iloc[0])
    obsolete = bool(obsolete_mask(match).iloc[0])
    pick = {
        "pump_id": str(row.get(PUMP_ID_COL, "")),
        "esp_model": str(row.get(ESP_MODEL_COL, "")),
        "manufacturer": str(row.get("manufacturer", "")),
        "series": str(row.get("series", "")),
        "bep_bpd": _num(row.get(BEP_BPD_COL)),
        "min_recommended_bpd": _num(row.get(MIN_RECOMMENDED_BPD_COL)),
        "max_recommended_bpd": _num(row.get(MAX_RECOMMENDED_BPD_COL)),
        "selectable": selectable,
        "is_obsolete": obsolete,
        "source": "manual",  # §1 ladder: v1 = manual; Track 2 flips covered wells to auto
    }
    if bep_tolerance is not None:
        pick["bep_tolerance"] = float(bep_tolerance)
    return pick


def bep_tolerance_from_context(resolved_inputs: Optional[Dict[str, Any]]) -> float:
    """Read the BEP tolerance off the session's setup-injected context (slider → param).

    Mirrors how depth/SG overrides and ``recommendation_method`` are read from
    ``resolved_inputs`` — the operator's slider value rides the same setup-injected path,
    NOT a model-facing argument. Falls back to the app's 0.25 default.
    """
    ctx = resolved_inputs or {}
    val = _num(ctx.get("bep_tolerance"))
    return val if (val is not None and val > 0) else DEFAULT_BEP_TOLERANCE


def set_pump_on_session(record: Dict[str, Any], pump: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Persist the pump pick onto the session record's dedicated ``pump`` field.

    Mirrors exactly how the Streamlit setup writes depth/SG overrides onto
    ``resolved_inputs`` (``streamlit_app._render_delta_p_input_overrides``): mutate the
    record, then ``save_session``. The engine then injects ``session['pump']`` into
    every tool call (top-level, like org/well) — so the pick is setup-injected and never
    a Converse argument. ``pump=None`` is an honest "no pick yet" (4b blocks on it),
    never a silent default pump.
    """
    record["pump"] = pump
    return session_mod.save_session(record)
