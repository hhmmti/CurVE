"""CurVE standalone well-depth (TVD) lookup — the real rrc depth read (M4 / ΔP tools).

The ΔP history tools need a real well depth for the hydrostatic term. The week-8
physics app already reads RRC-mined depth; this module **ports that exact read** into
``playground`` as its own small awswrangler query (it does NOT copy the app's data
layer wholesale, and it does NOT re-implement physics).

WHERE THE READ CAME FROM (resolved during execution, per prompt #4 step 1):
  The app's depth access is ``app/data/well_depth_db.py`` → ``fetch_well_depth()``.
  It queries:

      catalog : roam_dev_products
      database : well_depth_dev
      table    : rrc_well_depth
      column   : "API Depth"   (RRC-mined API/total depth, exposed by the app as
                                 well_depth_ft for the hydrostatic correction)
      query    : SELECT "API Depth" FROM "well_depth_dev"."rrc_well_depth"
                 WHERE well_id = '<id>' LIMIT 1

  NAMING FLAG (carried into the change report): the prompt referred to an
  ``rrc_mining`` table and "TVD". The app's real source is the ``rrc_well_depth``
  table, column ``"API Depth"`` — there is no table literally named ``rrc_mining``,
  and the column is RRC API depth (which the app treats as the depth_ft fed to the
  purely-hydrostatic correction). We port the **real** identifiers rather than
  inventing ``rrc_mining`` / a ``TVD`` column. The depth is used for hydrostatic
  ``0.433 × SG × depth_ft`` exactly as the app uses it.

Credentials: the same precedence as :mod:`curve.data` (explicit ``profile_name`` →
``AWS_PROFILE`` env → default credential chain). Region ``us-east-1``. Returns the
depth as a float (feet) or ``None`` when no positive depth record is found — never a
fabricated number, and never the CurVE default (the caller's resolution layer applies
the default and labels it, so provenance stays explicit).
"""

from __future__ import annotations

import os
from typing import Optional

import boto3

# Ported verbatim from the app's data/well_depth_db.py (the known-good binding).
RRC_CATALOG = "roam_dev_products"
RRC_DATABASE = "well_depth_dev"
RRC_TABLE = "rrc_well_depth"
RRC_DEPTH_COLUMN = "API Depth"
RRC_ATHENA_S3_OUTPUT = "s3://esp-athena-results-v2-411237692998/"
RRC_REGION = "us-east-1"


def _build_session(
    profile_name: Optional[str] = None, region_name: str = RRC_REGION
) -> boto3.Session:
    """Profile precedence identical to curve.data: explicit → AWS_PROFILE → chain."""
    if profile_name:
        return boto3.Session(profile_name=profile_name, region_name=region_name)
    if os.environ.get("AWS_PROFILE"):
        return boto3.Session(
            profile_name=os.environ["AWS_PROFILE"], region_name=region_name
        )
    return boto3.Session(region_name=region_name)  # role / env / default chain


def build_depth_query(well_id: str) -> str:
    """Build the rrc depth query for one well (pure string — testable without AWS)."""
    safe_well_id = str(well_id).replace("'", "''")
    return (
        f'SELECT "{RRC_DEPTH_COLUMN}" '
        f'FROM "{RRC_DATABASE}"."{RRC_TABLE}" '
        f"WHERE well_id = '{safe_well_id}' LIMIT 1"
    )


def fetch_well_depth_ft(
    well_id: str,
    *,
    profile_name: Optional[str] = None,
    region_name: str = RRC_REGION,
    s3_output: str = RRC_ATHENA_S3_OUTPUT,
    session: Optional[boto3.Session] = None,
) -> Optional[float]:
    """Return the real RRC well depth in feet for ``well_id``, or ``None``.

    ``None`` means "no real depth on record" — the resolution layer then falls back
    to the operator override or the CurVE default and labels the result Estimated.
    Any AWS/data error is swallowed to ``None`` (mirrors the app behavior) so a depth
    miss degrades to the default chain rather than crashing the tool.

    Kept as a free function (not a method) so cred-free tests can monkeypatch
    ``curve.well_depth.fetch_well_depth_ft`` with a fixture and never touch AWS.
    """
    # Imported lazily so importing this module (and cred-free tests that monkeypatch
    # this fetch) does not require awswrangler at import time.
    import awswrangler as wr

    sess = session or _build_session(profile_name=profile_name, region_name=region_name)
    try:
        df = wr.athena.read_sql_query(
            sql=build_depth_query(well_id),
            database=RRC_DATABASE,
            data_source=RRC_CATALOG,
            boto3_session=sess,
            ctas_approach=False,
            s3_output=s3_output,
        )
        if df is None or df.empty:
            return None
        val = df.iloc[0][RRC_DEPTH_COLUMN]
        if val is None:
            return None
        fval = float(val)
        return fval if fval > 0 else None
    except Exception:
        return None
