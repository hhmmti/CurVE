"""CurVE well enumeration — distinct orgs + org→wells map for the setup dropdowns (M5 step 3).

**Enumeration ONLY.** This reads the identity columns of the VE well-configuration
table so the Streamlit setup sidebar can offer dependent org → well dropdowns instead
of hand-typed ids. There is **no physics and no per-well data fetch** here — the chosen
org/well feed the exact same setup path (``session.new_session_record`` →
``_availability_probe``) that the typed inputs feed today. This module just lists what
is pickable.

WHERE THE READ CAME FROM (confirmed schema, not guessed)
--------------------------------------------------------
The VE well-configuration table lives in the **same catalog/database** as the
recommendation mirror (:mod:`curve.recommendations`). Its identity columns are
``organization_id`` and ``well_id`` — the **same spelling as the session-record
fields**, so the map to ``organization_id`` / ``well_id`` is direct (no rename)::

    catalog  : roam_prd_ddb            (DynamoDB via the Athena connector)
    database : default
    table    : esp_well_configuration_v2
    columns  : organization_id, well_id   (+ many config columns we do not read)

Read path + credential precedence mirror :mod:`curve.recommendations` **exactly**
(explicit ``profile_name`` → ``AWS_PROFILE`` env → default chain; region
``us-east-1``; ``wr.athena.read_sql_query`` with ``data_source`` = the catalog). No
new client or connection pattern is introduced.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import boto3

from curve import config

# VE well-configuration table — same catalog/database as the recommendation mirror.
# Sourced from curve.config (env-overridable; defaults equal the prior literals).
WELLCFG_CATALOG = config.WELLCFG_CATALOG  # DynamoDB via the Athena connector
WELLCFG_DATABASE = config.WELLCFG_DATABASE
WELLCFG_TABLE = config.WELLCFG_TABLE
WELLCFG_ATHENA_S3_OUTPUT = config.ATHENA_S3_OUTPUT
WELLCFG_REGION = config.AWS_REGION


def _build_session(
    profile_name: Optional[str] = None, region_name: str = WELLCFG_REGION
) -> boto3.Session:
    """Profile precedence identical to curve.recommendations: explicit → AWS_PROFILE → chain."""
    if profile_name:
        return boto3.Session(profile_name=profile_name, region_name=region_name)
    if os.environ.get("AWS_PROFILE"):
        return boto3.Session(
            profile_name=os.environ["AWS_PROFILE"], region_name=region_name
        )
    return boto3.Session(region_name=region_name)  # role / env / default chain


def build_enumeration_query() -> str:
    """Build the distinct-(org, well) enumeration query (pure string — testable).

    Selects only the two identity columns, drops null/empty ids, and orders for a
    stable dropdown. No org/well filter here on purpose: this lists every well the
    operator may pick from (the per-well confidentiality filter lives downstream in
    the actual data fetch, unchanged).
    """
    return (
        "SELECT DISTINCT organization_id, well_id "
        f"FROM {WELLCFG_DATABASE}.{WELLCFG_TABLE} "
        "WHERE organization_id IS NOT NULL AND organization_id <> '' "
        "AND well_id IS NOT NULL AND well_id <> '' "
        "ORDER BY organization_id, well_id"
    )


def fetch_org_well_map(
    *,
    profile_name: Optional[str] = None,
    region_name: str = WELLCFG_REGION,
    s3_output: str = WELLCFG_ATHENA_S3_OUTPUT,
    session: Optional[boto3.Session] = None,
) -> Dict[str, List[str]]:
    """Return ``{organization_id: [well_id, …]}`` from ``esp_well_configuration_v2``.

    An empty dict means the enumeration returned nothing — the caller falls back to
    manual entry with a visible notice (never a crash). Wells are de-duplicated and
    sorted per org.

    Kept as a free function (not a method) so cred-free tests can monkeypatch
    ``curve.well_catalog.fetch_org_well_map`` with a fixture and never touch AWS.
    """
    # Imported lazily so importing this module (and cred-free tests that monkeypatch
    # this fetch) does not require awswrangler at import time.
    import awswrangler as wr

    sess = session or _build_session(profile_name=profile_name, region_name=region_name)
    df = wr.athena.read_sql_query(
        sql=build_enumeration_query(),
        database=WELLCFG_DATABASE,
        data_source=WELLCFG_CATALOG,
        boto3_session=sess,
        ctas_approach=False,
        s3_output=s3_output,
    )

    org_wells: Dict[str, List[str]] = {}
    if df is None or df.empty:
        return org_wells
    for org, well in zip(df["organization_id"], df["well_id"]):
        if org is None or well is None:
            continue
        org_s, well_s = str(org).strip(), str(well).strip()
        if not org_s or not well_s:
            continue
        org_wells.setdefault(org_s, []).append(well_s)
    for org in org_wells:
        org_wells[org] = sorted(set(org_wells[org]))
    return org_wells
