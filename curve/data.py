"""CurVE data access — real Athena fetch for the preprocessed telemetry + production tables (M2).

This is the M2 data seam the M0 audit deliberately left open: the vendored
``services/`` and ``compute/`` layers are surface-independent and carry no I/O, so
the only new code needed to make a tool *real* is this thin awswrangler fetch.

What this module does (and only this):
  * fetch preprocessed **telemetry** and **production** for ONE ``(organization_id,
    well_id)`` over an optional ``observation_day`` window, from the VE-canonical
    ``esp_optimization_v2`` tables, via ``awswrangler.athena.read_sql_query``;
  * return the two raw dataframes. The **join + feature engineering** is the
    vendored ``services.preprocessed_pipeline_service`` (the tool calls it next) —
    we do not re-implement it here.

Catalog binding: ``roam_prd_products.esp_optimization_v2`` — the app's data layer
binding, confirmed identical (same rows/values) to the VE's
``esp_optimization_v2`` path. Pinned per the §4 "Athena source must be pinned"
carry-forward; swappable via constructor args if the VE binding diverges.

Confidentiality (cross-org): EVERY query filters on ``organization_id`` AND
``well_id`` together — a fetch without both is a ``ValueError``, never a broad scan.

Credentials: the same handling as the M1 wrapper — explicit ``profile_name`` →
``AWS_PROFILE`` env → default credential chain (so the eventual in-Lambda role path
is not broken). Region ``us-east-1``.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import boto3
import pandas as pd

# VE-canonical preprocessed source. Catalog confirmed identical to roam_prd_products.
CURVE_CATALOG = "roam_prd_products"
CURVE_DATABASE = "esp_optimization_v2"
CURVE_TELEMETRY_TABLE = "esp_telemetry_preprocessed"
CURVE_PRODUCTION_TABLE = "esp_production_preprocessed"
CURVE_ATHENA_S3_OUTPUT = "s3://esp-athena-results-v2-411237692998/"
CURVE_REGION = "us-east-1"


def _quote(value: str) -> str:
    """Single-quote a literal for SQL, escaping embedded quotes.

    org/well come from the trusted session record (backend-injected, never operator
    free-text), but we escape anyway so an apostrophe in an id can't break the query.
    """
    return "'" + str(value).replace("'", "''") + "'"


class PreprocessedDataAccess:
    """awswrangler-backed reader for the preprocessed telemetry + production tables.

    Mirrors the app's ``data/preprocessed_db.py`` query shape (the known-good
    binding) but trimmed to the single fetch CurVE's ``production_history`` needs.
    """

    def __init__(
        self,
        profile_name: Optional[str] = None,
        region_name: str = CURVE_REGION,
        catalog: str = CURVE_CATALOG,
        database: str = CURVE_DATABASE,
        telemetry_table: str = CURVE_TELEMETRY_TABLE,
        production_table: str = CURVE_PRODUCTION_TABLE,
        s3_output: str = CURVE_ATHENA_S3_OUTPUT,
    ):
        self.profile_name = profile_name
        self.region_name = region_name
        self.catalog = catalog
        self.database = database
        self.telemetry_table = telemetry_table
        self.production_table = production_table
        self.s3_output = s3_output
        self._session: Optional[boto3.Session] = None

    # -- credentials (lazy; building the accessor needs no creds) ----------------

    def _build_session(self) -> boto3.Session:
        """Profile precedence identical to the M1 wrapper's: explicit → env → chain."""
        if self.profile_name:
            return boto3.Session(
                profile_name=self.profile_name, region_name=self.region_name
            )
        if os.environ.get("AWS_PROFILE"):
            return boto3.Session(
                profile_name=os.environ["AWS_PROFILE"], region_name=self.region_name
            )
        return boto3.Session(region_name=self.region_name)  # role / env / default chain

    @property
    def session(self) -> boto3.Session:
        if self._session is None:
            self._session = self._build_session()
        return self._session

    # -- query builders (pure strings — unit-testable without AWS) ---------------

    def _build_window_query(
        self,
        table: str,
        organization_id: str,
        well_id: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> str:
        # Cross-org guard: both keys are mandatory, always ANDed. No broad scans.
        if not organization_id or not well_id:
            raise ValueError(
                "Both organization_id and well_id are required (cross-org "
                "confidentiality); refusing to query without both."
            )
        clauses = [
            f"organization_id = {_quote(organization_id)}",
            f"well_id = {_quote(well_id)}",
        ]
        if start_date:
            clauses.append(f"observation_day >= {_quote(start_date)}")
        if end_date:
            clauses.append(f"observation_day <= {_quote(end_date)}")
        where = " AND ".join(clauses)
        return (
            f"SELECT * FROM {self.database}.{table} "
            f"WHERE {where} ORDER BY observation_day"
        )

    def build_telemetry_query(
        self,
        organization_id: str,
        well_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        return self._build_window_query(
            self.telemetry_table, organization_id, well_id, start_date, end_date
        )

    def build_production_query(
        self,
        organization_id: str,
        well_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        return self._build_window_query(
            self.production_table, organization_id, well_id, start_date, end_date
        )

    # -- the fetch ---------------------------------------------------------------

    def _read_sql(self, query: str) -> pd.DataFrame:
        # Imported lazily so importing this module (and running cred-free tests that
        # monkeypatch the fetch) does not require awswrangler at import time.
        import awswrangler as wr

        return wr.athena.read_sql_query(
            sql=query,
            database=self.database,
            boto3_session=self.session,
            ctas_approach=False,
            data_source=self.catalog,
            s3_output=self.s3_output,
        )

    def fetch_window(
        self,
        organization_id: str,
        well_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(telemetry_df, production_df)`` for one well over a window.

        Both frames are filtered on ``organization_id`` AND ``well_id``. The join +
        feature engineering is the caller's next step (vendored service), not here.
        """
        telem = self._read_sql(
            self.build_telemetry_query(organization_id, well_id, start_date, end_date)
        )
        prod = self._read_sql(
            self.build_production_query(organization_id, well_id, start_date, end_date)
        )
        return telem, prod


def fetch_preprocessed_window(
    organization_id: str,
    well_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    profile_name: Optional[str] = None,
    region_name: str = CURVE_REGION,
    access: Optional[PreprocessedDataAccess] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Module-level fetch entry point the ``production_history`` tool calls.

    Kept as a free function (not just a method) so cred-free tests can monkeypatch
    ``curve.data.fetch_preprocessed_window`` with a fixture and never touch AWS.
    """
    access = access or PreprocessedDataAccess(
        profile_name=profile_name, region_name=region_name
    )
    return access.fetch_window(organization_id, well_id, start_date, end_date)
