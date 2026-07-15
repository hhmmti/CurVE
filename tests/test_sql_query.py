"""CurVE M2 — adversarial allow/deny suite for the ``sql_query`` guard + plane.

This is the milestone's **acceptance gate**. It runs with NO AWS credentials: the
``guard`` tests are pure (parse + rewrite only), and the ``explain`` / ``execute``
tests monkeypatch the Athena client with a fake so the choke-point, timeout, and
result shaping are exercised without touching AWS.

Coverage mirrors the Definition of Done:
  * ALLOW (pass, post-injection): plain SELECT on a scoped table (org+well + default
    date + LIMIT injected); ideal (no scoping); rrc (well-only); explicit in-range
    observation_day respected; explicit LIMIT <= cap respected.
  * DENY: DDL/DML; stacked; out-of-allowlist at top level AND hidden in
    subquery/CTE/JOIN/UNION; any model-supplied org/well predicate; LIMIT > cap;
    unparseable; a partitioned query with no coverage (defense-in-depth).
  * plane: choke-point refuses raw / forged SQL; EXPLAIN maps ok/error; execute
    returns the full result; timeout cancels and aborts.
"""

from __future__ import annotations

import pytest

from curve import config
from curve import sql_query
from curve.sql_query import (
    GuardedSQL,
    SqlExecutionError,
    execute,
    explain,
    guard,
)

# --- fully-qualified allowlist tables (must match schema_catalog / config) -----

T = "roam_prd_products.esp_optimization_v2.esp_telemetry_preprocessed"  # org+well+date
P = "roam_prd_products.esp_optimization_v2.esp_production_preprocessed"  # org+well+date
R = "roam_prd_ddb.default.esp_setpoint_recommendations_v2"  # org+well, federated, no date
D = "roam_dev_products.well_depth_dev.rrc_well_depth"  # well-only
I = "roam_dev_products.esp_ideal_pump_dev.ideal_pump_library_v1"  # no scoping

CAP = config.SQL_QUERY_LIMIT_CAP


@pytest.fixture
def session():
    return {
        "organization_id": "greenlakeenergy",
        "well_id": "W-42",
        "availability": {"coverage": {"min_day": "2024-01-01", "max_day": "2024-06-30"}},
    }


def _ok(sql, session):
    r = guard(sql, session)
    assert r.ok, f"expected ALLOW, got DENY:{r.rejection.code} — {r.rejection.reason}"
    return r.guarded.sql


def _deny(sql, session):
    r = guard(sql, session)
    assert not r.ok, f"expected DENY, got ALLOW -> {r.guarded.sql if r.ok else ''}"
    return r.rejection.code


# =========================== ALLOW ============================================


def test_allow_plain_scoped_injects_org_well_date_and_limit(session):
    out = _ok(f"SELECT motor_amps FROM {T}", session)
    assert "esp_telemetry_preprocessed.organization_id = 'greenlakeenergy'" in out
    assert "esp_telemetry_preprocessed.well_id = 'W-42'" in out
    assert "observation_day >= '2024-01-01'" in out
    assert "observation_day <= '2024-06-30'" in out
    assert out.rstrip().endswith(f"LIMIT {CAP}")


def test_allow_production_scoped_same_as_telemetry(session):
    out = _ok(f"SELECT alloc_oil_vol FROM {P}", session)
    assert "esp_production_preprocessed.organization_id = 'greenlakeenergy'" in out
    assert "esp_production_preprocessed.well_id = 'W-42'" in out
    assert "observation_day >=" in out


def test_allow_ideal_reference_no_scoping_injected(session):
    out = _ok(f"SELECT pump_id FROM {I} WHERE series = '4000'", session)
    assert "organization_id" not in out  # no org/well ever added to the ref table
    assert "well_id" not in out
    assert "observation_day" not in out  # ideal is unpartitioned
    assert out.rstrip().endswith(f"LIMIT {CAP}")


def test_allow_rrc_injects_well_only_no_org_no_date(session):
    out = _ok(f'SELECT "api depth" FROM {D}', session)
    assert "rrc_well_depth.well_id = 'W-42'" in out
    assert "organization_id" not in out  # rrc is well-only scoped
    assert "observation_day" not in out  # rrc is unpartitioned


def test_allow_recs_federated_injects_org_well_no_date(session):
    out = _ok(f"SELECT model_version FROM {R}", session)
    assert "organization_id = 'greenlakeenergy'" in out
    assert "well_id = 'W-42'" in out
    assert "observation_day" not in out  # federated recs is unpartitioned


def test_allow_explicit_in_range_observation_day_respected(session):
    out = _ok(f"SELECT motor_amps FROM {T} WHERE observation_day >= '2024-03-01'", session)
    assert "observation_day >= '2024-03-01'" in out
    # guard did NOT inject its own default window on top of the model's date filter
    assert "'2024-01-01'" not in out
    assert "'2024-06-30'" not in out


def test_allow_explicit_limit_under_cap_respected(session):
    out = _ok(f"SELECT motor_amps FROM {T} LIMIT 50", session)
    assert out.rstrip().endswith("LIMIT 50")


def test_allow_limit_exactly_cap_respected(session):
    out = _ok(f"SELECT motor_amps FROM {T} LIMIT {CAP}", session)
    assert out.rstrip().endswith(f"LIMIT {CAP}")


def test_allow_cte_scoped_table_gets_injection_inside_cte(session):
    out = _ok(f"WITH c AS (SELECT well_id, motor_amps FROM {T}) SELECT * FROM c", session)
    # injection lands inside the CTE body (the scope that references the base table)
    assert "esp_telemetry_preprocessed.organization_id = 'greenlakeenergy'" in out
    assert out.rstrip().endswith(f"LIMIT {CAP}")


def test_allow_join_two_scoped_tables_each_injected(session):
    sql = (
        f"SELECT t.motor_amps, p.alloc_oil_vol FROM {T} t "
        f"JOIN {P} p ON t.observation_day = p.observation_day"
    )
    out = _ok(sql, session)
    assert "t.organization_id = 'greenlakeenergy'" in out
    assert "p.organization_id = 'greenlakeenergy'" in out
    assert "t.well_id = 'W-42'" in out and "p.well_id = 'W-42'" in out


def test_allow_union_of_allowlist_tables(session):
    out = _ok(f"SELECT pump_id AS k FROM {I} UNION SELECT well_id AS k FROM {D}", session)
    assert "rrc_well_depth.well_id = 'W-42'" in out  # scoped side injected
    assert out.rstrip().endswith(f"LIMIT {CAP}")


def test_allow_reported_tables_are_the_referenced_allowlist_fqns(session):
    r = guard(f"SELECT t.motor_amps FROM {T} t", session)
    assert r.ok and r.guarded.tables == (T,)


def test_allow_scoped_table_in_where_in_subquery_injected(session):
    # A scoped table hidden inside WHERE … IN (subquery) must be org/well-injected in
    # its OWN (inner) scope, alias-qualified — the cross-org gate reaches every scope.
    sql = (
        f"SELECT motor_amps FROM {T} "
        f"WHERE observation_day IN (SELECT observation_day FROM {P} WHERE alloc_oil_vol > 0)"
    )
    out = _ok(sql, session)
    # inner production scope carries its own org/well predicate
    assert "esp_production_preprocessed.organization_id = 'greenlakeenergy'" in out
    assert "esp_production_preprocessed.well_id = 'W-42'" in out
    # outer telemetry scope still scoped too
    assert "esp_telemetry_preprocessed.well_id = 'W-42'" in out


def test_allow_scoped_table_in_scalar_subquery_injected_with_date(session):
    # SELECT-list scalar subquery over a partitioned scoped table: inner scope gets
    # org/well AND the default date window (no observation_day supplied anywhere here).
    sql = f"SELECT (SELECT max(alloc_oil_vol) FROM {P}) AS x, motor_amps FROM {T}"
    out = _ok(sql, session)
    assert "esp_production_preprocessed.organization_id = 'greenlakeenergy'" in out
    assert "esp_production_preprocessed.well_id = 'W-42'" in out
    assert "esp_production_preprocessed.observation_day >= '2024-01-01'" in out
    assert "esp_production_preprocessed.observation_day <= '2024-06-30'" in out


# =========================== DENY =============================================


@pytest.mark.parametrize(
    "sql",
    [
        f"DROP TABLE {T}",
        f"DELETE FROM {T}",
        f"UPDATE {T} SET motor_amps = 0",
        f"INSERT INTO {T} VALUES (1)",
        f"CREATE TABLE x AS SELECT * FROM {I}",  # CTAS
        f"MERGE INTO {T} USING {P} ON true WHEN MATCHED THEN DELETE",
    ],
)
def test_deny_ddl_dml(sql, session):
    assert _deny(sql, session) == "not_read_only"


def test_deny_input_describe_and_show(session):
    assert _deny(f"DESCRIBE {T}", session) == "not_read_only"
    assert _deny(f"SHOW TABLES", session) == "not_read_only"


def test_deny_stacked_statements(session):
    assert _deny(f"SELECT 1 FROM {I}; SELECT 2 FROM {I}", session) == "multi_statement"


def test_deny_out_of_allowlist_top_level(session):
    bad = "roam_prd_products.esp_optimization_v2.secret_table"
    assert _deny(f"SELECT * FROM {bad}", session) == "not_allowlisted"


def test_deny_out_of_allowlist_hidden_in_subquery(session):
    bad = "roam_prd_products.esp_optimization_v2.secret_table"
    sql = f"SELECT pump_id FROM {I} WHERE pump_id IN (SELECT pump_id FROM {bad})"
    assert _deny(sql, session) == "not_allowlisted"


def test_deny_out_of_allowlist_hidden_in_cte(session):
    bad = "roam_prd_products.esp_optimization_v2.secret_table"
    sql = f"WITH c AS (SELECT * FROM {bad}) SELECT * FROM c"
    assert _deny(sql, session) == "not_allowlisted"


def test_deny_out_of_allowlist_hidden_in_join(session):
    bad = "roam_prd_products.esp_optimization_v2.secret_table"
    sql = f"SELECT t.motor_amps FROM {T} t JOIN {bad} s ON t.well_id = s.well_id"
    assert _deny(sql, session) == "not_allowlisted"


def test_deny_out_of_allowlist_hidden_in_union(session):
    bad = "roam_prd_products.esp_optimization_v2.secret_table"
    sql = f"SELECT pump_id AS k FROM {I} UNION SELECT k FROM {bad}"
    assert _deny(sql, session) == "not_allowlisted"


def test_deny_underqualified_table_reference(session):
    # bare table name (no catalog.db) is NOT a fully-qualified allowlist match
    assert _deny("SELECT * FROM esp_telemetry_preprocessed", session) == "not_allowlisted"


def test_deny_model_supplied_org_predicate(session):
    assert _deny(f"SELECT motor_amps FROM {T} WHERE organization_id = 'acme'", session) == (
        "model_supplied_scope"
    )


def test_deny_model_supplied_well_predicate(session):
    assert _deny(f"SELECT motor_amps FROM {T} WHERE well_id = 'W-9'", session) == (
        "model_supplied_scope"
    )


def test_deny_model_supplied_well_predicate_in_join_on(session):
    # even a JOIN-ON equality on a scoping column is a protocol violation (guard owns it)
    sql = (
        f"SELECT t.motor_amps FROM {T} t JOIN {P} p "
        f"ON t.well_id = p.well_id AND t.observation_day = p.observation_day"
    )
    assert _deny(sql, session) == "model_supplied_scope"


def test_deny_limit_over_cap(session):
    assert _deny(f"SELECT motor_amps FROM {T} LIMIT {CAP + 1}", session) == "limit_over_cap"


def test_deny_non_integer_limit(session):
    # LIMIT that isn't a verifiable integer literal cannot be checked -> fail closed
    assert _deny(f"SELECT motor_amps FROM {T} LIMIT (1 + 1)", session) == (
        "limit_unverifiable"
    )


def test_deny_unparseable(session):
    assert _deny("SELECT FROM WHERE )(", session) == "parse_error"


def test_deny_empty(session):
    assert _deny("   ", session) == "empty"


def test_deny_partitioned_without_coverage_is_defense_in_depth():
    # No coverage range in the session AND no model date -> refuse (independent of
    # injection): a partitioned scan must never go out unbounded on time.
    sess = {"organization_id": "o", "well_id": "w", "availability": {}}
    assert _deny(f"SELECT motor_amps FROM {T}", sess) == "coverage_missing"


def test_deny_missing_session_scope():
    assert _deny(f"SELECT motor_amps FROM {T}", {"organization_id": "", "well_id": ""}) == (
        "session_scope_missing"
    )


# =========================== execution plane ==================================


class _FakeAthena:
    """Minimal Athena stub: scripts a state sequence + optional stop capture."""

    def __init__(self, states, reason="", stats=None):
        self._states = list(states)
        self._reason = reason
        self._stats = stats or {}
        self.stopped = []

    def start_query_execution(self, **kw):
        self.query = kw["QueryString"]
        return {"QueryExecutionId": "qid-123"}

    def get_query_execution(self, **kw):
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return {
            "QueryExecution": {
                "Status": {"State": state, "StateChangeReason": self._reason},
                "Statistics": self._stats,
            }
        }

    def stop_query_execution(self, QueryExecutionId):
        self.stopped.append(QueryExecutionId)


def _patch_client(monkeypatch, fake):
    monkeypatch.setattr(sql_query, "_athena_client", lambda: fake)


def _guarded(sql, session):
    r = guard(sql, session)
    assert r.ok
    return r.guarded


def test_choke_point_rejects_raw_string():
    with pytest.raises(SqlExecutionError, match="choke-point"):
        explain("SELECT 1")
    with pytest.raises(SqlExecutionError, match="choke-point"):
        execute("SELECT 1")


def test_choke_point_rejects_forged_guarded_sql():
    forged = GuardedSQL(sql="SELECT 1", tables=())  # no valid token
    with pytest.raises(SqlExecutionError, match="choke-point"):
        explain(forged)


def test_explain_ok_on_success(monkeypatch, session):
    fake = _FakeAthena(states=["SUCCEEDED"])
    _patch_client(monkeypatch, fake)
    g = _guarded(f"SELECT motor_amps FROM {T}", session)
    assert explain(g) == {"ok": True, "error": None}
    assert fake.query.startswith("EXPLAIN ")  # runs EXPLAIN on the guarded SQL


def test_explain_error_string_is_parseable(monkeypatch, session):
    fake = _FakeAthena(states=["FAILED"], reason="line 1: bad column x")
    _patch_client(monkeypatch, fake)
    g = _guarded(f"SELECT motor_amps FROM {T}", session)
    out = explain(g)
    assert out["ok"] is False and "bad column x" in out["error"]


def test_execute_returns_full_result(monkeypatch, session):
    import pandas as pd

    fake = _FakeAthena(states=["SUCCEEDED"], stats={"DataScannedInBytes": 4096})
    _patch_client(monkeypatch, fake)
    df = pd.DataFrame({"pump_id": ["A", "B"], "series": ["4000", "5000"]})

    class _WR:
        class athena:
            @staticmethod
            def get_query_results(query_execution_id, boto3_session):
                return df

    monkeypatch.setitem(__import__("sys").modules, "awswrangler", _WR)
    # PreprocessedDataAccess().session is only used to build a boto3_session arg here.
    monkeypatch.setattr(
        sql_query, "PreprocessedDataAccess", lambda *a, **k: type("S", (), {"session": None})()
    )
    g = _guarded(f"SELECT pump_id, series FROM {I} LIMIT 5", session)
    res = execute(g)
    assert res.row_count == 2
    assert res.columns == ["pump_id", "series"]
    assert res.data_scanned_bytes == 4096
    assert res.query_execution_id == "qid-123"


def test_execute_failed_query_aborts_with_reason(monkeypatch, session):
    fake = _FakeAthena(states=["FAILED"], reason="INVALID_CAST")
    _patch_client(monkeypatch, fake)
    g = _guarded(f"SELECT pump_id FROM {I} LIMIT 5", session)
    with pytest.raises(SqlExecutionError, match="INVALID_CAST"):
        execute(g)


def test_execute_timeout_cancels_and_aborts(monkeypatch, session):
    fake = _FakeAthena(states=["RUNNING"])  # never terminal
    _patch_client(monkeypatch, fake)
    monkeypatch.setattr(config, "SQL_QUERY_TIMEOUT_SECONDS", 0.0)  # immediate breach
    g = _guarded(f"SELECT pump_id FROM {I} LIMIT 5", session)
    with pytest.raises(SqlExecutionError, match="timeout"):
        execute(g)
    assert fake.stopped == ["qid-123"]  # stop_query_execution was called on breach


# ============ validation is INDEPENDENT of injection (fail-closed) =============
#
# The cross-org gate must not piggyback on the injection traversal: if injection
# silently skips a scope (a future traversal/refactor bug), a standalone post-
# injection validation must still catch the unscoped ref and reject. We prove that by
# stubbing injection and asserting a hard reject — never an unscoped ALLOW.


def test_validation_rejects_when_injection_skipped_entirely(monkeypatch, session):
    # Injection stubbed to a no-op: org/well never get added, yet the guard must NOT
    # emit an unscoped query — the independent validator rejects it.
    monkeypatch.setattr(sql_query, "_inject_scope_predicates", lambda *a, **k: None)
    r = guard(f"SELECT motor_amps FROM {T}", session)
    assert not r.ok, "unscoped query must never be ALLOWED (cross-org leak)"
    assert r.rejection.code == "session_scope_missing"


def test_validation_rejects_missing_date_independently(monkeypatch, session):
    # Injection that pins org/well but SKIPS the date window on a partitioned table.
    # The independent validator must still reject the unbounded partitioned scan.
    def _org_well_only(scoped_refs, org, well, date_supplied, cov_min, cov_max):
        from sqlglot import exp
        for scope, tnode, schema in scoped_refs:
            alias = tnode.alias_or_name
            for pcol in schema.mandatory_predicates:
                scope.expression.where(
                    exp.EQ(
                        this=exp.column(pcol, table=alias),
                        expression=exp.Literal.string({"organization_id": org, "well_id": well}[pcol]),
                    ),
                    append=True,
                    copy=False,
                )

    monkeypatch.setattr(sql_query, "_inject_scope_predicates", _org_well_only)
    r = guard(f"SELECT motor_amps FROM {T}", session)
    assert not r.ok and r.rejection.code == "coverage_missing"


def test_validation_rejects_when_one_join_side_is_skipped(monkeypatch, session):
    # Independence at scope granularity: inject only the FIRST scoped ref, skip the
    # rest. A two-table join must then fail validation on the un-injected side.
    real = sql_query._inject_scope_predicates

    def _only_first(scoped_refs, *a, **k):
        real(scoped_refs[:1], *a, **k)

    monkeypatch.setattr(sql_query, "_inject_scope_predicates", _only_first)
    sql = (
        f"SELECT t.motor_amps, p.alloc_oil_vol FROM {T} t "
        f"JOIN {P} p ON t.observation_day = p.observation_day"
    )
    r = guard(sql, session)
    assert not r.ok and r.rejection.code in ("session_scope_missing", "coverage_missing")


def test_validation_passes_cleanly_under_normal_injection(session):
    # Sanity: with real injection, the independent validator adds no false rejects.
    r = guard(f"SELECT t.motor_amps FROM {T} t JOIN {P} p ON t.observation_day = p.observation_day", session)
    assert r.ok
