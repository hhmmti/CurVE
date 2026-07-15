"""CurVE ``sql_query`` data plane (M2) — the guard + execution choke point.

This is the single, **no-LLM** boundary every generated query passes through before
it can touch Athena. Three primitives at one choke point:

    guard(sql, session)   -> GuardResult   (accepted GuardedSQL | structured Rejection)
    explain(guarded)      -> {"ok", "error"}   (zero-scan Athena EXPLAIN pre-validate)
    execute(guarded)      -> ExecuteResult      (full DataFrame + columns + row_count)

It is a **security boundary**: the disposition on any doubt is **reject**. Fail-closed
everywhere — a parse error, an ambiguity, an unrecognized construct, or any exception
is a rejection, never a silent pass and never an unbounded query.

What the guard enforces (locked decisions C8 · C9 · B6):
  1. Parse with ``sqlglot`` (Trino/Athena dialect). Any parse failure -> reject.
  2. Exactly ONE statement, and it must be a pure read (SELECT / WITH…SELECT / UNION
     of SELECTs). Stacked statements, DDL/DML, CTAS, SHOW/DESCRIBE/EXPLAIN -> reject.
  3. Whole-tree allowlist: every base table referenced anywhere (top-level, JOIN,
     subquery, CTE body, UNION side) must be a fully-qualified ``schema_catalog``
     table. CTE aliases are not base tables. Any stray reference -> reject.
  4. Table-aware scoping INJECTION: the session's org/well predicate the catalog's
     ``mandatory_predicates`` map requires is injected into every scope that
     references a scoped table (org+well / well-only / none). Values come from the
     session, never the model.
  5. Reject-on-presence: if the model emitted ANY org/well predicate on a scoped
     table, reject (protocol violation — the guard owns scoping; no value compare).
  6. Date partition (cost lever, the two ``esp_optimization_v2`` tables only): if no
     ``observation_day`` predicate is present, inject the session coverage window;
     if the model supplied one, respect it (widening allowed, no override).
  7. LIMIT: absent -> inject the configured cap; present and <= cap -> respect;
     present and > cap -> REJECT (never clamp).

The execution plane reuses ``curve.data``'s Athena credential path (same profile /
region / results bucket / catalog / database pins from ``curve.config``) — it does
NOT build a new client. ``execute`` enforces a client-side timeout with
``stop_query_execution`` on breach. The choke point is structural: ``explain`` and
``execute`` accept only a :class:`GuardedSQL` minted by ``guard`` — a hand-built one
without the private token is refused, so un-guarded SQL can never reach Athena.

Kickoff-verified live (2026-07-14, profile ``roam-ai``):
  * EXPLAIN is zero-scan over BOTH a Glue table and the FEDERATED recs table
    (``roam_prd_ddb`` DynamoDB connector) — DataScannedInBytes=0, no Lambda
    misbehavior — so ``explain`` runs over every allowlist table including recs.
  * Workgroup ``primary`` has ``EnforceWorkGroupConfiguration=False`` and no
    ``BytesScannedCutoffPerQuery``; the shared workgroup is not mutated here, so the
    scan-cost lever is the forced LIMIT + partition filter + client-side timeout
    (the documented C9 fallback), not a workgroup byte cutoff.
  * Session accessors: ``organization_id`` / ``well_id`` are top-level keys; the
    coverage window is ``session['availability']['coverage']['min_day'|'max_day']``
    (same accessor ``curve.prompt.format_setup_context`` reads).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from curve import config
from curve.data import PreprocessedDataAccess
from curve.schema_catalog import SCHEMA_TABLES

# Trino is Athena's engine-v3 dialect in sqlglot; the guard parses in it so Athena
# constructs (UNNEST, struct dot-access, quoted spaced identifiers) round-trip.
_DIALECT = "trino"

# The scoping columns the guard OWNS. A model-supplied predicate on either (on any
# scoped table) is a protocol violation -> reject-on-presence (requirement 5). These
# are the only column names carrying org/well confidentiality across the catalog.
_ORG_COL = "organization_id"
_WELL_COL = "well_id"
_SCOPING_COLS = frozenset({_ORG_COL, _WELL_COL})

# The date-partition column (the cost lever; requirement 6). Present only on the two
# partitioned esp_optimization_v2 tables.
_DATE_PART_COL = "observation_day"

# AST nodes that put a column in a *filtering* position (WHERE / HAVING / QUALIFY /
# JOIN ON). A scoping column appearing under any of these is a model-supplied
# predicate; a projection / ORDER BY / GROUP BY reference is not.
_FILTER_ANCESTORS = (exp.Where, exp.Having, exp.Qualify, exp.Join)

# Read-only allow-set for the single top-level statement. Everything else — DML, DDL,
# CTAS (exp.Create), SHOW/GRANT/USE/etc (exp.Command), DESCRIBE (exp.Describe), an
# input EXPLAIN (exp.Command) — is rejected.
_ALLOWED_TOP = (exp.Select, exp.Union, exp.Subquery)


# --- allowlist derived once from schema_catalog (never hardcoded here) ---------

# fqn (lowercased) -> TableSchema, so a base-table reference resolves to its
# mandatory_predicates + partition_columns. Athena identifiers are case-insensitive,
# so we match case-folded.
_ALLOWLIST: Dict[str, Any] = {t.fqn.lower(): t for t in SCHEMA_TABLES}


# --- result / rejection types --------------------------------------------------


@dataclass(frozen=True)
class Rejection:
    """A structured, machine-readable reason the guard refused a query."""

    code: str
    reason: str

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "reason": self.reason}


# A private token proving a GuardedSQL was minted by ``guard`` and nowhere else. This
# is what makes the choke point structural: ``explain`` / ``execute`` verify it, so a
# hand-constructed GuardedSQL(sql=...) cannot smuggle un-guarded SQL to Athena.
_GUARD_TOKEN = object()


@dataclass(frozen=True)
class GuardedSQL:
    """SQL that has passed the guard: the injected, capped, allowlist-only statement.

    Only ``guard`` sets a valid ``_token``. ``explain`` / ``execute`` refuse any
    instance whose token is not the module-private sentinel.
    """

    sql: str
    tables: Tuple[str, ...]  # fully-qualified allowlist tables referenced
    _token: Any = None


@dataclass(frozen=True)
class GuardResult:
    """Outcome of ``guard``. Exactly one of ``guarded`` / ``rejection`` is set."""

    ok: bool
    guarded: Optional[GuardedSQL] = None
    rejection: Optional[Rejection] = None


@dataclass(frozen=True)
class ExecuteResult:
    """Full result of a guarded query — M3/M4 shape the trimmed payload from this."""

    dataframe: Any  # pandas.DataFrame (typed loosely to avoid an import-time dep)
    columns: List[str]
    row_count: int
    query_execution_id: str
    data_scanned_bytes: int = 0


class SqlExecutionError(RuntimeError):
    """Raised by ``execute`` on any abort (timeout, query failure, choke violation).

    Carries a plain reason string so the caller (M3 self-correct) can feed it back.
    """


# --- small helpers -------------------------------------------------------------


def _reject(code: str, reason: str) -> GuardResult:
    return GuardResult(ok=False, rejection=Rejection(code=code, reason=reason))


def _table_fqn(node: exp.Table) -> str:
    """``catalog.db.name`` for a Table node, or a partial string if under-qualified."""
    parts = [node.catalog, node.db, node.name]
    return ".".join(p for p in parts if p)


def _is_base_table(scope, node: exp.Table) -> bool:
    """True if this Table node is a real base table (not a CTE / derived-table ref).

    sqlglot resolves a CTE / derived reference to a nested ``Scope`` in
    ``scope.sources``; a physical base table resolves to the Table node itself (or is
    absent). We treat only non-Scope sources as base tables.
    """
    src = scope.sources.get(node.alias_or_name)
    return not _is_scope(src)


def _is_scope(src: Any) -> bool:
    # Import-free duck check: a sqlglot Scope has a ``.expression`` and ``.sources``.
    return hasattr(src, "expression") and hasattr(src, "sources") and not isinstance(
        src, exp.Expression
    )


def _in_filter_position(col: exp.Column) -> bool:
    """True if ``col`` sits inside a WHERE / HAVING / QUALIFY / JOIN-ON subtree."""
    node = col.parent
    while node is not None:
        if isinstance(node, _FILTER_ANCESTORS):
            return True
        node = node.parent
    return False


# --- scoping: inject then independently validate -------------------------------
#
# These two passes are deliberately separate. ``_inject_scope_predicates`` writes the
# org/well (+ date) predicates into each scope; ``_validate_injected_scoping`` then
# re-walks every scoped ref and proves the predicate is actually present, WITHOUT
# trusting that injection ran or reached that scope. The cross-org gate therefore
# fails closed on an injection/traversal defect instead of emitting an unscoped query.
# Both read the SAME schema-derived map (``schema.mandatory_predicates`` /
# ``partition_columns``) — there is no second, divergent scoping definition.


def _inject_scope_predicates(scoped_refs, org: str, well: str, date_supplied: bool,
                             cov_min, cov_max) -> None:
    """AND each scoped table's required org/well (+ date) predicate into its scope."""
    inject_values = {_ORG_COL: org, _WELL_COL: well}
    for scope, tnode, schema in scoped_refs:
        alias = tnode.alias_or_name
        preds: List[exp.Expression] = []
        for pcol in schema.mandatory_predicates:  # org+well / well-only / [] per table
            preds.append(
                exp.EQ(
                    this=exp.column(pcol, table=alias),
                    expression=exp.Literal.string(inject_values[pcol]),
                )
            )
        if (
            _DATE_PART_COL in schema.partition_columns
            and not date_supplied
            and cov_min
            and cov_max
        ):
            col = exp.column(_DATE_PART_COL, table=alias)
            preds.append(exp.GTE(this=col, expression=exp.Literal.string(str(cov_min))))
            preds.append(
                exp.LTE(
                    this=exp.column(_DATE_PART_COL, table=alias),
                    expression=exp.Literal.string(str(cov_max)),
                )
            )
        if not preds:
            continue  # unscoped reference table (ideal_pump_library_v1) — nothing to add
        condition = preds[0]
        for extra in preds[1:]:
            condition = exp.and_(condition, extra)
        # scope.expression is the Select owning this table ref; AND into its WHERE.
        scope.expression.where(condition, append=True, copy=False)


def _scope_has_alias_predicate(scope, alias: str, colname: str) -> bool:
    """True if ``alias.colname`` appears in a filter position of ``scope``'s own WHERE.

    Confined to THIS scope's WHERE (not descendant subquery scopes, which are their own
    entries in ``scoped_refs``) and required to be alias-qualified, so a predicate on a
    sibling table's identically-named column cannot vouch for this one.
    """
    where = scope.expression.args.get("where")
    if where is None:
        return False
    for col in where.find_all(exp.Column):
        if (col.name or "").lower() == colname and (col.table or "") == alias and (
            _in_filter_position(col)
        ):
            return True
    return False


def _validate_injected_scoping(scoped_refs, date_supplied: bool) -> Optional[Rejection]:
    """Independently confirm every scoped ref carries its mandated predicate.

    Runs AFTER injection but does not assume injection succeeded: it re-derives what
    each ref requires from the schema map and checks the emitted tree. A missing
    org/well predicate -> ``session_scope_missing``; a missing date window on a
    partitioned table (when the session didn't supply one) -> ``coverage_missing``.
    """
    for scope, tnode, schema in scoped_refs:
        alias = tnode.alias_or_name
        for pcol in schema.mandatory_predicates:  # org / well per this table
            if not _scope_has_alias_predicate(scope, alias, pcol):
                return Rejection(
                    "session_scope_missing",
                    f"scoped table '{schema.fqn}' reached validation without its "
                    f"required '{pcol}' predicate — refusing (cross-org gate, "
                    "fail-closed on an injection defect).",
                )
        if (
            _DATE_PART_COL in schema.partition_columns
            and not date_supplied
            and not _scope_has_alias_predicate(scope, alias, _DATE_PART_COL)
        ):
            return Rejection(
                "coverage_missing",
                f"partitioned table '{schema.fqn}' reached validation without an "
                "observation_day window — refusing an unbounded scan (fail-closed).",
            )
    return None


# --- the guard -----------------------------------------------------------------


def guard(sql: str, session: Dict[str, Any]) -> GuardResult:
    """Validate + rewrite ``sql`` into a safe, scoped, capped :class:`GuardedSQL`.

    Order (fail-closed): parse -> single-statement -> read-only -> whole-tree
    allowlist -> reject-on-presence (org/well) -> LIMIT policy -> inject scoping +
    date window -> emit. Any failure at any step returns a structured Rejection.

    ``session`` supplies ``organization_id`` / ``well_id`` and the coverage window
    (``availability.coverage.min_day|max_day``). Values are read from the session
    only — never from the model's SQL.
    """
    # 0. session values (trusted; the only source of org/well/date defaults).
    org = (session or {}).get("organization_id")
    well = (session or {}).get("well_id")
    if not org or not well:
        return _reject(
            "session_scope_missing",
            "session lacks organization_id/well_id — cannot scope a query safely.",
        )
    coverage = ((session or {}).get("availability") or {}).get("coverage") or {}
    cov_min = coverage.get("min_day")
    cov_max = coverage.get("max_day")

    # 1. parse (Trino). ANY parse error / empty input -> reject.
    if not sql or not sql.strip():
        return _reject("empty", "empty SQL.")
    try:
        statements = sqlglot.parse(sql, dialect=_DIALECT)
    except Exception as e:  # sqlglot.errors.ParseError and anything else — fail closed
        return _reject("parse_error", f"unparseable SQL: {type(e).__name__}: {e}")
    statements = [s for s in statements if s is not None]

    # 2a. exactly one statement (reject stacked / multi-statement).
    if len(statements) != 1:
        return _reject(
            "multi_statement",
            f"exactly one statement required; got {len(statements)} (stacked SQL is rejected).",
        )
    stmt = statements[0]

    # 2b. top-level must be a pure read; walk the tree for any DML/DDL/command node.
    if not isinstance(stmt, _ALLOWED_TOP):
        return _reject(
            "not_read_only",
            f"only SELECT / WITH…SELECT / UNION reads are allowed; got {type(stmt).__name__}.",
        )
    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop,
        exp.Alter, exp.Command, exp.Describe, exp.Set, exp.Use, exp.TruncateTable,
    )
    bad = stmt.find(*forbidden)
    if bad is not None:
        return _reject(
            "not_read_only",
            f"non-read construct present ({type(bad).__name__}); only pure SELECT reads are allowed.",
        )

    # 3. whole-tree allowlist + collect the scoped/partitioned tables per scope.
    try:
        scopes = traverse_scope(stmt)
    except Exception as e:  # malformed tree the parser accepted — fail closed
        return _reject("parse_error", f"could not resolve query scopes: {type(e).__name__}: {e}")

    referenced_fqns: List[str] = []
    # (scope, Table node, TableSchema) triples for every base allowlist table ref.
    scoped_refs: List[Tuple[Any, exp.Table, Any]] = []
    for scope in scopes:
        for tnode in scope.tables:
            if not _is_base_table(scope, tnode):
                continue  # CTE alias / derived table — not a base table
            fqn = _table_fqn(tnode)
            schema = _ALLOWLIST.get(fqn.lower())
            if schema is None:
                return _reject(
                    "not_allowlisted",
                    f"table '{fqn or tnode.name}' is not in the allowlist "
                    "(must be one of the five fully-qualified schema_catalog tables).",
                )
            referenced_fqns.append(schema.fqn)
            scoped_refs.append((scope, tnode, schema))

    if not scoped_refs:
        return _reject(
            "no_base_table",
            "no allowlist base table referenced — nothing to query.",
        )

    # 4. reject-on-presence: any org/well column in a filter position -> reject.
    #    Any observation_day filter -> mark date_supplied (respect, do not inject).
    date_supplied = False
    for col in stmt.find_all(exp.Column):
        name = (col.name or "").lower()
        if name in _SCOPING_COLS and _in_filter_position(col):
            return _reject(
                "model_supplied_scope",
                f"model emitted a '{name}' predicate — the guard owns scoping; "
                "org/well filters must not be model-supplied.",
            )
        if name == _DATE_PART_COL and _in_filter_position(col):
            date_supplied = True

    # 5. LIMIT policy on the OUTERMOST statement (forced cap; reject if over).
    cap = int(config.SQL_QUERY_LIMIT_CAP)
    limit_node = stmt.args.get("limit")
    if limit_node is not None:
        lit = limit_node.expression
        if not isinstance(lit, exp.Literal) or not lit.is_int:
            return _reject(
                "limit_unverifiable",
                "LIMIT is not an integer literal; cannot verify it against the cap.",
            )
        n = int(lit.name)
        if n > cap:
            return _reject(
                "limit_over_cap",
                f"LIMIT {n} exceeds the cap of {cap} (rejected — the guard never clamps).",
            )
        # <= cap: respect as-is.
    else:
        stmt.set("limit", exp.Limit(expression=exp.Literal.number(cap)))

    # 6. inject scoping (+ date window) per scope. A partitioned table needs a date
    #    window; if none was supplied and the session has no coverage range, fail
    #    closed rather than emit an unbounded partitioned scan.
    needs_date = any(
        _DATE_PART_COL in schema.partition_columns for _, _, schema in scoped_refs
    )
    if needs_date and not date_supplied and not (cov_min and cov_max):
        return _reject(
            "coverage_missing",
            "partitioned table needs an observation_day window but the session has no "
            "coverage range and the model supplied none — refusing an unbounded scan.",
        )

    # Inject the required predicates into every scoped-table scope, then INDEPENDENTLY
    # re-validate that each scoped ref actually carries them. The validation pass does
    # NOT trust that injection ran or reached every scope — a traversal/injection bug
    # that skips a scope is caught here and rejected, never emitted. This is the
    # load-bearing cross-org gate: it fails closed on an injection defect.
    _inject_scope_predicates(scoped_refs, str(org), str(well), date_supplied, cov_min, cov_max)
    scope_gap = _validate_injected_scoping(scoped_refs, date_supplied)
    if scope_gap is not None:
        return GuardResult(ok=False, rejection=scope_gap)

    guarded_sql = stmt.sql(dialect=_DIALECT)
    # De-dupe referenced tables preserving order (a table may appear in >1 scope).
    seen: Dict[str, None] = {}
    for f in referenced_fqns:
        seen.setdefault(f, None)
    guarded = GuardedSQL(sql=guarded_sql, tables=tuple(seen), _token=_GUARD_TOKEN)
    return GuardResult(ok=True, guarded=guarded)


# --- execution plane (reuses curve.data's Athena credential path) --------------


def _athena_client():
    """boto3 Athena client built from ``curve.data``'s credential path.

    Reuses ``PreprocessedDataAccess`` session-building (explicit profile -> AWS_PROFILE
    env -> default chain; region from config) so this does not re-plumb a new client.
    """
    return PreprocessedDataAccess().session.client("athena")


def _require_guarded(obj: Any) -> str:
    """Enforce the choke point: only a guard-minted GuardedSQL is executable."""
    if not isinstance(obj, GuardedSQL) or obj._token is not _GUARD_TOKEN:
        raise SqlExecutionError(
            "choke-point violation: explain/execute require guard() output, not raw SQL."
        )
    return obj.sql


def _start(client, query: str) -> str:
    return client.start_query_execution(
        QueryString=query,
        # fqn in the SQL overrides these defaults; AwsDataCatalog is the default
        # catalog and federated fqns (roam_prd_ddb.*) resolve regardless (verified).
        QueryExecutionContext={
            "Catalog": "AwsDataCatalog",
            "Database": config.PREPROCESSED_DATABASE,
        },
        ResultConfiguration={"OutputLocation": config.ATHENA_S3_OUTPUT},
    )["QueryExecutionId"]


def _poll(client, qid: str, timeout_s: float) -> Dict[str, Any]:
    """Poll a query to a terminal state; cancel + abort on timeout (fail-closed)."""
    deadline = time.monotonic() + timeout_s
    delay = 0.2
    while True:
        qe = client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = qe["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return qe
        if time.monotonic() >= deadline:
            try:
                client.stop_query_execution(QueryExecutionId=qid)
            except Exception:
                pass  # best-effort cancel; we abort regardless
            raise SqlExecutionError(
                f"query {qid} exceeded the {timeout_s:.0f}s timeout and was cancelled."
            )
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        delay = min(delay * 1.5, 2.0)


def explain(guarded: GuardedSQL) -> Dict[str, Any]:
    """Run Athena ``EXPLAIN`` on the GUARDED SQL — a zero-scan pre-validate (B6).

    Returns ``{"ok": bool, "error": Optional[str]}``. The error string is the Athena
    ``StateChangeReason`` (parseable, so M3's self-correct loop can feed it back).
    Verified zero-scan over every allowlist table, including the federated recs table.
    """
    sql = _require_guarded(guarded)
    try:
        client = _athena_client()
        qid = _start(client, "EXPLAIN " + sql)
        qe = _poll(client, qid, config.SQL_QUERY_TIMEOUT_SECONDS)
    except SqlExecutionError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # AWS / boto error — fail closed with a reason
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if qe["Status"]["State"] == "SUCCEEDED":
        return {"ok": True, "error": None}
    return {"ok": False, "error": qe["Status"].get("StateChangeReason", "EXPLAIN failed")}


def execute(guarded: GuardedSQL) -> ExecuteResult:
    """Execute the GUARDED SQL on Athena and return the FULL result.

    Enforces a client-side timeout (``stop_query_execution`` on breach). Returns an
    :class:`ExecuteResult` (DataFrame + columns + row_count + metadata) — not the
    trimmed B7 payload; M3/M4 shape that. Any failure aborts with a reason
    (``SqlExecutionError``) — never a silent pass, never an unbounded wait.
    """
    sql = _require_guarded(guarded)
    client = _athena_client()
    qid = _start(client, sql)
    qe = _poll(client, qid, config.SQL_QUERY_TIMEOUT_SECONDS)
    state = qe["Status"]["State"]
    if state != "SUCCEEDED":
        reason = qe["Status"].get("StateChangeReason", state)
        raise SqlExecutionError(f"query {qid} {state}: {reason}")

    # Read the already-computed result set by execution id (no re-run). Lazy import so
    # importing this module (and cred-free guard tests) needs no awswrangler.
    import awswrangler as wr

    df = wr.athena.get_query_results(
        query_execution_id=qid,
        boto3_session=PreprocessedDataAccess().session,
    )
    scanned = int(qe.get("Statistics", {}).get("DataScannedInBytes", 0) or 0)
    return ExecuteResult(
        dataframe=df,
        columns=list(df.columns),
        row_count=int(len(df)),
        query_execution_id=qid,
        data_scanned_bytes=scanned,
    )
