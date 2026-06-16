# M0 — Layer Compatibility Audit

Systemic sweep over the three vendored layers (`compute/`, `services/`, `plotting/`) to
catch **blanket** Streamlit / data coupling before later milestones commit. This is a
*report*, not a remediation — per-tool compatibility is re-confirmed in each tool's DoD
in M2–M4. Checks are from the M0 section of `CurVe-build-plan-v1.md` and the target
tool envelope in `CurVE-decisions.md` §3 (`{values, trust_label + flags, figure_ref}`)
and §4 (gate returns `{available | blocked | proxy, trust_label}`).

**Source vendored from:** `intership-experience/8/real-ideal-analysis/app/{compute,services,plotting}/`
(verbatim copy, `__pycache__` excluded). The `data/` layer was intentionally **not** vendored.

**Headline:** no Streamlit coupling anywhere in the three layers — no `import streamlit`,
no `st.session_state`, no `@st.cache_data`, no `st.plotly_chart`. Compute and Plotting are
CurVE-ready as-is. The only real blockers live in `services/`: three dangling `data/` imports
and a gate whose vocabulary needs mapping to the CurVE envelope.

---

## Compute

Files: `physics_common.py`, `core_calcs.py`, `preprocessed_calcs.py`, `ideal_curve_overlay.py`,
`affinity_validator.py`, `bubble_point_screen.py`, `gas_interference_screen.py`,
`npsh_screen.py`, `energy_efficiency.py`, `ml_recommendation_calcs.py`.

| Check                              | Finding |
|------------------------------------|---------|
| Injected-context args vs. `st.session_state` | **Pass** — functions take explicit args (DataFrames, dicts, floats). Zero `st.session_state` reads. |
| `@st.cache_data` coupling          | **Pass** — none. |
| Data-fetch inside compute          | **Pass** — no `from data ...`, no `boto3`, no I/O. Pure. |
| Return shape fits `{values}`       | **Mostly pass** — functions return plain dicts/scalars/DataFrames (e.g. `compute_affinity_law_validator` returns a nested results dict). Some dicts mix display strings (labels/messages) in with the numbers, so a per-tool `values` projection will be needed in M2–M4 — but nothing is Streamlit-shaped. |

**Blockers:** None. Compute is **CurVE-ready**. (Note for later, not a blocker: per-tool
`values` extraction from the richer result dicts is normal M2–M4 work.)

---

## Services

Files: `data_availability_gate.py`, `pipeline_service.py`, `preprocessed_pipeline_service.py`,
`extension_hooks.py`, `ideal_curve_service.py`, `ml_recommendation_service.py`,
`app_service.py` (legacy Vital, not imported by the active app).

**The gate (`data_availability_gate.py`):**

| Check                                        | Finding |
|----------------------------------------------|---------|
| Decoupled from Page-2 wiring / Streamlit     | **Pass** — `run_data_availability_gate(context, calculation_keys)` takes an injected `context` dict; no Streamlit, no `data/` import. |
| Returns `{available \| blocked \| proxy, trust_label}` | **Partial — mapping needed.** The concepts exist but the vocabulary differs from CurVE §4. Per-field status is `ResolutionStatus = {"Direct", "Manual required", "Fallback allowed", "Proxy allowed", "Blocked"}` (gate `:16`), and a trust label exists as `OutputTrustLabel = {"Validated", "Estimated", "Proxy", "Research prototype"}` carried on each `CalculationContract.output_label` (gate `:24`). But the top-level `run_data_availability_gate` return is **summary-shaped** (`total_calculations_checked`, `ready_calculations`, `blocked_calculations`, `fields_requiring_manual_input`, `fields_using_fallback_or_proxy`, `blocking_fields` — gate `:731`) and does **not** emit a per-tool `{available\|blocked\|proxy, trust_label}` envelope. A thin adapter is required (M2). |
| Reads a data source directly                 | **Pass** — the gate reads only its passed-in `context`; no data source. |

**Other services:**

| Module                              | Finding |
|-------------------------------------|---------|
| `ideal_curve_service.py`            | **Blocker** — `from data.ideal_pump_catalog import load_ideal_catalog` (`:10`) and `import boto3` (`:6`); `load_catalog()` is a data-fetch service (takes a `boto3.Session`, queries Athena). Dangling import + data coupling. |
| `ml_recommendation_service.py`      | **Blocker** — `from data.preprocessed_db import PreprocessedDataDB` (`:14`). Dangling import. |
| `app_service.py`                    | **Blocker (low priority)** — `from data import VitalEnergyDB` (`:7`). Legacy Vital module, **not imported by the active app**; candidate to drop rather than reconcile. |
| `pipeline_service.py`               | **Pass** — imports only `compute.core_calcs`, `plotting.curves`, `services.extension_hooks`. No `data/`, no Streamlit. |
| `preprocessed_pipeline_service.py`  | **Pass** — imports only `compute.preprocessed_calcs` + pandas/numpy. No `data/`, no Streamlit. |
| `extension_hooks.py`                | **Pass** — dataclasses/typing/pandas only. |

**Blockers:**
1. `services/ideal_curve_service.py` — `from data.ideal_pump_catalog import load_ideal_catalog` (`:10`), `import boto3` (`:6`); data-fetch service. (Catalog read path is M4 territory.)
2. `services/ml_recommendation_service.py` — `from data.preprocessed_db import PreprocessedDataDB` (`:14`).
3. `services/app_service.py` — `from data import VitalEnergyDB` (`:7`); legacy/unused, likely droppable.
4. Gate vocabulary/return-shape mapping to CurVE §4 `{available | blocked | proxy, trust_label}` (gate `:16`, `:24`, `:731`) — adapter, not a rewrite.

All four are **report-only** here; reconciliation is M2 (data-path injection) and M4 (catalog path).

---

## Plotting

Files: `affinity_charts.py`, `bep_charts.py`, `bubble_npsh_charts.py`, `curves.py`,
`energy_charts.py`, `ml_recommendation_charts.py`, `preprocessed_charts.py`.

| Check                                   | Finding |
|-----------------------------------------|---------|
| Returns a Plotly figure **object** vs. inline `st.plotly_chart` | **Pass** — builders are typed `-> go.Figure` and `return fig` (e.g. `affinity_charts.build_affinity_law_panel`, `bubble_npsh_charts.build_bubble_point_strip`). No `st.plotly_chart` anywhere. |
| Self-contained for `figure_ref`         | **Pass** — figures are constructed from passed-in dicts/DataFrames and handed back; a caller can hold the object as a `figure_ref` and render UI-side. |
| Streamlit in function body              | **Pass** — none. (The only `st.` hits in the sweep were a docstring mention in `services/ideal_curve_service.py` and `trust.lower()` substring matches in `bubble_npsh_charts.py` — not Streamlit calls.) |

**Blockers:** None. Plotting is **CurVE-ready** — returns figure objects, suitable for the
`figure_ref` envelope (figures render to UI, never back into the model).

---

## Dangling data-layer imports

The `data/` layer was deliberately not vendored (deferred to M2). As expected, that leaves
dangling imports — **all in `services/`, none in `compute/` or `plotting/`**:

| File | Line | Import |
|------|------|--------|
| `services/ideal_curve_service.py`     | `:10` | `from data.ideal_pump_catalog import load_ideal_catalog` |
| `services/ml_recommendation_service.py` | `:14` | `from data.preprocessed_db import PreprocessedDataDB` |
| `services/app_service.py`             | `:7`  | `from data import VitalEnergyDB` (legacy Vital; unused by active app) |

These are **not** fixed in M0. They are the natural seam for the M2 backend-injection /
data-layer reconciliation (and M4 catalog read path); `app_service.py` is a drop candidate.

---

## Summary

| Layer      | Verdict |
|------------|---------|
| Compute    | **Clean / CurVE-ready** — pure, no Streamlit, no I/O. |
| Services   | **Blocked** — 3 dangling `data/` imports + gate envelope mapping (all report-only, M2/M4). |
| Plotting   | **Clean / CurVE-ready** — returns `go.Figure` objects, no Streamlit. |
