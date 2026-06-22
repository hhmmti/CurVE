"""CurVE session record — the front-loaded setup state, re-supplied each turn (M2).

CurVE-decisions §2 Decision 8: setup state (selected well, resolved/proxied inputs,
chosen pump, the availability report) lives in a **dedicated server-side session
record**, separate from the message thread, and is re-supplied to the model every
turn (the engine loads it by id and injects org/well into each tool call + a setup
context line into the prompt). The *within-question* tool turns stay in-memory; this
record is the *across-question* setup state.

The store **body** is swappable behind an unchanged ``load_session`` /
``save_session`` interface (``use_store``). M2-core defaults to an in-memory module
dict; the M2 **surface** points the store at a dict living inside Streamlit's
``st.session_state`` — only the backing mapping changes, never the interface or the
record shape. (DDB is a later milestone.) Callers never reach into the store.

Record shape (CurVE-decisions §2 D8 / §3 D3 backend-injection):
    {
      "session_id":      str,
      "organization_id": str,    # backend-injected into every tool call
      "well_id":         str,    # backend-injected into every tool call
      "resolved_inputs": dict,   # operator-resolved / proxied inputs (depth, SG, …)
      "pump":            None,   # set in M4 (curve_position connection); None in M2
      "availability":    dict,   # the front-loaded per-tool availability report
    }
"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

# Default in-memory store (M2-core). The surface (Streamlit) swaps the *backing
# mapping* via ``use_store`` — e.g. a dict inside ``st.session_state`` — without
# touching the load/save interface or the record shape. DDB is a later milestone.
_DEFAULT_STORE: Dict[str, Dict[str, Any]] = {}
_active_store: MutableMapping[str, Dict[str, Any]] = _DEFAULT_STORE


def use_store(store: MutableMapping[str, Dict[str, Any]]) -> None:
    """Point the session store at a different backing mapping (storage-body swap).

    The mapping only needs dict-like ``__getitem__`` / ``__setitem__`` / ``.get`` /
    ``.clear``. The Streamlit surface passes a dict held in ``st.session_state`` so
    sessions survive script reruns; tests/CLI use the default in-memory dict.
    """
    global _active_store
    _active_store = store


def new_session_record(
    session_id: str,
    organization_id: str,
    well_id: str,
    resolved_inputs: Optional[Dict[str, Any]] = None,
    pump: Optional[Dict[str, Any]] = None,
    availability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a session record dict in the canonical shape (does not store it)."""
    return {
        "session_id": session_id,
        "organization_id": organization_id,
        "well_id": well_id,
        "resolved_inputs": resolved_inputs or {},
        "pump": pump,  # None until M4 resolves a pump connection
        "availability": availability or {},
    }


def save_session(record: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a session record by its ``session_id`` into the active store."""
    if not record.get("session_id"):
        raise ValueError("session record requires a non-empty 'session_id'")
    _active_store[record["session_id"]] = record
    return record


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a session record by id from the active store, or ``None`` if absent."""
    return _active_store.get(session_id)


def clear_sessions() -> None:
    """Drop all sessions in the active store (test/CLI hygiene)."""
    _active_store.clear()
