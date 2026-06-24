"""CurVE shared tool envelope — one canonical shape for success AND failure (M3).

THE CONTRACT (CurVE-decisions §3 D2/D3):
  Every real tool returns ONE dict shape, regardless of outcome::

      {status, values, trust_label, flags, figure_ref, figure}

  * ``status``      — ``available`` (success) | ``blocked`` | ``error``.
  * ``values``      — the model-facing narration payload on success; on failure it
                      carries identity only (``{"well_id": …}`` or ``{}``) — **never a
                      fabricated number** (Validated = measured-or-missing).
  * ``trust_label`` — carried faithfully from the gate on success; ``None`` on failure.
  * ``flags``       — ``[]`` on a clean success; the reason(s) on failure.
  * ``figure_ref`` / ``figure`` — the UI artifacts on success; ``None`` on failure.
    The engine strips both before the result reaches the model (no image tokens).

WHY A SHARED HELPER (the M3 contract that the rest of M3 builds on):
  Before M3 each tool hand-rolled its own dict literals for the blocked/missing
  cases, so the success and failure shapes drifted per tool. M3 makes the failure
  path a **structured envelope with the same keys as success**, defined ONCE here and
  reused by every tool. A data/compute failure returns ``error_envelope(...)`` — a
  well-formed envelope with the reason in ``flags`` — never a raw exception bubbling
  to the engine. ``production_history`` and ``water_cut_gor_history`` both build their
  envelopes through these two helpers, so the shapes are structurally identical.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# The canonical key set every tool envelope carries — success and failure alike.
# Tests assert ``set(env) == ENVELOPE_KEYS`` to lock shape parity across tools.
ENVELOPE_KEYS = {"status", "values", "trust_label", "flags", "figure_ref", "figure"}


def success_envelope(
    *,
    values: Dict[str, Any],
    trust_label: Optional[str],
    flags: List[str],
    figure_ref: str,
    figure: Any,
    status: str = "available",
) -> Dict[str, Any]:
    """Build a success envelope. ``trust_label`` is carried from the gate, never hardcoded."""
    return {
        "status": status,
        "values": values,
        "trust_label": trust_label,
        "flags": list(flags),
        "figure_ref": figure_ref,
        "figure": figure,
    }


def error_envelope(
    status: str,
    flags: List[str],
    *,
    well_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a failure envelope with the SAME shape as :func:`success_envelope`.

    No physics numbers are fabricated: ``values`` carries identity only (the
    ``well_id`` for UI context when known, else ``{}``), ``trust_label`` is ``None``,
    and the figure artifacts are ``None``. ``status`` is ``blocked`` (gate/presence)
    or ``error`` (data/compute failure); ``flags`` carries the reason(s).
    """
    return {
        "status": status,
        "values": {"well_id": well_id} if well_id else {},
        "trust_label": None,
        "flags": list(flags),
        "figure_ref": None,
        "figure": None,
    }
