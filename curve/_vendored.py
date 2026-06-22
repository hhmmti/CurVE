"""Load the two CLEAN vendored service modules without running services/__init__.py.

Why this exists
---------------
The vendored ``services/__init__.py`` eagerly imports ``ml_recommendation_service``
and ``pipeline_service``, which carry dangling ``data/`` imports
(``from data.preprocessed_db import …``). Per the M0 audit those modules are
**import-broken until the data layer lands** in later milestones — the ``data/``
layer is the deliberate M2+ seam and was intentionally not vendored. So *any*
``from services.<x> import …`` triggers the package ``__init__`` and explodes, even
when ``<x>`` itself is clean.

M2 core needs only two service modules, both of which import nothing but
``compute`` / numpy / pandas:
  * ``preprocessed_pipeline_service`` — the join + engineered-dataframe pipeline
  * ``data_availability_gate``        — the gate the per-tool gate adapts

We load them **by file path under synthetic module names**, so the package
``__init__`` (with its broken siblings) never runs. Their own absolute imports
(``from compute import preprocessed_calcs``) resolve normally against ``sys.path``.

This respects the M2 constraint "do not modify the vendored ``services/`` source —
put adapters in ``curve/``": nothing under ``services/`` is touched; this adapter
lives in ``curve/``. When the data layer is built out (M3+), importing the package
normally becomes possible and this shim can retire.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

_SERVICES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "services"
)


def _load_service_module(filename: str, synthetic_name: str) -> ModuleType:
    """Exec a single ``services/*.py`` file under a synthetic top-level name."""
    if synthetic_name in sys.modules:
        return sys.modules[synthetic_name]
    path = os.path.join(_SERVICES_DIR, filename)
    spec = importlib.util.spec_from_file_location(synthetic_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load vendored service module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so the module is resolvable if it self-references.
    sys.modules[synthetic_name] = module
    spec.loader.exec_module(module)
    return module


preprocessed_pipeline_service = _load_service_module(
    "preprocessed_pipeline_service.py", "_curve_vendored_preprocessed_pipeline_service"
)
data_availability_gate = _load_service_module(
    "data_availability_gate.py", "_curve_vendored_data_availability_gate"
)
