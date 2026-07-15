"""Pytest root conftest — makes the repo importable without per-file path hacks.

Placing this file at the repo root puts the root on ``sys.path`` during collection,
so tests can ``import curve`` / ``compute`` / ``services`` / ``plotting`` directly.
The explicit insert below is belt-and-suspenders (works regardless of pytest's import
mode) and replaces the ``sys.path.insert(...)`` bootstrap that used to sit atop each
test module.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
