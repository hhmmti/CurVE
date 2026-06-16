"""Extension hook contracts for future checkpoint integrations."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class AnalysisExtensionHooks:
    """Optional extension hooks for future data and analysis integrations.

    All hooks are optional and are no-ops unless explicitly provided.
    """

    # Checkpoint 2: optional alternate data-loading path.
    preprocessed_loader: Optional[Callable[[str], Tuple[pd.DataFrame, pd.DataFrame]]] = None

    # Checkpoint 2: optional schema mapping path.
    schema_mapper: Optional[Callable[[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]] = None

    # Checkpoint 3: optional ML recommendation input path.
    ml_recommendation_loader: Optional[Callable[[str], pd.DataFrame]] = None

    # Checkpoint 4: optional ideal curve overlay path.
    ideal_overlay_builder: Optional[Callable[[Dict], Dict]] = None
