"""MarketTemperature v0.4 event-study toolkit.

The package deliberately consumes the frozen v0.3.1 state table.  It does not
change state definitions or make portfolio decisions.
"""

from .events import (
    CONFIRMED_SIGNALS,
    EVENT_TYPES,
    build_event_catalog,
    decluster_events,
)
from .forward_returns import HORIZONS, build_event_outcomes, prepare_benchmark_frame
from .statistics import bootstrap_ci, sample_warning, summarize_outcomes

__all__ = [
    "CONFIRMED_SIGNALS",
    "EVENT_TYPES",
    "HORIZONS",
    "build_event_catalog",
    "decluster_events",
    "HORIZONS",
    "build_event_outcomes",
    "prepare_benchmark_frame",
    "bootstrap_ci",
    "sample_warning",
    "summarize_outcomes",
]
