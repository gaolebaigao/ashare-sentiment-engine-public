"""MarketTemperature v0.4.1 human decision-support advisory layer."""

ADVISORY_VERSION = "0.4.1"

from .confidence import calculate_signal_confidence, confidence_components, module_confirmation, research_evidence_for
from .engine import apply_advisory_layer, build_advisory, build_advisory_frame
from .history import build_advisory_history, load_state_frame, no_future_data_stability
from .models import (
    ADVISORY_REGIMES,
    ADVISORY_SIGNALS,
    MarketAdvisory,
    RESEARCH_EVIDENCE,
)

__all__ = [
    "ADVISORY_REGIMES",
    "ADVISORY_SIGNALS",
    "ADVISORY_VERSION",
    "MarketAdvisory",
    "RESEARCH_EVIDENCE",
    "apply_advisory_layer",
    "build_advisory",
    "build_advisory_frame",
    "build_advisory_history",
    "calculate_signal_confidence",
    "confidence_components",
    "load_state_frame",
    "module_confirmation",
    "no_future_data_stability",
    "research_evidence_for",
]
