"""Market-temperature regime and turning-point research engine."""

from .models import (
    EpisodeAnchorConfig,
    STATE_PRIORITY,
    SIGNALS,
    STATES,
    StateMachineConfig,
)
from .episode_anchors import add_episode_anchor_flags
from .smoothing import add_smoothing_metrics, causal_valid_ema, valid_lag, valid_rolling_extreme
from .state_machine import apply_state_machine, transition_matrix
from .turning_points import build_regime_indicators

__all__ = [
    "STATE_PRIORITY",
    "SIGNALS",
    "STATES",
    "StateMachineConfig",
    "EpisodeAnchorConfig",
    "add_episode_anchor_flags",
    "add_smoothing_metrics",
    "apply_state_machine",
    "build_regime_indicators",
    "causal_valid_ema",
    "transition_matrix",
    "valid_lag",
    "valid_rolling_extreme",
]
