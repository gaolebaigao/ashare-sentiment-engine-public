"""Configuration and stable vocabulary for MarketTemperature v0.3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


STATES = (
    "DATA_INVALID",
    "EXTREME_PANIC",
    "PANIC_FALLING",
    "ICE_REVERSAL_WATCH",
    "ICE_REVERSAL",
    "COLD",
    "NORMAL",
    "HOT",
    "EUPHORIA_RISING",
    "HOT_ROLLOVER_WATCH",
    "HOT_ROLLOVER",
)

# The order is part of the research contract. A turning-point candidate wins
# over a descriptive temperature label on the same valid observation.
STATE_PRIORITY = (
    "DATA_INVALID",
    "ICE_REVERSAL",
    "HOT_ROLLOVER",
    "ICE_REVERSAL_WATCH",
    "HOT_ROLLOVER_WATCH",
    "PANIC_FALLING",
    "EUPHORIA_RISING",
    "EXTREME_PANIC",
    "COLD",
    "HOT",
    "NORMAL",
)

SIGNALS = (
    "NONE",
    "ICE_REVERSAL_WATCH",
    "ICE_REVERSAL_CONFIRMED",
    "HOT_ROLLOVER_WATCH",
    "HOT_ROLLOVER_CONFIRMED",
)


@dataclass(frozen=True)
class StateMachineConfig:
    """Research defaults for the deterministic v0.3.1 state machine."""

    ema_span: int = 3
    cold_threshold: float = 20.0
    cool_threshold: float = 30.0
    warm_threshold: float = 70.0
    hot_threshold: float = 80.0
    turning_window: int = 10
    turning_delta: float = 5.0
    slope_lookback: int = 3
    slope3_threshold: float = 4.0
    minimum_confirming_modules: int = 3
    confirmation_days: int = 2
    watch_timeout: int = 5
    zone_hysteresis: float = 3.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "StateMachineConfig":
        values = dict((config or {}).get("state_machine", config or {}))
        defaults = cls()
        result = cls(
            ema_span=int(values.get("ema_span", defaults.ema_span)),
            cold_threshold=float(values.get("cold_threshold", defaults.cold_threshold)),
            cool_threshold=float(values.get("cool_threshold", defaults.cool_threshold)),
            warm_threshold=float(values.get("warm_threshold", defaults.warm_threshold)),
            hot_threshold=float(values.get("hot_threshold", defaults.hot_threshold)),
            turning_window=int(values.get("turning_window", defaults.turning_window)),
            turning_delta=float(values.get("turning_delta", defaults.turning_delta)),
            slope_lookback=int(values.get("slope_lookback", defaults.slope_lookback)),
            slope3_threshold=float(values.get("slope3_threshold", defaults.slope3_threshold)),
            minimum_confirming_modules=int(values.get("minimum_confirming_modules", defaults.minimum_confirming_modules)),
            confirmation_days=int(values.get("confirmation_days", defaults.confirmation_days)),
            watch_timeout=int(values.get("watch_timeout", defaults.watch_timeout)),
            zone_hysteresis=float(values.get("zone_hysteresis", defaults.zone_hysteresis)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.ema_span < 1:
            raise ValueError("state_machine.ema_span must be >= 1")
        if self.turning_window < 1 or self.slope_lookback < 1:
            raise ValueError("state_machine windows must be >= 1")
        if self.confirmation_days < 1 or self.watch_timeout < 1:
            raise ValueError("confirmation_days and watch_timeout must be >= 1")
        if self.minimum_confirming_modules < 1 or self.minimum_confirming_modules > 4:
            raise ValueError("minimum_confirming_modules must be between 1 and 4")
        if not self.cold_threshold < self.cool_threshold < self.warm_threshold < self.hot_threshold:
            raise ValueError("state_machine thresholds must be cold < cool < warm < hot")
        if self.turning_delta < 0 or self.slope3_threshold < 0:
            raise ValueError("turning_delta and slope3_threshold must be non-negative")
        if self.zone_hysteresis < 0:
            raise ValueError("zone_hysteresis must be non-negative")


@dataclass(frozen=True)
class EpisodeAnchorConfig:
    """Causal extreme-event anchor and memory defaults for v0.3.1."""

    raw_panic_threshold: float = 20.0
    smoothed_panic_threshold: float = 20.0
    module_panic_threshold: float = 20.0
    raw_euphoria_threshold: float = 85.0
    smoothed_euphoria_threshold: float = 80.0
    module_euphoria_threshold: float = 80.0
    minimum_extreme_modules: int = 2
    memory_window: int = 10

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "EpisodeAnchorConfig":
        values = dict((config or {}).get("episode_anchor", config or {}))
        defaults = cls()
        result = cls(
            raw_panic_threshold=float(values.get("raw_panic_threshold", defaults.raw_panic_threshold)),
            smoothed_panic_threshold=float(values.get("smoothed_panic_threshold", defaults.smoothed_panic_threshold)),
            module_panic_threshold=float(values.get("module_panic_threshold", defaults.module_panic_threshold)),
            raw_euphoria_threshold=float(values.get("raw_euphoria_threshold", defaults.raw_euphoria_threshold)),
            smoothed_euphoria_threshold=float(values.get("smoothed_euphoria_threshold", defaults.smoothed_euphoria_threshold)),
            module_euphoria_threshold=float(values.get("module_euphoria_threshold", defaults.module_euphoria_threshold)),
            minimum_extreme_modules=int(values.get("minimum_extreme_modules", defaults.minimum_extreme_modules)),
            memory_window=int(values.get("memory_window", defaults.memory_window)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.minimum_extreme_modules < 1 or self.minimum_extreme_modules > 4:
            raise ValueError("episode_anchor.minimum_extreme_modules must be between 1 and 4")
        if self.memory_window < 1:
            raise ValueError("episode_anchor.memory_window must be >= 1")
