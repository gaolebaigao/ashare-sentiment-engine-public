"""One-factor-at-a-time v0.4 parameter robustness helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from ..regime import apply_state_machine, build_regime_indicators


ROBUSTNESS_GRID: dict[str, tuple[float, ...]] = {
    "raw_panic_threshold": (15.0, 20.0, 25.0),
    "raw_euphoria_threshold": (80.0, 85.0, 90.0),
    "ema_span": (2.0, 3.0, 5.0),
    "turning_delta": (3.0, 5.0, 7.0),
    "slope3_threshold": (2.0, 4.0, 6.0),
}


def variant_config(config: dict[str, Any], parameter: str, value: float) -> dict[str, Any]:
    """Return a copy with exactly one research parameter changed."""

    result = deepcopy(config)
    if parameter in {"raw_panic_threshold", "raw_euphoria_threshold"}:
        result.setdefault("episode_anchor", {})[parameter] = float(value)
    else:
        result.setdefault("state_machine", {})[parameter] = int(value) if parameter in {"ema_span", "slope3_threshold"} else float(value)
    return result


def build_variant_state(base_market_temperature: pd.DataFrame, config: dict[str, Any], parameter: str, value: float) -> pd.DataFrame:
    """Recompute a causal state table for one-factor diagnostics only."""

    variant = variant_config(config, parameter, value)
    indicators = build_regime_indicators(base_market_temperature, variant)
    return apply_state_machine(indicators, variant)
