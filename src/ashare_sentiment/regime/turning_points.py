"""Turning-point features and module confirmation inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import StateMachineConfig
from .episode_anchors import add_episode_anchor_flags
from .smoothing import add_smoothing_metrics, valid_lag


MODULE_COLUMNS = (
    "breadth_score",
    "profit_effect_score",
    "liquidity_score",
    "stretch_score",
)


def _quality_column(frame: pd.DataFrame) -> pd.Series:
    if "market_temperature_quality" in frame:
        return frame["market_temperature_quality"].astype(str).str.upper()
    if "quality" in frame:
        return frame["quality"].astype(str).str.upper()
    return pd.Series("A", index=frame.index, dtype=object)


def build_regime_indicators(frame: pd.DataFrame, config: StateMachineConfig | dict) -> pd.DataFrame:
    """Build all v0.3 numeric inputs without filling INVALID observations."""

    settings = config if isinstance(config, StateMachineConfig) else StateMachineConfig.from_mapping(config)
    output = frame.copy().sort_values("trade_date").reset_index(drop=True)
    if "raw_temperature" not in output:
        output["raw_temperature"] = pd.to_numeric(output.get("raw_market_temperature"), errors="coerce")
    else:
        output["raw_temperature"] = pd.to_numeric(output["raw_temperature"], errors="coerce")
    quality = _quality_column(output)
    invalid = quality.eq("INVALID") | output["raw_temperature"].isna()
    output.loc[invalid, "raw_temperature"] = np.nan
    output["regime_valid_observation"] = ~invalid
    output = add_smoothing_metrics(
        output,
        raw_column="raw_temperature",
        span=settings.ema_span,
        slope_lookback=settings.slope_lookback,
        turning_window=settings.turning_window,
    )
    for column in ("raw_temperature", "smoothed_temperature", "temperature_shock", "slope1", "slope3", "rolling_high_10", "rolling_low_10", "recovery_from_low", "drop_from_high"):
        output.loc[invalid, column] = np.nan

    for module in MODULE_COLUMNS:
        if module not in output:
            output[module] = np.nan
        values = pd.to_numeric(output[module], errors="coerce").mask(invalid)
        output[module] = values
        lagged = valid_lag(values, 3)
        output[f"{module.replace('_score', '')}_delta3"] = (values - lagged).to_numpy()

    deltas = output[[f"{module.replace('_score', '')}_delta3" for module in MODULE_COLUMNS]]
    output["improving_modules"] = deltas.gt(0).sum(axis=1).astype(int)
    output["deteriorating_modules"] = deltas.lt(0).sum(axis=1).astype(int)
    output.loc[invalid, ["improving_modules", "deteriorating_modules"]] = 0
    output["quality"] = quality.where(~invalid, "INVALID")
    output["confidence"] = output.get("confidence", output["quality"].map({"A": "HIGH", "B": "MEDIUM", "C": "LOW", "INVALID": "NONE"}))
    output.loc[invalid, "confidence"] = "NONE"
    return add_episode_anchor_flags(output, config)
