"""Extreme episode anchor flags for MarketTemperature v0.3.1.

This module only identifies current-observation anchor events. Memory,
episode IDs, closing and re-arming are handled by the causal state-machine
loop because confirmation can close an episode on the current observation.
"""

from __future__ import annotations

import pandas as pd

from .models import EpisodeAnchorConfig, StateMachineConfig


MODULE_COLUMNS = (
    "breadth_score",
    "profit_effect_score",
    "liquidity_score",
    "stretch_score",
)


def add_episode_anchor_flags(frame: pd.DataFrame, config: EpisodeAnchorConfig | dict) -> pd.DataFrame:
    """Add causal current-day panic/euphoria anchor flags and module counts."""

    if isinstance(config, EpisodeAnchorConfig):
        settings = config
    elif isinstance(config, StateMachineConfig):
        settings = EpisodeAnchorConfig()
    else:
        settings = EpisodeAnchorConfig.from_mapping(config)
    output = frame.copy()
    valid_flag = output.get("regime_valid_observation", pd.Series(True, index=output.index))
    quality = output.get("quality", output.get("market_temperature_quality", pd.Series("A", index=output.index)))
    invalid = valid_flag.eq(False) | quality.astype(str).str.upper().eq("INVALID")
    for column in MODULE_COLUMNS:
        if column not in output:
            output[column] = pd.NA
    modules = output[list(MODULE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    output["panic_module_count"] = modules.le(settings.module_panic_threshold).sum(axis=1).astype(int)
    output["euphoria_module_count"] = modules.ge(settings.module_euphoria_threshold).sum(axis=1).astype(int)
    raw = pd.to_numeric(output.get("raw_temperature", pd.Series(index=output.index, dtype=float)), errors="coerce")
    smoothed = pd.to_numeric(output.get("smoothed_temperature", pd.Series(index=output.index, dtype=float)), errors="coerce")
    invalid = invalid | raw.isna() | smoothed.isna()
    output["raw_panic_anchor"] = (
        raw.le(settings.raw_panic_threshold)
        & output["panic_module_count"].ge(settings.minimum_extreme_modules)
        & ~invalid
    ).fillna(False)
    output["smoothed_panic_anchor"] = (smoothed.le(settings.smoothed_panic_threshold) & ~invalid).fillna(False)
    output["raw_euphoria_anchor"] = (
        raw.ge(settings.raw_euphoria_threshold)
        & output["euphoria_module_count"].ge(settings.minimum_extreme_modules)
        & ~invalid
    ).fillna(False)
    output["smoothed_euphoria_anchor"] = (smoothed.ge(settings.smoothed_euphoria_threshold) & ~invalid).fillna(False)
    output["panic_anchor_event"] = output["raw_panic_anchor"] | output["smoothed_panic_anchor"]
    output["euphoria_anchor_event"] = output["raw_euphoria_anchor"] | output["smoothed_euphoria_anchor"]
    output.loc[invalid, ["panic_module_count", "euphoria_module_count"]] = 0
    return output
