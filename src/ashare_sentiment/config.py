"""Configuration loading and validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML config, expand environment placeholders and validate essentials."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("The top-level YAML value must be a mapping")
    config = _expand_env(loaded)
    if overrides:
        config = _deep_merge(config, _expand_env(dict(overrides)))
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    data = config.get("data")
    scoring = config.get("scoring")
    if not isinstance(data, Mapping) or not isinstance(scoring, Mapping):
        raise ConfigError("Config must contain 'data' and 'scoring' mappings")
    weights = scoring.get("weights")
    required = {"breadth", "profit_effect", "liquidity", "options", "stretch"}
    if not isinstance(weights, Mapping) or set(weights) != required:
        raise ConfigError(f"scoring.weights must contain exactly: {sorted(required)}")
    numeric_weights = {key: float(value) for key, value in weights.items()}
    if any(value < 0 for value in numeric_weights.values()):
        raise ConfigError("scoring.weights cannot contain negative values")
    if sum(numeric_weights.values()) <= 0:
        raise ConfigError("scoring.weights must have a positive sum")
    percentile = scoring.get("percentile", {})
    if int(percentile.get("lookback", 0)) < int(percentile.get("min_periods", 0)):
        raise ConfigError("percentile.lookback must be >= percentile.min_periods")
    state_machine = config.get("state_machine", {})
    if state_machine:
        try:
            from .regime.models import EpisodeAnchorConfig, StateMachineConfig

            StateMachineConfig.from_mapping(state_machine)
            EpisodeAnchorConfig.from_mapping(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid regime configuration: {exc}") from exc
