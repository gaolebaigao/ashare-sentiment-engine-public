"""Evidence-aware confidence calculations for the advisory layer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MODULE_LABELS = {
    "breadth": "Breadth",
    "profit_effect": "Profit Effect",
    "liquidity": "Liquidity",
    "stretch": "Stretch",
}
MODULE_SCORE_COLUMNS = tuple(f"{name}_score" for name in MODULE_LABELS)
MODULE_DELTA_COLUMNS = tuple(f"{name}_delta3" for name in MODULE_LABELS)


def _number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    if pd.isna(value) and column == "raw_temperature":
        value = pd.to_numeric(row.get("market_temperature", row.get("temperature")), errors="coerce")
    if pd.isna(value) and column == "smoothed_temperature":
        value = pd.to_numeric(row.get("ema_temperature"), errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def module_confirmation(row: pd.Series, direction: str) -> tuple[int, tuple[str, ...]]:
    """Count modules confirming an ICE (up) or HOT (down) direction."""

    sign = 1 if direction.lower() in {"ice", "recovery", "up", "improving"} else -1
    names: list[str] = []
    for name, column in zip(MODULE_LABELS, MODULE_DELTA_COLUMNS):
        value = _number(row, column)
        if np.isfinite(value) and ((value > 0) if sign > 0 else (value < 0)):
            names.append(MODULE_LABELS[name])
    # Synthetic rows used by the v0.3 tests sometimes provide only the count.
    if not names:
        count_column = "improving_modules" if sign > 0 else "deteriorating_modules"
        count = _number(row, count_column)
        if np.isfinite(count) and count > 0:
            return min(int(count), 4), tuple()
    return len(names), tuple(names)


def _quality_score(row: pd.Series) -> float:
    quality = str(row.get("quality", row.get("market_temperature_quality", "A"))).upper()
    if quality == "INVALID" or str(row.get("data_quality_status", "")).upper() == "INVALID":
        return 0.0
    return {"A": 1.0, "B": 0.72, "C": 0.45}.get(quality, 0.45)


def confidence_components(
    row: pd.Series,
    *,
    confirming_modules: int | None = None,
    state_persistence: int | None = None,
) -> dict[str, float]:
    """Return transparent components used to classify today's signal clarity.

    These are confidence in the *state identification*, not probabilities of a
    future return. The formula is a stable interpretive rubric and does not
    tune the frozen MarketTemperature/state-machine parameters.
    """

    state = str(row.get("state", "DATA_INVALID"))
    modules = confirming_modules
    if modules is None:
        direction = "ice" if "ICE" in state or state in {"PANIC_FALLING", "EXTREME_PANIC"} else "hot"
        modules, _ = module_confirmation(row, direction)
    modules = max(0, min(int(modules), 4))
    agreement = 0.55 + 0.45 * (modules / 4.0)

    smooth = _number(row, "smoothed_temperature")
    raw = _number(row, "raw_temperature")
    slope = _number(row, "slope3")
    if not np.isfinite(smooth) and np.isfinite(raw):
        smooth = raw
    if state == "DATA_INVALID" or not np.isfinite(smooth):
        distance = 0.0
    elif state in {"PANIC_FALLING", "EXTREME_PANIC", "COLD"}:
        distance = float(np.clip((40.0 - smooth) / 40.0, 0.0, 1.0))
    elif state in {"HOT", "EUPHORIA_RISING", "HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"}:
        distance = float(np.clip((smooth - 60.0) / 40.0, 0.0, 1.0))
    elif state in {"ICE_REVERSAL_WATCH", "ICE_REVERSAL"}:
        recovery = _number(row, "recovery_from_post_panic_low")
        if not np.isfinite(recovery):
            recovery = _number(row, "recovery_from_low")
        distance = float(np.clip((recovery if np.isfinite(recovery) else 0.0) / 20.0, 0.0, 1.0))
    else:
        distance = 0.65

    slope_strength = float(np.clip(abs(slope) / 8.0, 0.0, 1.0)) if np.isfinite(slope) else 0.0
    persistence_value = state_persistence
    if persistence_value is None:
        persistence_value = pd.to_numeric(row.get("state_duration"), errors="coerce")
    persistence = float(np.clip(float(persistence_value) / 3.0, 0.0, 1.0)) if pd.notna(persistence_value) else 0.25
    data_quality = _quality_score(row)

    return {
        "data_quality": data_quality,
        "module_agreement": agreement,
        "temperature_distance": distance,
        "slope_strength": slope_strength,
        "persistence": persistence,
    }


def calculate_signal_confidence(
    row: pd.Series,
    *,
    confirming_modules: int | None = None,
    state_persistence: int | None = None,
) -> str:
    """Classify signal clarity as LOW, MEDIUM or HIGH."""

    parts = confidence_components(
        row,
        confirming_modules=confirming_modules,
        state_persistence=state_persistence,
    )
    score = (
        0.30 * parts["data_quality"]
        + 0.25 * parts["module_agreement"]
        + 0.20 * parts["temperature_distance"]
        + 0.15 * parts["slope_strength"]
        + 0.10 * parts["persistence"]
    )
    if parts["data_quality"] <= 0:
        return "LOW"
    if score >= 0.70:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def research_evidence_for(state: str, advisory_signal: str) -> dict[str, str]:
    """Return static, non-probabilistic evidence metadata from v0.4 research."""

    if state in {"ICE_REVERSAL_WATCH", "ICE_REVERSAL"} or advisory_signal in {"BUY_WATCH", "BUY_REFERENCE"} and "ICE" in state:
        medium = "MODERATE"
        short = "WEAK"
        overall = "MODERATE" if state == "ICE_REVERSAL" else "WEAK"
    elif state in {"HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"}:
        short = "WEAK"
        medium = "WEAK_TO_MODERATE"
        overall = "WEAK_TO_MODERATE"
    elif state in {"HOT", "EUPHORIA_RISING"}:
        short = "WEAK"
        medium = "WEAK_TO_MODERATE"
        overall = "WEAK"
    else:
        short = "UNPROVEN"
        medium = "UNPROVEN"
        overall = "UNPROVEN"
    return {
        "research_evidence": overall,
        "short_term_evidence": short,
        "medium_term_evidence": medium,
    }
