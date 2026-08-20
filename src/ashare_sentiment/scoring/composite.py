"""Weighted scores with row-wise missing-factor renormalization."""

from __future__ import annotations

import json
from typing import Mapping

import pandas as pd


def renormalize_weights(
    available: Mapping[str, bool],
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Renormalize configured weights over available factors only."""
    active = {name: float(weight) for name, weight in weights.items() if available.get(name, False) and float(weight) > 0}
    total = sum(active.values())
    if total <= 0:
        return {name: 0.0 for name in weights}
    return {name: active.get(name, 0.0) / total for name in weights}


def combine_scores(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    score_column: str,
    metadata_prefix: str | None = None,
) -> pd.DataFrame:
    """Combine score columns row by row and record missing/effective weights."""
    result = frame.copy()
    columns = list(weights)
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    values = result[columns].apply(pd.to_numeric, errors="coerce")
    score_values: list[float] = []
    missing_values: list[str] = []
    effective_values: list[str] = []
    for _, row in values.iterrows():
        available = {column: pd.notna(row[column]) for column in columns}
        effective = renormalize_weights(available, weights)
        valid = [column for column in columns if available[column] and effective[column] > 0]
        score_values.append(sum(float(row[column]) * effective[column] for column in valid) if valid else float("nan"))
        missing_values.append(",".join(column for column in columns if not available[column]))
        effective_values.append(json.dumps(effective, ensure_ascii=False, sort_keys=True))
    result[score_column] = score_values
    prefix = metadata_prefix or score_column
    result[f"{prefix}_missing_factors"] = missing_values
    result[f"{prefix}_effective_weights"] = effective_values
    return result
