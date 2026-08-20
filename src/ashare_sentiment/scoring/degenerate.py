"""Detection of factors that are too flat to carry a composite weight."""

from __future__ import annotations

import pandas as pd


def degenerate_mask(
    series: pd.Series,
    *,
    window: int = 20,
    max_unique: int = 2,
    minimum_valid: int = 5,
    variance_floor: float = 1e-12,
) -> pd.Series:
    """Return a trailing mask for missing or effectively constant factors."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    valid_count = values.rolling(window, min_periods=1).count()
    unique_count = values.rolling(window, min_periods=1).apply(
        lambda current: float(pd.Series(current).dropna().nunique()), raw=False
    )
    variance = values.rolling(window, min_periods=1).var(ddof=0)
    return (
        valid_count.lt(minimum_valid)
        | unique_count.le(max_unique)
        | variance.fillna(0.0).le(variance_floor)
    ).rename(series.name)


def mask_degenerate_scores(
    raw: pd.DataFrame,
    score_frame: pd.DataFrame,
    factor_names: list[str],
    *,
    window: int = 20,
    max_unique: int = 2,
    minimum_valid: int = 5,
    variance_floor: float = 1e-12,
) -> tuple[pd.DataFrame, pd.Series]:
    """Set degenerate factor scores to unavailable and return row metadata."""
    flags = pd.DataFrame(index=raw.index)
    for name in factor_names:
        if name not in raw.columns or name not in score_frame.columns:
            continue
        mask = degenerate_mask(
            raw[name],
            window=window,
            max_unique=max_unique,
            minimum_valid=minimum_valid,
            variance_floor=variance_floor,
        )
        score_frame.loc[mask, name] = pd.NA
        flags[name] = mask.to_numpy()
    if flags.empty:
        return score_frame, pd.Series("", index=raw.index, dtype="string")
    labels = flags.apply(
        lambda row: ",".join(row.index[row.fillna(False).astype(bool)]), axis=1
    )
    return score_frame, labels
