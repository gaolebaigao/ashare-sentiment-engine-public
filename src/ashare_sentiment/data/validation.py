"""Data quality checks run before a dataset enters factor research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    rows: int = 0
    severity: str = "error"


class DataValidationError(ValueError):
    """Raised when strict validation finds a blocking issue."""


def validate_timeseries(
    frame: pd.DataFrame,
    *,
    date_column: str = "trade_date",
    key_columns: Iterable[str] | None = None,
    price_columns: Iterable[str] = ("open", "high", "low", "close"),
    volume_columns: Iterable[str] = ("volume", "vol"),
    outlier_return: float = 0.80,
    strict: bool = False,
) -> list[ValidationIssue]:
    """Return issues for duplicates, invalid prices, volume and extreme returns.

    Missing trading days are reported only when a calendar is supplied by a
    higher-level caller; weekends and exchange holidays are not missing data.
    """
    issues: list[ValidationIssue] = []
    if date_column not in frame.columns:
        issues.append(ValidationIssue("MISSING_DATE_COLUMN", f"Missing column: {date_column}"))
        return _finish(issues, strict)

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    invalid_dates = int(dates.isna().sum())
    if invalid_dates:
        issues.append(ValidationIssue("INVALID_DATES", "Some trade dates cannot be parsed", invalid_dates))
    duplicate_key = frame[list(key_columns)].duplicated() if key_columns else dates.duplicated()
    duplicate_dates = int(duplicate_key.sum())
    if duplicate_dates:
        issues.append(ValidationIssue("DUPLICATE_DATES", "Duplicate trade dates found", duplicate_dates))

    for column in price_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = int((values <= 0).sum() + values.isna().sum())
        if bad:
            issues.append(ValidationIssue("IMPOSSIBLE_PRICES", f"Invalid values in {column}", bad))

    if {"high", "low"}.issubset(frame.columns):
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        inconsistent_range = int((high < low).sum())
        if inconsistent_range:
            issues.append(
                ValidationIssue(
                    "INCONSISTENT_PRICE_RANGE",
                    "High price is below low price",
                    inconsistent_range,
                )
            )
    for bound in ("high", "low"):
        if {"close", bound}.issubset(frame.columns):
            close = pd.to_numeric(frame["close"], errors="coerce")
            value = pd.to_numeric(frame[bound], errors="coerce")
            inconsistent_close = int(((close > value) if bound == "high" else (close < value)).sum())
            if inconsistent_close:
                issues.append(
                    ValidationIssue(
                        "CLOSE_OUTSIDE_RANGE",
                        "Close price is above high" if bound == "high" else "Close price is below low",
                        inconsistent_close,
                    )
                )

    for column in volume_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = int((values < 0).sum())
        if bad:
            issues.append(ValidationIssue("NEGATIVE_VOLUME", f"Negative values in {column}", bad))

    if "close" in frame.columns:
        close = pd.to_numeric(frame["close"], errors="coerce")
        returns = close.pct_change(fill_method=None).abs()
        outliers = int((returns > outlier_return).sum())
        if outliers:
            issues.append(
                ValidationIssue(
                    "OUTLIER_RETURNS",
                    f"Absolute close-to-close returns exceed {outlier_return:.0%}",
                    outliers,
                    severity="warning",
                )
            )
    return _finish(issues, strict)


def validate_universe_membership(
    frame: pd.DataFrame,
    expected_members: set[str],
    *,
    code_column: str = "ts_code",
    strict: bool = False,
) -> list[ValidationIssue]:
    """Check that all expected members are represented in a snapshot."""
    if code_column not in frame.columns:
        return _finish([ValidationIssue("MISSING_UNIVERSE_COLUMN", f"Missing column: {code_column}")], strict)
    present = set(frame[code_column].dropna().astype(str))
    missing = expected_members - present
    issues = []
    if missing:
        issues.append(
            ValidationIssue(
                "MISSING_UNIVERSE_MEMBERS",
                f"{len(missing)} expected universe members are absent",
                len(missing),
                severity="warning",
            )
        )
    return _finish(issues, strict)


def _finish(issues: list[ValidationIssue], strict: bool) -> list[ValidationIssue]:
    if strict and any(issue.severity == "error" for issue in issues):
        summary = "; ".join(issue.code for issue in issues if issue.severity == "error")
        raise DataValidationError(summary)
    return issues
