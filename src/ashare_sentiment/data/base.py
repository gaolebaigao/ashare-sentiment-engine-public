"""Provider contract shared by all external data adapters.

Provider methods return raw, per-security observations. Feature engineering belongs
in ``factors/`` so the research layer never depends directly on a vendor API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd


class ProviderError(RuntimeError):
    """Base class for provider and transport failures."""


class ProviderDataUnavailable(ProviderError):
    """Raised when a provider cannot supply a requested dataset honestly."""


class DataProvider(ABC):
    """Stable interface for market-data providers.

    Date arguments are inclusive and may be ``YYYYMMDD`` strings, ISO date
    strings, ``datetime.date`` or ``datetime.datetime`` objects. Returned frames
    must contain a ``trade_date`` column when the source has a trading date.
    """

    name: str

    @abstractmethod
    def get_stock_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str | None = None,
    ) -> pd.DataFrame:
        """Return raw A-share daily OHLCV observations."""

    @abstractmethod
    def get_index_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str,
    ) -> pd.DataFrame:
        """Return daily observations for one index."""

    @abstractmethod
    def get_market_breadth(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Return per-stock observations used to derive breadth statistics."""

    @abstractmethod
    def get_limit_up_down(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Return per-stock limit-price and close observations."""

    @abstractmethod
    def get_margin_data(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Return daily margin-financing observations."""

    @abstractmethod
    def get_option_data(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Return raw option observations, or raise if unavailable."""


def format_date(value: str | date | Any) -> str:
    """Normalize supported date-like values to provider format ``YYYYMMDD``."""
    if isinstance(value, str):
        digits = value.replace("-", "").replace("/", "")
        if len(digits) == 8 and digits.isdigit():
            return digits
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d")
    raise ValueError(f"Unsupported date value: {value!r}")


def normalize_trade_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized, sorted, timezone-naive trade_date."""
    if "trade_date" not in frame.columns:
        return frame.copy()
    result = frame.copy()
    raw_dates = result["trade_date"]
    date_text = raw_dates.astype("string").str.replace(r"[-/]", "", regex=True)
    yyyymmdd = pd.to_datetime(
        date_text.where(date_text.str.fullmatch(r"\d{8}")),
        format="%Y%m%d",
        errors="coerce",
    )
    generic = pd.to_datetime(raw_dates, errors="coerce")
    result["trade_date"] = yyyymmdd.fillna(generic).dt.normalize()
    return result.sort_values("trade_date").reset_index(drop=True)
