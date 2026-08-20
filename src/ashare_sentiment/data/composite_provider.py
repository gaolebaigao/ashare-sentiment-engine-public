"""Free provider with an explicit fallback chain."""

from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd

from .akshare_provider import AkShareProvider
from .base import DataProvider, ProviderDataUnavailable, format_date, normalize_trade_date
from .eastmoney_provider import EastMoneyProvider
from .tencent_provider import TencentProvider


class CompositeFreeProvider(DataProvider):
    """Try Eastmoney JSON, Tencent JSON, then optional AkShare."""

    name = "free-composite"

    def __init__(
        self,
        primary: DataProvider | None = None,
        fallback: DataProvider | None = None,
        fallbacks: list[DataProvider] | None = None,
    ):
        self.primary = primary or EastMoneyProvider()
        providers = list(fallbacks or [])
        if fallback is not None:
            providers.insert(0, fallback)
        try:
            providers.append(AkShareProvider())
        except ProviderDataUnavailable:
            # The no-key path must not require the optional AkShare package.
            pass
        self.fallbacks = providers

    def get_stock_daily(self, start_date: str | date, end_date: str | date, ts_code: str | None = None) -> pd.DataFrame:
        return self._try("get_stock_daily", start_date, end_date, ts_code)

    def get_index_daily(self, start_date: str | date, end_date: str | date, ts_code: str) -> pd.DataFrame:
        return self._try("get_index_daily", start_date, end_date, ts_code)

    def get_market_breadth(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        return self._try("get_market_breadth", start_date, end_date)

    def get_limit_up_down(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        return self._try("get_limit_up_down", start_date, end_date)

    def get_margin_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        return self._try("get_margin_data", start_date, end_date)

    def get_option_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        return self._try("get_option_data", start_date, end_date)

    def _try(self, method_name: str, *args: object) -> pd.DataFrame:
        errors: list[str] = []
        for provider in (self.primary, *self.fallbacks):
            if provider is None:
                continue
            try:
                method: Callable[..., pd.DataFrame] = getattr(provider, method_name)
                frame = method(*args)
                if frame is None or frame.empty:
                    raise ProviderDataUnavailable(f"{provider.name} returned an empty dataframe")
                frame = normalize_trade_date(frame)
                self._validate_candidate(frame, provider.name, args)
                return frame
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
        raise ProviderDataUnavailable(f"All free providers failed for {method_name}: {' | '.join(errors)}")

    @staticmethod
    def _validate_candidate(frame: pd.DataFrame, provider_name: str, args: tuple[object, ...]) -> None:
        """Reject a partial fallback before it can be merged into the cache."""
        if "trade_date" not in frame.columns or len(args) < 2:
            return
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
        if dates.empty:
            raise ProviderDataUnavailable(f"{provider_name} returned no usable trade dates")
        start = pd.to_datetime(format_date(args[0]), format="%Y%m%d")
        end = pd.to_datetime(format_date(args[1]), format="%Y%m%d")
        if dates.min() > start or dates.max() < end:
            raise ProviderDataUnavailable(
                f"{provider_name} returned a non-atomic date range {dates.min().date()} to {dates.max().date()}"
            )
