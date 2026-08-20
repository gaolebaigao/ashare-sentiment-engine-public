"""Best-effort AkShare adapter for simple historical/index retrieval.

AkShare aggregates public web endpoints and is useful for exploration, but it is
not treated as a substitute for an exchange-authorized point-in-time dataset.
Methods that cannot meet the engine's historical requirements fail explicitly.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from .base import DataProvider, ProviderDataUnavailable, format_date, normalize_trade_date


class AkShareProvider(DataProvider):
    name = "akshare"

    def __init__(self, *, ak_module: Any | None = None):
        if ak_module is None:
            try:
                import akshare as ak
            except ImportError as exc:
                raise ProviderDataUnavailable(
                    "AkShare is not installed. Install the data extra: pip install -e '.[data]'"
                ) from exc
            ak_module = ak
        self.ak = ak_module

    def get_stock_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str | None = None,
    ) -> pd.DataFrame:
        if not ts_code:
            raise ProviderDataUnavailable(
                "AkShare historical stock endpoint is symbol-oriented; provide ts_code "
                "or use Tushare for full-universe downloads."
            )
        symbol = ts_code.split(".")[0]
        try:
            frame = self.ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=format_date(start_date),
                end_date=format_date(end_date),
                adjust="",
            )
        except Exception as exc:
            raise ProviderDataUnavailable(f"AkShare stock request failed: {exc}") from exc
        return self._normalize_ak_daily(frame)

    def get_index_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str,
    ) -> pd.DataFrame:
        symbol = self._ak_index_symbol(ts_code)
        try:
            frame = self.ak.stock_zh_index_daily_em(symbol=symbol)
        except Exception as exc:
            raise ProviderDataUnavailable(f"AkShare index request failed: {exc}") from exc
        result = self._normalize_ak_daily(frame)
        start = pd.to_datetime(format_date(start_date))
        end = pd.to_datetime(format_date(end_date))
        return result[result["trade_date"].between(start, end)].reset_index(drop=True)

    def get_market_breadth(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        universe_method = getattr(self.ak, "stock_info_a_code_name", None)
        history_method = getattr(self.ak, "stock_zh_a_hist", None)
        if universe_method is None or history_method is None:
            raise ProviderDataUnavailable("AkShare full-panel historical methods are unavailable")
        try:
            universe = universe_method()
        except Exception as exc:
            raise ProviderDataUnavailable(f"AkShare stock universe request failed: {exc}") from exc
        code_column = "code" if "code" in universe.columns else "股票代码"
        name_column = "name" if "name" in universe.columns else "股票简称"
        frames = []
        for record in universe.to_dict("records"):
            code = str(record.get(code_column, "")).zfill(6)
            try:
                history = history_method(
                    symbol=code,
                    period="daily",
                    start_date=format_date(start_date),
                    end_date=format_date(end_date),
                    adjust="",
                )
                normalized = self._normalize_ak_daily(history)
                if normalized.empty:
                    continue
                normalized["ts_code"] = code
                normalized["name"] = record.get(name_column)
                frames.append(normalized)
            except Exception:
                continue
        if not frames:
            raise ProviderDataUnavailable("AkShare returned no full-panel history")
        return pd.concat(frames, ignore_index=True)

    def get_limit_up_down(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        up_method = getattr(self.ak, "stock_zt_pool_em", None)
        down_method = getattr(self.ak, "stock_zt_pool_dtgc_em", None)
        if up_method is None or down_method is None:
            raise ProviderDataUnavailable("AkShare limit-pool methods are unavailable")
        rows = []
        for day in pd.date_range(start_date, end_date, freq="D"):
            date_text = day.strftime("%Y%m%d")
            for method, is_up in ((up_method, True), (down_method, False)):
                try:
                    pool = method(date=date_text)
                except Exception:
                    continue
                if pool is None or pool.empty:
                    continue
                code_column = "代码" if "代码" in pool.columns else "code"
                name_column = "名称" if "名称" in pool.columns else "name"
                for row in pool.to_dict("records"):
                    code = str(row.get(code_column, "")).zfill(6)
                    rows.append(
                        {
                            "trade_date": day,
                            "ts_code": code,
                            "name": row.get(name_column),
                            "is_limit_up": is_up,
                            "is_limit_down": not is_up,
                        }
                    )
        if not rows:
            raise ProviderDataUnavailable("AkShare returned no limit-pool rows")
        return pd.DataFrame(rows)

    def get_margin_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable(
            "AkShare margin endpoints need exchange-specific reconciliation; no unified "
            "dataset is silently substituted in V0.1."
        )

    def get_option_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable(
            "AkShare option data is not yet mapped to the engine's CSI 300/1000 IV and "
            "put-call contract universe."
        )

    @staticmethod
    def _ak_index_symbol(ts_code: str) -> str:
        code, _, exchange = ts_code.partition(".")
        prefix = {"SH": "sh", "SZ": "sz"}.get(exchange.upper(), "")
        return f"{prefix}{code}" if prefix else code

    @staticmethod
    def _normalize_ak_daily(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["trade_date"])
        mapping = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover_rate",
        }
        result = frame.rename(columns=mapping).copy()
        return normalize_trade_date(result)
