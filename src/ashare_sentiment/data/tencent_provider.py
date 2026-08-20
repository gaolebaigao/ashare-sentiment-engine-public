"""Free Tencent historical daily-data provider.

Tencent exposes a public, symbol-oriented JSON endpoint used by several
open-source market-data adapters.  It is used here as a fallback for daily
stock and index history when the Eastmoney endpoint is unavailable.  The
endpoint does not provide the full-market panels required by breadth or the
limit-up/down pool, so those methods fail explicitly instead of returning a
partial substitute.
"""

from __future__ import annotations

from datetime import date
import re
from typing import Any

import pandas as pd
import requests

from .base import DataProvider, ProviderDataUnavailable, format_date, normalize_trade_date
from .http import CachedJsonClient, HttpDataError


class TencentProvider(DataProvider):
    name = "tencent-free"

    _DAILY_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    _QUOTE_URL = "https://qt.gtimg.cn/q="

    def __init__(
        self,
        *,
        cache_root: str = "data/raw/http",
        timeout: float = 15.0,
        min_interval: float = 0.20,
        retries: int = 3,
        client: CachedJsonClient | None = None,
        # The public endpoint is reliable up to roughly 2,000 rows; larger
        # values can return an empty payload instead of an error.
        max_rows: int = 2000,
        quote_session: requests.Session | None = None,
    ):
        self.client = client or CachedJsonClient(
            cache_root,
            timeout=timeout,
            min_interval=min_interval,
            retries=retries,
        )
        self.max_rows = max(1, int(max_rows))
        self.quote_session = quote_session or requests.Session()

    def get_stock_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str | None = None,
    ) -> pd.DataFrame:
        if not ts_code:
            raise ProviderDataUnavailable(
                "Tencent stock history is symbol-oriented; --symbol is required."
            )
        code, market = self._parse_ts_code(ts_code)
        return self._get_daily(code, market, start_date, end_date, ts_code=ts_code)

    def get_index_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str,
    ) -> pd.DataFrame:
        code, market = self._parse_ts_code(ts_code)
        return self._get_daily(code, market, start_date, end_date, ts_code=ts_code)

    def get_market_breadth(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable(
            "Tencent free endpoint is symbol-oriented and does not provide a complete market panel."
        )

    def get_limit_up_down(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable(
            "Tencent free endpoint does not provide the historical limit-up/down pool required here."
        )

    def get_margin_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable("Tencent free endpoint does not provide margin history.")

    def get_option_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable("Options are intentionally out of scope for MarketTemperature v0.1.")

    def get_realtime_snapshot(
        self,
        ts_codes: list[str],
        *,
        trade_date: str | date | None = None,
        batch_size: int = 500,
    ) -> pd.DataFrame:
        """Fetch uncached realtime quotes in bounded Tencent batches."""
        rows: list[dict[str, Any]] = []
        codes = [str(item).upper() for item in ts_codes if str(item).upper().endswith((".SH", ".SZ"))]
        for offset in range(0, len(codes), max(1, batch_size)):
            batch = codes[offset : offset + max(1, batch_size)]
            symbols = [f"{'sh' if code.endswith('.SH') else 'sz'}{code[:6]}" for code in batch]
            try:
                response = self.quote_session.get(
                    self._QUOTE_URL + ",".join(symbols),
                    timeout=self.client.timeout,
                    headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                detail = f"HTTP {status}" if status else type(exc).__name__
                raise ProviderDataUnavailable(f"Tencent realtime quote request failed ({detail})") from exc
            rows.extend(self._parse_realtime_text(response.content.decode("gbk", errors="ignore"), trade_date=trade_date))
        if not rows:
            raise ProviderDataUnavailable("Tencent returned no parseable realtime A-share quotes")
        frame = pd.DataFrame(rows)
        frame["universe_count"] = len(set(codes))
        frame["is_st"] = frame["name"].astype(str).str.contains(r"\*?ST", case=False, regex=True)
        frame = frame[~frame["is_st"]].copy()
        return normalize_trade_date(frame).drop_duplicates(["ts_code", "trade_date"], keep="last").reset_index(drop=True)

    @staticmethod
    def _parse_realtime_text(text: str, *, trade_date: str | date | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        fallback_date = pd.Timestamp(trade_date or date.today()).normalize()
        for symbol, payload in re.findall(r'v_((?:sh|sz)\d+)="([^"]*)"', text):
            values = payload.split("~")
            if len(values) < 36:
                continue
            ts_code = f"{symbol[2:8]}.{'SH' if symbol.startswith('sh') else 'SZ'}"
            stamp = str(values[30])[:8]
            parsed_date = pd.to_datetime(stamp, format="%Y%m%d", errors="coerce")
            turnover = str(values[35]).split("/")
            exact_amount = pd.to_numeric(turnover[2], errors="coerce") if len(turnover) >= 3 else pd.NA
            rows.append(
                {
                    "trade_date": fallback_date if pd.isna(parsed_date) else parsed_date,
                    "ts_code": ts_code,
                    "name": values[1],
                    "exchange": "SSE" if symbol.startswith("sh") else "SZSE",
                    "close": pd.to_numeric(values[3], errors="coerce"),
                    "pre_close": pd.to_numeric(values[4], errors="coerce"),
                    "open": pd.to_numeric(values[5], errors="coerce"),
                    "volume": pd.to_numeric(values[6], errors="coerce"),
                    "high": pd.to_numeric(values[33], errors="coerce"),
                    "low": pd.to_numeric(values[34], errors="coerce"),
                    "pct_chg": pd.to_numeric(values[32], errors="coerce"),
                    "amount_rmb": exact_amount,
                    "quote_time": values[30],
                }
            )
        return rows

    def _get_daily(
        self,
        code: str,
        market: str,
        start_date: str | date,
        end_date: str | date,
        *,
        ts_code: str,
    ) -> pd.DataFrame:
        symbol = f"{market}{code}"
        payload = self.client.get_json(
            self._DAILY_URL,
            {
                "param": f"{symbol},day,,,{self.max_rows},",
            },
        )
        if payload.get("code") not in (None, 0):
            raise HttpDataError(f"Tencent endpoint code={payload.get('code')}")
        data = payload.get("data") or {}
        symbol_data = data.get(symbol) or {}
        rows = symbol_data.get("day") or symbol_data.get("qfqday") or symbol_data.get("hfqday") or []
        if not rows:
            raise ProviderDataUnavailable(f"Tencent returned no daily rows for {ts_code}")

        records: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 9:
                continue
            records.append(
                {
                    "trade_date": row[0],
                    "open": row[1],
                    "close": row[2],
                    "high": row[3],
                    "low": row[4],
                    "volume": row[5],
                    "pct_chg": row[7],
                    # Tencent's public response uses the same amount unit as
                    # its front-end API; retain the raw numeric value and do
                    # not claim a unit conversion that the endpoint does not
                    # document consistently.
                    "amount_rmb": row[8],
                    "ts_code": ts_code,
                }
            )
        frame = pd.DataFrame(records)
        if frame.empty:
            raise ProviderDataUnavailable(f"Tencent returned no parseable daily rows for {ts_code}")
        for column in ("open", "close", "high", "low", "volume", "pct_chg", "amount_rmb"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["pre_close"] = frame["close"] / (1.0 + frame["pct_chg"] / 100.0)
        frame.loc[frame["pct_chg"].isna(), "pre_close"] = pd.NA
        frame = normalize_trade_date(frame)
        start = pd.Timestamp(format_date(start_date))
        end = pd.Timestamp(format_date(end_date))
        return frame[frame["trade_date"].between(start, end)].reset_index(drop=True)

    @staticmethod
    def _parse_ts_code(ts_code: str) -> tuple[str, str]:
        code, _, exchange = ts_code.upper().partition(".")
        market = {"SZ": "sz", "SH": "sh"}.get(exchange)
        if market is None:
            raise ProviderDataUnavailable(f"Only .SH and .SZ codes are supported by free provider: {ts_code}")
        return code.zfill(6), market
