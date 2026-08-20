"""Free Eastmoney JSON endpoint provider.

The provider uses public JSON endpoints documented by the open-source AkShare
ecosystem, without a token or login. It does not bypass authentication, CAPTCHA,
or rate limits. Requests are cached and throttled conservatively.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd

from .base import DataProvider, ProviderDataUnavailable, format_date, normalize_trade_date
from .http import CachedJsonClient, HttpDataError


class EastMoneyProvider(DataProvider):
    name = "eastmoney-free"

    _KLINE_URLS = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
    )
    _LIST_URLS = (
        "https://82.push2.eastmoney.com/api/qt/clist/get",
        "https://push2.eastmoney.com/api/qt/clist/get",
    )
    _LIMIT_UP_URLS = ("https://push2ex.eastmoney.com/getTopicZTPool",)
    _LIMIT_DOWN_URLS = ("https://push2ex.eastmoney.com/getTopicDTPool",)
    _UT = "7eea3edcaed734bea9cbfc24409ed989"
    _LIST_UT = "bd1d9ddb04089700cf9c27f6f7426281"

    def __init__(
        self,
        *,
        cache_root: str = "data/raw/http",
        timeout: float = 15.0,
        min_interval: float = 0.20,
        retries: int = 3,
        include_beijing: bool = False,
        exclude_st: bool = True,
        max_symbols: int | None = None,
        client: CachedJsonClient | None = None,
    ):
        self.client = client or CachedJsonClient(
            cache_root,
            timeout=timeout,
            min_interval=min_interval,
            retries=retries,
        )
        # A stale or host-only localhost proxy can make the public quote host
        # unreachable from a container even though direct access works.  Only
        # bypass environment proxies for the internally-created Eastmoney
        # session; injected test/custom clients keep their own policy.
        if client is None:
            self.client.session.trust_env = False
        self.include_beijing = include_beijing
        self.exclude_st = exclude_st
        self.max_symbols = max_symbols
        self._universe: pd.DataFrame | None = None

    def get_stock_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str | None = None,
    ) -> pd.DataFrame:
        if not ts_code:
            raise ProviderDataUnavailable(
                "Eastmoney stock history is symbol-oriented; use get_market_breadth for the full panel."
            )
        code, market = self._parse_ts_code(ts_code)
        return self._get_kline(code, market, start_date, end_date, ts_code=ts_code)

    def get_index_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str,
    ) -> pd.DataFrame:
        code, market = self._parse_ts_code(ts_code)
        return self._get_kline(code, market, start_date, end_date, ts_code=ts_code)

    def get_realtime_index(self, ts_code: str, *, trade_date: str | date | None = None) -> pd.DataFrame:
        """Return the current in-progress daily bar for one index."""
        day = pd.Timestamp(trade_date or date.today()).date().isoformat()
        code, market = self._parse_ts_code(ts_code)
        return self._get_kline(code, market, day, day, ts_code=ts_code, refresh=True)

    def get_market_breadth(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        universe = self.get_stock_universe()
        if self.max_symbols is not None:
            universe = universe.head(self.max_symbols).copy()
        if universe.empty:
            raise ProviderDataUnavailable("Free provider returned an empty SSE/SZSE stock universe")

        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        for row in universe.itertuples(index=False):
            try:
                frame = self._get_kline(
                    row.code,
                    int(row.market),
                    start_date,
                    end_date,
                    ts_code=row.ts_code,
                )
                if not frame.empty:
                    frame["name"] = row.name
                    frame["exchange"] = row.exchange
                    if getattr(row, "list_date", None):
                        frame["list_date"] = row.list_date
                    frames.append(frame)
            except ProviderDataUnavailable:
                failures.append(row.ts_code)

        if not frames:
            raise ProviderDataUnavailable("Free provider failed to retrieve every requested stock history")
        result = pd.concat(frames, ignore_index=True)
        # A partial panel is more dangerous than a hard failure. By default we
        # require every requested symbol; smoke tests may explicitly cap symbols.
        if failures and self.max_symbols is None:
            raise ProviderDataUnavailable(
                f"Free provider failed for {len(failures)} of {len(universe)} symbols; "
                "refusing to compute breadth from a partial panel"
            )
        return normalize_trade_date(result)

    def get_realtime_snapshot(self, *, trade_date: str | date | None = None) -> pd.DataFrame:
        """Return one uncached full-market quote snapshot for intraday scoring."""
        fields = "f2,f3,f5,f6,f8,f12,f13,f14,f15,f16,f17,f18,f124"
        records: list[dict[str, Any]] = []
        page = 1
        total: int | None = None
        while total is None or len(records) < total:
            payload = self._get_from_hosts(
                self._LIST_URLS,
                {
                    "pn": page,
                    "pz": 5000,
                    "po": 1,
                    "np": 1,
                    "ut": self._LIST_UT,
                    "fltt": 2,
                    "invt": 2,
                    "fid": "f3",
                    "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23" + (",m:0 t:81" if self.include_beijing else ""),
                    "fields": fields,
                },
                refresh=True,
            )
            data = payload.get("data") or {}
            diff = data.get("diff") or []
            if total is None:
                total = int(data.get("total") or 0)
            if not diff:
                break
            records.extend(diff)
            page += 1
            if page > 20:
                break
        if not records:
            raise ProviderDataUnavailable("Eastmoney returned no realtime A-share quotes")
        frame = pd.DataFrame(records).rename(
            columns={
                "f2": "close", "f3": "pct_chg", "f5": "volume", "f6": "amount_rmb",
                "f8": "turnover_rate", "f12": "code", "f13": "market", "f14": "name",
                "f15": "high", "f16": "low", "f17": "open", "f18": "pre_close",
                "f124": "quote_time",
            }
        )
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["market"] = pd.to_numeric(frame["market"], errors="coerce").astype("Int64")
        frame["exchange"] = frame["market"].map({0: "SZSE", 1: "SSE"})
        frame["ts_code"] = frame["code"] + "." + frame["market"].map({0: "SZ", 1: "SH"}).fillna("")
        for column in ("open", "high", "low", "close", "pre_close", "volume", "amount_rmb", "pct_chg", "turnover_rate"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "quote_time" in frame:
            frame["quote_time"] = (
                pd.to_datetime(frame["quote_time"], unit="s", errors="coerce", utc=True)
                .dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
            )
        frame = frame[frame["exchange"].isin({"SSE", "SZSE"}) & frame["close"].gt(0)].copy()
        if self.exclude_st:
            frame = frame[~frame["name"].astype(str).str.contains(r"\*?ST", case=False, regex=True)].copy()
        if self.max_symbols is not None:
            frame = frame.sort_values("ts_code").head(max(int(self.max_symbols), 0)).copy()
        frame["trade_date"] = pd.Timestamp(trade_date or date.today()).normalize()
        frame["universe_count"] = int(frame["ts_code"].nunique())
        frame["is_st"] = False
        return normalize_trade_date(frame).drop_duplicates(["ts_code", "trade_date"], keep="last").reset_index(drop=True)

    def get_realtime_limit_up_down(self, *, trade_date: str | date | None = None) -> pd.DataFrame:
        """Return uncached current limit pools; an empty pool is a valid snapshot."""
        day_text = pd.Timestamp(trade_date or date.today()).strftime("%Y%m%d")
        # Pool rows already carry code, market and name.  Do not make the
        # realtime path depend on the less reliable full-market list endpoint.
        universe = pd.DataFrame(columns=["name"]).rename_axis("ts_code")
        rows: list[dict[str, Any]] = []
        for urls, is_up in ((self._LIMIT_UP_URLS, True), (self._LIMIT_DOWN_URLS, False)):
            sort = "fbt:asc" if is_up else "fund:asc"
            for item in self._get_limit_pool(urls, day_text, refresh=True, sort=sort):
                row = self._limit_row(item, day_text, universe, is_up=is_up)
                row["limit_up_source"] = "eastmoney-intraday"
                row["limit_down_source"] = "eastmoney-intraday"
                row["limit_up_status"] = "REAL_LIST" if is_up else "REAL_ZERO"
                row["limit_down_status"] = "REAL_ZERO" if is_up else "REAL_LIST"
                rows.append(row)
        columns = ["trade_date", "ts_code", "is_limit_up", "is_limit_down", "limit_up_source", "limit_down_source", "limit_up_status", "limit_down_status"]
        return normalize_trade_date(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=columns)

    def get_limit_up_down(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        universe = self.get_stock_universe().set_index("ts_code")
        rows: list[dict[str, Any]] = []
        for day in pd.date_range(start_date, end_date, freq="D"):
            day_text = day.strftime("%Y%m%d")
            up = self._get_limit_pool(self._LIMIT_UP_URLS, day_text)
            down = self._get_limit_pool(self._LIMIT_DOWN_URLS, day_text)
            for item in up:
                rows.append(self._limit_row(item, day_text, universe, is_up=True))
            for item in down:
                rows.append(self._limit_row(item, day_text, universe, is_up=False))
        if not rows:
            raise ProviderDataUnavailable(
                f"No free limit-up/down records returned for {format_date(start_date)} to {format_date(end_date)}"
            )
        return normalize_trade_date(pd.DataFrame(rows))

    def get_margin_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable(
            "No stable, no-key, full-history margin source was verified for the free path; "
            "margin remains optional in MarketTemperature v0.1."
        )

    def get_option_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable("Options are intentionally out of scope for MarketTemperature v0.1")

    def get_stock_universe(self) -> pd.DataFrame:
        if self._universe is not None:
            return self._universe.copy()
        # f12/f13/f14 are stable list fields. Listing dates are not exposed
        # consistently by this free snapshot endpoint, so do not pretend f26 is
        # a point-in-time universe source.
        fields = "f12,f13,f14"
        records: list[dict[str, Any]] = []
        page = 1
        total = None
        while total is None or len(records) < total:
            payload = self._get_from_hosts(
                self._LIST_URLS,
                {
                    "pn": page,
                    "pz": 100,
                    "po": 1,
                    "np": 1,
                    "ut": self._LIST_UT,
                    "fltt": 2,
                    "invt": 2,
                    "fid": "f3",
                    "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23" + (",m:0 t:81" if self.include_beijing else ""),
                    "fields": fields,
                }
            )
            data = payload.get("data") or {}
            diff = data.get("diff") or []
            if total is None:
                total = int(data.get("total") or 0)
            if not diff:
                break
            records.extend(diff)
            page += 1
            if page > 20:
                break
        if not records:
            raise ProviderDataUnavailable("Eastmoney returned no stock-universe rows")
        frame = pd.DataFrame(records)
        frame = frame.rename(columns={"f12": "code", "f13": "market", "f14": "name", "f26": "list_date"})
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["market"] = pd.to_numeric(frame["market"], errors="coerce").astype("Int64")
        frame["exchange"] = frame["market"].map({0: "SZSE", 1: "SSE"})
        frame["ts_code"] = frame["code"] + "." + frame["market"].map({0: "SZ", 1: "SH"}).fillna("")
        if "list_date" not in frame.columns:
            frame["list_date"] = pd.NA
        frame["list_date"] = frame["list_date"].replace({"-": pd.NA, "": pd.NA})
        if self.exclude_st:
            frame = frame[~frame["name"].astype(str).str.contains(r"\*?ST", case=False, regex=True)]
        frame = frame[frame["exchange"].isin({"SSE", "SZSE"})].copy()
        frame = frame.drop_duplicates("ts_code").reset_index(drop=True)
        self._universe = frame
        return frame.copy()

    def _get_kline(
        self,
        code: str,
        market: int,
        start_date: str | date,
        end_date: str | date,
        *,
        ts_code: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        payload = self._get_from_hosts(
            self._KLINE_URLS,
            {
                "secid": f"{market}.{code}",
                "klt": 101,
                "fqt": 0,
                "beg": format_date(start_date),
                "end": format_date(end_date),
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "ut": self._UT,
            },
            refresh=refresh,
        )
        data = payload.get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            return pd.DataFrame(columns=["trade_date", "ts_code", "open", "high", "low", "close"])
        columns = [
            "trade_date", "open", "close", "high", "low", "volume", "amount_rmb",
            "amplitude", "pct_chg", "change", "turnover_rate",
        ]
        frame = pd.DataFrame([str(item).split(",") for item in klines], columns=columns)
        for column in columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["ts_code"] = ts_code
        frame["pre_close"] = frame["close"] / (1.0 + frame["pct_chg"] / 100.0)
        frame.loc[frame["pct_chg"].isna(), "pre_close"] = pd.NA
        return normalize_trade_date(frame)

    def _get_limit_pool(self, urls: Iterable[str], day_text: str, *, refresh: bool = False, sort: str = "fbt:asc") -> list[dict[str, Any]]:
        payload = self._get_from_hosts(
            urls,
            {
                "ut": self._UT,
                "dpt": "wz.ztzt",
                "Pageindex": 0,
                "pagesize": 10000,
                "sort": sort,
                "date": day_text,
                "_": "1621590489736",
            },
            refresh=refresh,
        )
        return list(((payload.get("data") or {}).get("pool") or []))

    def _get_from_hosts(self, urls: Iterable[str], params: dict[str, Any], *, refresh: bool = False) -> dict[str, Any]:
        last_error: Exception | None = None
        for url in urls:
            try:
                payload = self.client.get_json(url, params, refresh=True) if refresh else self.client.get_json(url, params)
                if payload.get("rc") not in (None, 0):
                    raise HttpDataError(f"endpoint rc={payload.get('rc')}")
                return payload
            except Exception as exc:
                last_error = exc
        raise ProviderDataUnavailable(f"Free Eastmoney endpoint unavailable: {last_error}") from last_error

    @staticmethod
    def _parse_ts_code(ts_code: str) -> tuple[str, int]:
        code, _, exchange = ts_code.upper().partition(".")
        market = {"SZ": 0, "SH": 1}.get(exchange)
        if market is None:
            raise ProviderDataUnavailable(f"Only .SH and .SZ codes are supported by free provider: {ts_code}")
        return code.zfill(6), market

    @staticmethod
    def _limit_row(
        item: dict[str, Any],
        day_text: str,
        universe: pd.DataFrame,
        *,
        is_up: bool,
    ) -> dict[str, Any]:
        code = str(item.get("c", "")).zfill(6)
        market = int(item.get("m", 0))
        ts_code = f"{code}.{'SH' if market == 1 else 'SZ'}"
        name = item.get("n")
        if ts_code in universe.index:
            name = universe.loc[ts_code, "name"]
        return {
            "trade_date": day_text,
            "ts_code": ts_code,
            "name": name,
            "is_limit_up": bool(is_up),
            "is_limit_down": bool(not is_up),
            "pct_chg": float(item.get("zdp")) if item.get("zdp") is not None else pd.NA,
            "amount_rmb": float(item.get("amount")) if item.get("amount") is not None else pd.NA,
        }
