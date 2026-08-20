"""Non-persistent intraday market-temperature snapshots."""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, time as clock_time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ..advisory.engine import build_advisory_frame
from ..data.cache import ParquetCache
from ..data.eastmoney_provider import EastMoneyProvider
from ..data.base import ProviderDataUnavailable
from ..data.tencent_provider import TencentProvider
from ..regime import apply_state_machine, build_regime_indicators
from ..scoring.market_temperature import calculate_intraday_market_temperature
from .repository import AdvisoryRepository
from .viewmodels import DailyAdvisoryViewModel


class IntradaySnapshotService:
    """Build a full-market intraday estimate without changing EOD artifacts."""

    def __init__(self, config: dict[str, Any], repository: AdvisoryRepository):
        self.config = config
        self.repository = repository
        data = config.get("data", {})
        realtime = config.get("realtime", {})
        self.enabled = bool(realtime.get("enabled", True))
        self.timezone = ZoneInfo(str(config.get("project", {}).get("timezone", "Asia/Shanghai")))
        self.ttl_seconds = max(15, int(realtime.get("refresh_seconds", 60)))
        self.max_staleness_seconds = max(30, int(realtime.get("max_staleness_seconds", 180)))
        requested_sources = str(realtime.get("source", "tencent/eastmoney")).lower().replace(",", "/")
        self.source_priority = tuple(
            source for source in requested_sources.split("/") if source in {"tencent", "eastmoney"}
        ) or ("tencent", "eastmoney")
        percentile = config.get("scoring", {}).get("percentile", {})
        self.history_days = int(percentile.get("lookback", 756)) + 90
        windows = config.get("factors", {}).get("windows", {})
        self.factor_history_days = max(90, max((int(value) for value in windows.values()), default=60) + 30)
        realtime_timeout = float(realtime.get("request_timeout_seconds", 5))
        realtime_retries = int(realtime.get("request_retries", 0))
        self.provider = EastMoneyProvider(
            cache_root=data.get("http_cache_root", "data/raw/http"),
            timeout=realtime_timeout,
            min_interval=float(data.get("request_min_interval_seconds", 0.20)),
            retries=realtime_retries,
            include_beijing=bool(data.get("include_beijing", False)),
            exclude_st=bool(data.get("exclude_st", True)),
        )
        self.fallback_provider = TencentProvider(
            cache_root=data.get("http_cache_root", "data/raw/http"),
            timeout=realtime_timeout,
            min_interval=float(data.get("request_min_interval_seconds", 0.20)),
            retries=realtime_retries,
        )
        self.raw_cache = ParquetCache(data.get("cache_root", "data/cache"))
        self._lock = threading.Lock()
        self._cached: dict[str, Any] | None = None
        self._cached_monotonic = 0.0
        self._stock_history: pd.DataFrame | None = None
        self._limit_history: pd.DataFrame | None = None
        self._index_history: dict[str, pd.DataFrame] | None = None
        self._score_history: pd.DataFrame | None = None

    def available_now(self, now: datetime | None = None) -> bool:
        current = now.astimezone(self.timezone) if now else datetime.now(self.timezone)
        return self.enabled and current.weekday() < 5 and current.time() >= clock_time(9, 30)

    def snapshot(self, *, force: bool = False) -> dict[str, Any]:
        if not self.available_now():
            raise RuntimeError("盘中快照仅在交易日 09:30 后提供")
        with self._lock:
            age = time.monotonic() - self._cached_monotonic
            if self._cached is not None and not force and age < self.ttl_seconds:
                return self._cached
            payload = self._build_snapshot()
            self._cached = payload
            self._cached_monotonic = time.monotonic()
            return payload

    def _build_snapshot(self) -> dict[str, Any]:
        now = datetime.now(self.timezone)
        today = now.date()
        market_status, session_fraction = self._market_clock(now)
        stock_history = self._history_stock(today)
        history_dates = pd.to_datetime(stock_history["trade_date"], errors="coerce")
        latest_history_date = history_dates.max()
        codes = stock_history.loc[history_dates.eq(latest_history_date), "ts_code"].dropna().astype(str).drop_duplicates().tolist()
        snapshot, source = self._fetch_stock_snapshot(codes, today)
        if snapshot.empty:
            raise RuntimeError("实时行情源暂未返回有效的全市场快照")
        quote_as_of = self._validate_quote_freshness(snapshot, now, market_status, source)
        # Accumulated turnover is projected to a full-session equivalent so
        # an early-morning observation is comparable with historical closes.
        snapshot["amount_rmb_actual"] = snapshot["amount_rmb"]
        if market_status in {"OPEN", "LUNCH"}:
            snapshot["amount_rmb"] = snapshot["amount_rmb"] / max(session_fraction, 0.08)
        stock_panel = pd.concat([stock_history, snapshot], ignore_index=True, sort=False)

        try:
            live_limits = self.provider.get_realtime_limit_up_down(trade_date=today)
        except ProviderDataUnavailable:
            live_limits = self._infer_realtime_limits(snapshot, today)
        limit_panel = self._history_limits(today)
        if not live_limits.empty:
            limit_panel = pd.concat([limit_panel, live_limits], ignore_index=True, sort=False)

        index_frames: dict[str, pd.DataFrame] = {}
        aliases = {"沪深300": "hs300", "中证1000": "csi1000", "创业板指": "chinext"}
        for benchmark in self.config.get("benchmarks", [])[:3]:
            code = benchmark["ts_code"]
            alias = aliases.get(benchmark.get("name"), code.replace(".", "_").lower())
            history = self._history_indices().get(alias, pd.DataFrame()).copy()
            history = history[pd.to_datetime(history.get("trade_date"), errors="coerce").dt.date.ne(today)] if not history.empty else history
            try:
                live = self.fallback_provider.get_realtime_snapshot([code], trade_date=today)
            except ProviderDataUnavailable:
                live = self.provider.get_realtime_index(code, trade_date=today)
            index_frames[alias] = pd.concat([history, live], ignore_index=True, sort=False)

        historical_scores = self._history_scores(today)
        current_score_frame = calculate_intraday_market_temperature(
            stock_panel,
            limit_panel,
            index_frames,
            historical_scores,
            self.config,
        )
        scored = pd.concat([historical_scores, current_score_frame], ignore_index=True, sort=False)
        scored = scored.sort_values("trade_date").reset_index(drop=True)
        indicators = build_regime_indicators(scored, self.config)
        regime = apply_state_machine(indicators, self.config)
        regime["date"] = pd.to_datetime(regime["trade_date"], errors="coerce").dt.normalize()
        advisory = build_advisory_frame(regime, config=self.config)
        current_advisory = advisory[pd.to_datetime(advisory["date"]).dt.date.eq(today)]
        current_state = regime[regime["trade_date"].dt.date.eq(today)]
        if current_advisory.empty or current_state.empty:
            raise RuntimeError("实时快照已取得，但未能生成当日情绪值")
        row = current_advisory.iloc[-1].to_dict()
        row.update(current_state.iloc[-1].to_dict())
        row["date"] = today.isoformat()
        daily = DailyAdvisoryViewModel.from_row(row)
        current_score = current_state.iloc[-1]
        return {
            "daily": daily,
            "point": {
                "date": today.isoformat(),
                "market_temperature": self._number(current_score.get("market_temperature", current_score.get("raw_temperature"))),
                "smoothed_temperature": self._number(current_score.get("smoothed_temperature")),
                "advisory_signal": daily.advisory_signal,
            },
            "is_intraday": market_status != "CLOSED",
            "market_status": market_status,
            "as_of": quote_as_of.isoformat(timespec="seconds"),
            "calculated_at": now.isoformat(timespec="seconds"),
            "refresh_seconds": self.ttl_seconds,
            "universe_count": int(snapshot["ts_code"].nunique()),
            "session_fraction": session_fraction,
            "source": source,
        }

    def _fetch_stock_snapshot(self, codes: list[str], today: date) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []
        for source in self.source_priority:
            try:
                if source == "tencent":
                    return self.fallback_provider.get_realtime_snapshot(codes, trade_date=today), "tencent-realtime"
                return self.provider.get_realtime_snapshot(trade_date=today), "eastmoney-realtime"
            except ProviderDataUnavailable as exc:
                errors.append(f"{source}: {self._compact_error(exc)}")
        raise ProviderDataUnavailable("实时行情源均不可用；" + "；".join(errors))

    def _validate_quote_freshness(
        self, snapshot: pd.DataFrame, now: datetime, market_status: str, source: str
    ) -> datetime:
        raw_stamps = snapshot["quote_time"] if "quote_time" in snapshot else pd.Series(dtype="object")
        stamps = pd.to_datetime(raw_stamps, errors="coerce").dropna()
        if stamps.empty:
            # Eastmoney's list endpoint may omit exchange timestamps. Its
            # uncached response is still current, but it cannot claim a more
            # precise market timestamp than the fetch time.
            if source == "eastmoney-realtime":
                return now
            raise ProviderDataUnavailable("实时行情缺少可校验的交易所时间戳")
        quote_as_of = stamps.max().to_pydatetime().replace(tzinfo=self.timezone)
        if quote_as_of.date() != now.date():
            raise ProviderDataUnavailable(f"行情源最新数据仍停留在 {quote_as_of:%Y-%m-%d %H:%M:%S}")
        if market_status == "OPEN" and now - quote_as_of > timedelta(seconds=self.max_staleness_seconds):
            raise ProviderDataUnavailable(
                f"行情已延迟 {int((now - quote_as_of).total_seconds())} 秒，超过 {self.max_staleness_seconds} 秒上限"
            )
        if market_status == "LUNCH" and quote_as_of.time() < clock_time(11, 25):
            raise ProviderDataUnavailable(f"午间行情时间戳异常：{quote_as_of:%H:%M:%S}")
        if market_status == "CLOSED" and quote_as_of.time() < clock_time(14, 55):
            raise ProviderDataUnavailable(f"收盘行情时间戳异常：{quote_as_of:%H:%M:%S}")
        return quote_as_of

    @staticmethod
    def _infer_realtime_limits(snapshot: pd.DataFrame, today: date) -> pd.DataFrame:
        """Last-resort limit flags when the dedicated pool endpoint is down."""
        frame = snapshot[["ts_code", "close", "pre_close"]].copy()
        code = frame["ts_code"].astype(str).str[:6]
        rate = pd.Series(0.10, index=frame.index)
        rate.loc[code.str.startswith(("300", "301", "688"))] = 0.20
        pre_close = pd.to_numeric(frame["pre_close"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        upper = (pre_close * (1 + rate)).round(2)
        lower = (pre_close * (1 - rate)).round(2)
        frame["trade_date"] = pd.Timestamp(today)
        frame["is_limit_up"] = close.ge(upper)
        frame["is_limit_down"] = close.le(lower)
        frame = frame[frame["is_limit_up"] | frame["is_limit_down"]].copy()
        frame["limit_up_source"] = "realtime-price-inferred"
        frame["limit_down_source"] = "realtime-price-inferred"
        frame["limit_up_status"] = "APPROX_LIST"
        frame["limit_down_status"] = "APPROX_LIST"
        return frame

    @staticmethod
    def _compact_error(exc: Exception, limit: int = 240) -> str:
        message = " ".join(str(exc).split())
        return message if len(message) <= limit else message[: limit - 1] + "…"

    def _history_stock(self, today: date) -> pd.DataFrame:
        if self._stock_history is None:
            self._stock_history = self.repository.processed.load("full_market_daily")
        frame = self._recent(self._stock_history, self.factor_history_days)
        dates = pd.to_datetime(frame["trade_date"], errors="coerce")
        return frame[dates.dt.date.ne(today)].copy()

    def _history_limits(self, today: date) -> pd.DataFrame:
        if self._limit_history is None:
            self._limit_history = self.raw_cache.load("limit_up_down")
        frame = self._recent(self._limit_history, self.factor_history_days)
        dates = pd.to_datetime(frame["trade_date"], errors="coerce")
        return frame[dates.dt.date.ne(today)].copy()

    def _history_indices(self) -> dict[str, pd.DataFrame]:
        if self._index_history is None:
            aliases = {"沪深300": "hs300", "中证1000": "csi1000", "创业板指": "chinext"}
            self._index_history = {}
            for benchmark in self.config.get("benchmarks", [])[:3]:
                code = benchmark["ts_code"]
                alias = aliases.get(benchmark.get("name"), code.replace(".", "_").lower())
                self._index_history[alias] = self._recent(self.raw_cache.load(f"index_{code}"), self.factor_history_days)
        return self._index_history

    def _history_scores(self, today: date) -> pd.DataFrame:
        if self._score_history is None:
            self._score_history = self.repository.processed.load("market_sentiment_daily")
            self._score_history["trade_date"] = pd.to_datetime(self._score_history["trade_date"], errors="coerce").dt.normalize()
        dates = pd.to_datetime(self._score_history["trade_date"], errors="coerce")
        return self._score_history[dates.dt.date.ne(today)].copy()

    def _recent(self, frame: pd.DataFrame, days: int | None = None) -> pd.DataFrame:
        days = days or self.history_days
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna().drop_duplicates().sort_values()
        if len(dates) <= days:
            return frame
        cutoff = dates.iloc[-days]
        return frame[pd.to_datetime(frame["trade_date"], errors="coerce").ge(cutoff)].copy()

    @staticmethod
    def _market_clock(now: datetime) -> tuple[str, float]:
        current = now.time()
        if current < clock_time(9, 30):
            return "PREOPEN", 0.0
        if current <= clock_time(11, 30):
            minutes = (now.hour * 60 + now.minute) - (9 * 60 + 30)
            return "OPEN", max(0.0, min(1.0, minutes / 240.0))
        if current < clock_time(13, 0):
            return "LUNCH", 0.5
        if current <= clock_time(15, 0):
            minutes = 120 + (now.hour * 60 + now.minute) - (13 * 60)
            return "OPEN", max(0.5, min(1.0, minutes / 240.0))
        return "CLOSED", 1.0

    @staticmethod
    def _number(value: Any) -> float | None:
        result = pd.to_numeric(value, errors="coerce")
        return None if pd.isna(result) else float(result)
