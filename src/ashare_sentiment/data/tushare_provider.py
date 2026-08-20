"""Tushare Pro adapter with local-only credential configuration.

The provider keeps the token outside project files, serializes requests with
a 0.2 second minimum interval, and checkpoints long market downloads by
trading date so they can resume safely.
"""

from __future__ import annotations

from datetime import date, timedelta
import re
from pathlib import Path
import os
import tempfile
import time
from typing import Any, Callable

import pandas as pd

from .base import DataProvider, ProviderDataUnavailable, format_date, normalize_trade_date


DEFAULT_ENDPOINT = "https://api.tushare.pro"
STOCK_DAILY_FIELDS = (
    "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
)
DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,total_mv,circ_mv,limit_status"
)
STOCK_BASIC_FIELDS = (
    "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
)


class TushareProvider(DataProvider):
    """Tushare Pro adapter with conservative, resumable download behavior."""

    name = "tushare"

    def __init__(
        self,
        token: str | None = None,
        *,
        api: Any | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        min_interval: float = 0.20,
        retries: int = 3,
        checkpoint_root: str | Path = "data/raw/tushare_checkpoints",
        include_beijing: bool = False,
        exclude_st: bool = True,
        include_delisted: bool = False,
        max_symbols: int | None = None,
        progress_every: int = 50,
    ):
        self.endpoint = str(endpoint).rstrip("/")
        self.min_interval = max(float(min_interval), 0.20)
        self.retries = max(int(retries), 1)
        self.checkpoint_root = Path(checkpoint_root)
        self.include_beijing = include_beijing
        self.exclude_st = exclude_st
        self.include_delisted = include_delisted
        self.max_symbols = int(max_symbols) if max_symbols is not None else None
        self.progress_every = max(int(progress_every), 1)
        self._last_request_at: float | None = None
        self._live_api = api is None
        self._universe_frame: pd.DataFrame | None = None
        self._universe_cache: set[str] | None = None

        if api is None:
            if not token:
                raise ProviderDataUnavailable(
                    "Tushare requires TUSHARE_TOKEN. Put it in .env; never commit the token."
                )
            try:
                import tushare as ts
            except ImportError as exc:
                raise ProviderDataUnavailable(
                    "Tushare is not installed. Install the data extra: pip install -e '.[data]'"
                ) from exc
            # Pass the token directly so SDK initialization does not write a
            # global ~/tk.csv credential file outside the project workspace.
            api = ts.pro_api(token=token)
            # Apply the endpoint locally and never print or persist the key.
            setattr(api, "_DataApi_token", token)
            setattr(api, "_DataApi__http_url", self.endpoint)
        self.api = api

    def get_stock_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str | None = None,
    ) -> pd.DataFrame:
        frame = self._call(
            "daily",
            ts_code=ts_code or "",
            start_date=format_date(start_date),
            end_date=format_date(end_date),
            fields=STOCK_DAILY_FIELDS,
        )
        return self._normalize_daily(frame)

    def get_index_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str,
    ) -> pd.DataFrame:
        frame = self._call(
            "index_daily",
            ts_code=ts_code,
            start_date=format_date(start_date),
            end_date=format_date(end_date),
            fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )
        return self._normalize_daily(frame)

    def get_stock_universe(self, *, end_date: str | date | None = None) -> pd.DataFrame:
        """Return the current SSE/SZSE investable universe and listing dates."""
        if self._universe_frame is not None:
            return self._universe_frame.copy()
        if getattr(self.api, "stock_basic", None) is None and not self._live_api:
            # Lightweight fakes used by unit tests may intentionally expose
            # only one endpoint. Preserve the old adapter's no-filter behavior
            # for those injected APIs; a live account must expose stock_basic.
            self._universe_frame = pd.DataFrame(columns=["ts_code"])
            return self._universe_frame.copy()
        statuses = ["L"]
        if self.include_delisted:
            statuses = ["L", "D", "P"]
        frames: list[pd.DataFrame] = []
        for status in statuses:
            try:
                frame = self._call(
                    "stock_basic",
                    exchange="",
                    list_status=status,
                    fields=STOCK_BASIC_FIELDS,
                )
            except ProviderDataUnavailable:
                if status == "L":
                    raise
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
        if not frames:
            raise ProviderDataUnavailable("Tushare stock_basic returned no A-share symbols")
        frame = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="last")
        frame["ts_code"] = frame["ts_code"].astype("string")
        frame["exchange"] = frame.get("exchange", pd.Series(index=frame.index, dtype="string")).astype("string")
        code_mask = frame["ts_code"].str.endswith((".SH", ".SZ", ".BJ"), na=False)
        allowed = {"SSE", "SZSE"}
        if self.include_beijing:
            allowed.add("BSE")
        exchange_mask = frame["exchange"].isin(allowed)
        # Some compatible endpoints may omit exchange; suffixes are a safe
        # fallback for Shanghai/Shenzhen codes without admitting Northbound.
        if frame["exchange"].isna().all() or not exchange_mask.any():
            exchange_mask = frame["ts_code"].str.endswith((".SH", ".SZ"), na=False)
            if self.include_beijing:
                exchange_mask |= frame["ts_code"].str.endswith(".BJ", na=False)
        result = frame[code_mask & exchange_mask].copy()
        if self.exclude_st and "name" in result.columns:
            result = result[~result["name"].astype(str).str.contains(r"\*?ST", case=False, regex=True)].copy()
        if self.max_symbols is not None:
            result = result.sort_values("ts_code").head(max(self.max_symbols, 0)).copy()
        for column in ("list_date", "delist_date"):
            if column not in result.columns:
                result[column] = pd.NA
        result = result.reset_index(drop=True)
        self._universe_frame = result
        self._universe_cache = set(result["ts_code"].astype(str))
        return result.copy()

    def get_market_breadth(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Download the full daily breadth panel in resumable date batches."""
        start = format_date(start_date)
        end = format_date(end_date)
        universe = self.get_stock_universe()
        dates = self._trading_dates(start, end)
        if not dates:
            return pd.DataFrame()

        if self._live_api:
            return self._get_market_breadth_by_date(start, end, universe)

        frames: list[pd.DataFrame] = []
        for index, trade_date in enumerate(dates, start=1):
            checkpoint = self._checkpoint_path("market_breadth", trade_date)
            if self._live_api and checkpoint.exists():
                day_frame = self._read_checkpoint(checkpoint)
            else:
                day_frame = self._download_breadth_day(trade_date, universe)
                if self._live_api:
                    self._write_checkpoint(day_frame, checkpoint, columns=STOCK_DAILY_FIELDS.split(","))
            if not day_frame.empty:
                frames.append(day_frame)
            if self._live_api and (index == 1 or index % self.progress_every == 0 or index == len(dates)):
                print(f"[Tushare] market breadth {index}/{len(dates)} trade dates, rows={sum(len(item) for item in frames):,}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])

    def _get_market_breadth_by_date(
        self,
        start_date: str,
        end_date: str,
        universe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Download one trading day of all symbols per request.

        The compatible endpoint returns the complete market when ``daily`` is
        bounded by ``start_date``/``end_date``. A ``trade_date``-only request
        can be truncated to one row, so this method deliberately uses the
        date-range form even when the range is one day.
        """
        dates = self._trading_dates(start_date, end_date)
        frames: list[pd.DataFrame] = []
        for index, trade_date in enumerate(dates, start=1):
            checkpoint = self._checkpoint_path("market_breadth_by_date", trade_date)
            if checkpoint.exists():
                frame = self._read_checkpoint(checkpoint)
            else:
                daily = self._call(
                    "daily",
                    start_date=trade_date,
                    end_date=trade_date,
                    fields=STOCK_DAILY_FIELDS,
                )
                frame = self._normalize_daily(daily).rename(columns={"vol": "volume"})
                if not frame.empty and not universe.empty:
                    frame = frame.merge(
                        universe[[column for column in ("ts_code", "name", "exchange", "list_date", "delist_date") if column in universe.columns]],
                        on="ts_code",
                        how="inner",
                    )
                if not frame.empty:
                    frame["universe_count"] = len(universe)
                    frame["is_st"] = 0
                    if "amount" in frame.columns:
                        frame["amount_rmb"] = pd.to_numeric(frame["amount"], errors="coerce") * 1000.0
                self._write_checkpoint(frame, checkpoint, columns=STOCK_DAILY_FIELDS.split(","))
            frame = self._drop_suspended_rows(frame)
            if not frame.empty:
                frames.append(frame)
            if index == 1 or index % self.progress_every == 0 or index == len(dates):
                print(f"[Tushare] market breadth {index}/{len(dates)} trade dates, rows={sum(len(item) for item in frames):,}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])

    def _get_market_breadth_by_code(
        self,
        start_date: str,
        end_date: str,
        universe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Download daily OHLCV one symbol at a time, with resume checkpoints."""
        frames: list[pd.DataFrame] = []
        records = universe.to_dict("records")
        for index, record in enumerate(records, start=1):
            ts_code = str(record["ts_code"])
            checkpoint = self._checkpoint_path(
                "market_breadth_by_code",
                f"{ts_code.replace('.', '_')}_{start_date}_{end_date}",
            )
            if self._live_api and checkpoint.exists():
                frame = self._read_checkpoint(checkpoint)
            else:
                frame = self._call(
                    "daily",
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields=STOCK_DAILY_FIELDS,
                )
                frame = self._normalize_daily(frame).rename(columns={"vol": "volume"})
                if not frame.empty:
                    frame = frame.merge(
                        pd.DataFrame([record])[[column for column in ("ts_code", "name", "exchange", "list_date", "delist_date") if column in record]],
                        on="ts_code",
                        how="left",
                    )
                    frame["universe_count"] = len(universe)
                    frame["is_st"] = 0
                    if "amount" in frame.columns:
                        frame["amount_rmb"] = pd.to_numeric(frame["amount"], errors="coerce") * 1000.0
                if self._live_api:
                    self._write_checkpoint(frame, checkpoint, columns=STOCK_DAILY_FIELDS.split(","))
            frame = self._drop_suspended_rows(frame)
            if not frame.empty:
                frames.append(frame)
            if index == 1 or index % self.progress_every == 0 or index == len(records):
                print(f"[Tushare] market breadth {index}/{len(records)} symbols, rows={sum(len(item) for item in frames):,}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])

    @staticmethod
    def _drop_suspended_rows(frame: pd.DataFrame) -> pd.DataFrame:
        """Remove zero-activity rows whose OHLC fields are all zero."""
        if frame is None or frame.empty:
            return frame
        required = {"open", "high", "low", "close", "volume", "amount"}
        if not required.issubset(frame.columns):
            return frame
        numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce")
        suspended = (
            numeric[["open", "high", "low"]].eq(0).all(axis=1)
            & numeric["volume"].eq(0)
            & numeric["amount"].eq(0)
            & numeric["close"].gt(0)
        )
        return frame.loc[~suspended].copy()

    def get_limit_up_down(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Fetch real up and down pools, then fall back to daily limit prices."""
        errors: list[str] = []
        frames: list[pd.DataFrame] = []
        for limit_type, dataset in (("U", "limit_list_d_up"), ("D", "limit_list_d_down")):
            try:
                frame = self._get_limit_list_daily(
                    start_date, end_date, limit_type=limit_type, checkpoint_dataset=dataset
                )
                if not frame.empty:
                    frames.append(frame)
            except ProviderDataUnavailable as exc:
                errors.append(f"limit_list_d({limit_type}): {exc}")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sort_values(["trade_date", "ts_code"])
            bools = combined.groupby(["trade_date", "ts_code"], as_index=False)[
                ["is_limit_up", "is_limit_down"]
            ].max()
            metadata = combined.drop_duplicates(["trade_date", "ts_code"], keep="first").drop(
                columns=["is_limit_up", "is_limit_down"], errors="ignore"
            )
            return metadata.merge(bools, on=["trade_date", "ts_code"], how="outer")

        try:
            return self._get_limit_prices(start_date, end_date)
        except ProviderDataUnavailable as fallback_error:
            detail = f"{'; '.join(errors) or 'limit_list_d returned no rows'}; stk_limit: {fallback_error}"
            raise ProviderDataUnavailable(f"No usable Tushare limit dataset. {detail}") from fallback_error

    def get_margin_data(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        return normalize_trade_date(
            self._call(
                "margin",
                start_date=format_date(start_date),
                end_date=format_date(end_date),
            )
        )

    def get_option_data(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Return raw option daily data if the account has it enabled."""
        if getattr(self.api, "opt_daily", None) is None:
            raise ProviderDataUnavailable(
                "The selected Tushare account/API does not expose opt_daily; "
                "options are omitted rather than fabricated."
            )
        try:
            frame = self._call(
                "opt_daily",
                start_date=format_date(start_date),
                end_date=format_date(end_date),
            )
        except ProviderDataUnavailable as exc:
            raise ProviderDataUnavailable(f"Tushare option data unavailable: {exc}") from exc
        return normalize_trade_date(frame)

    def _download_breadth_day(self, trade_date: str, universe: pd.DataFrame) -> pd.DataFrame:
        daily = self._call("daily", trade_date=trade_date, fields=STOCK_DAILY_FIELDS)
        if daily.empty:
            return pd.DataFrame()
        basics = self._call("daily_basic", trade_date=trade_date, fields=DAILY_BASIC_FIELDS)
        frame = daily.merge(basics, on=["ts_code", "trade_date"], how="left", suffixes=("", "_basic"))
        if not universe.empty:
            frame = frame.merge(
                universe[[column for column in ("ts_code", "name", "exchange", "list_date", "delist_date") if column in universe.columns]],
                on="ts_code",
                how="inner",
            )
        frame = self._normalize_daily(frame).rename(columns={"vol": "volume"})
        if not universe.empty:
            frame["universe_count"] = len(universe)
        frame["is_st"] = frame.get("name", pd.Series(index=frame.index, dtype="string")).astype(str).str.contains(
            r"\*?ST", case=False, regex=True
        ).astype(int)
        if self.exclude_st:
            frame = frame[frame["is_st"].ne(1)].copy()
        if "amount" in frame.columns:
            # Tushare's daily amount is in thousand RMB; factors consume RMB.
            frame["amount_rmb"] = pd.to_numeric(frame["amount"], errors="coerce") * 1000.0
        return frame

    def _get_limit_list_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        *,
        limit_type: str | None = None,
        checkpoint_dataset: str = "limit_list_d",
    ) -> pd.DataFrame:
        dates = self._trading_dates(format_date(start_date), format_date(end_date))
        universe = self.get_stock_universe()
        frames: list[pd.DataFrame] = []
        for offset in range(0, len(dates), 10):
            chunk_dates = dates[offset:offset + 10]
            if not chunk_dates:
                continue
            missing_dates = [
                trade_date for trade_date in chunk_dates
                if (
                    not self._live_api
                    or not self._checkpoint_path(checkpoint_dataset, trade_date).exists()
                    or self._read_checkpoint(self._checkpoint_path(checkpoint_dataset, trade_date)).empty
                )
            ]
            if missing_dates:
                try:
                    params: dict[str, Any] = {
                        "start_date": missing_dates[0],
                        "end_date": missing_dates[-1],
                        "fields": "trade_date,ts_code,close,up_stat,down_stat",
                    }
                    if limit_type:
                        params["limit_type"] = limit_type
                    raw = self._call("limit_list_d", **params)
                except ProviderDataUnavailable:
                    # Some Tushare-compatible deployments expose only the
                    # trade_date form. Preserve compatibility with that mode.
                    raw_parts = []
                    for trade_date in missing_dates:
                        params = {"trade_date": trade_date, "fields": "trade_date,ts_code,close,up_stat,down_stat"}
                        if limit_type:
                            params["limit_type"] = limit_type
                        raw_parts.append(self._call("limit_list_d", **params))
                    raw = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
                raw = normalize_trade_date(raw)
                for trade_date in missing_dates:
                    checkpoint = self._checkpoint_path(checkpoint_dataset, trade_date)
                    day = raw[raw.get("trade_date", pd.Series(dtype="datetime64[ns]")).eq(pd.Timestamp(trade_date))].copy()
                    if not day.empty and not universe.empty:
                        day = day.merge(universe[["ts_code", "name"]], on="ts_code", how="inner")
                    day = self._normalize_limit_frame(day, assume_limit_type=limit_type)
                    if not day.empty:
                        source = "limit_list_d"
                        day["limit_up_source"] = source
                        day["limit_down_source"] = source
                        day["limit_up_status"] = "REAL_ROWS" if day["is_limit_up"].any() else "REAL_ZERO"
                        day["limit_down_status"] = "REAL_ROWS" if day["is_limit_down"].any() else "REAL_ZERO"
                    if self._live_api:
                        self._write_checkpoint(
                            day,
                            checkpoint,
                            columns=[
                                "trade_date", "ts_code", "is_limit_up", "is_limit_down",
                                "limit_method", "limit_up_source", "limit_down_source",
                                "limit_up_status", "limit_down_status",
                            ],
                        )

            for trade_date in chunk_dates:
                checkpoint = self._checkpoint_path(checkpoint_dataset, trade_date)
                if self._live_api and checkpoint.exists():
                    day = self._normalize_limit_frame(self._read_checkpoint(checkpoint), assume_limit_type=limit_type)
                else:
                    continue
                if not day.empty:
                    frames.append(day)
            completed = min(offset + len(chunk_dates), len(dates))
            if self._live_api and (completed == len(dates) or completed % self.progress_every == 0):
                print(f"[Tushare] limit_list_d {completed}/{len(dates)} trade dates, rows={sum(len(item) for item in frames):,}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

    def _get_limit_prices(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        limits = self._call(
            "stk_limit",
            start_date=format_date(start_date),
            end_date=format_date(end_date),
            fields="trade_date,ts_code,pre_close,up_limit,down_limit",
        )
        if limits.empty:
            return limits
        daily = self._call(
            "daily",
            start_date=format_date(start_date),
            end_date=format_date(end_date),
            fields="ts_code,trade_date,close,pct_chg",
        )
        frame = limits.merge(daily, on=["ts_code", "trade_date"], how="left")
        frame = self._filter_universe(frame)
        frame["is_limit_up"] = pd.to_numeric(frame["close"], errors="coerce").ge(
            pd.to_numeric(frame["up_limit"], errors="coerce") - 1e-8
        )
        frame["is_limit_down"] = pd.to_numeric(frame["close"], errors="coerce").le(
            pd.to_numeric(frame["down_limit"], errors="coerce") + 1e-8
        )
        frame["limit_method"] = "stk_limit"
        frame["limit_up_source"] = "stk_limit"
        frame["limit_down_source"] = "stk_limit"
        frame["limit_up_status"] = frame["is_limit_up"].map({True: "REAL_ROWS", False: "REAL_ZERO"})
        frame["limit_down_status"] = frame["is_limit_down"].map({True: "REAL_ROWS", False: "REAL_ZERO"})
        return normalize_trade_date(frame)

    def _trading_dates(self, start_date: str, end_date: str) -> list[str]:
        method = getattr(self.api, "trade_cal", None)
        if method is None and not self._live_api:
            return [item.strftime("%Y%m%d") for item in pd.date_range(start_date, end_date, freq="B")]
        # The compatible service caps one trade_cal response. Querying in
        # annual chunks prevents a multi-year request from silently omitting
        # the earliest dates.
        cursor = date.fromisoformat(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
        final = date.fromisoformat(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}")
        collected: set[str] = set()
        while cursor <= final:
            chunk_end = min(cursor + timedelta(days=364), final)
            calendar = self._call(
                "trade_cal",
                exchange="",
                start_date=cursor.strftime("%Y%m%d"),
                end_date=chunk_end.strftime("%Y%m%d"),
                fields="cal_date,is_open",
            )
            if not calendar.empty:
                open_mask = pd.to_numeric(calendar.get("is_open"), errors="coerce").eq(1)
                dates = calendar.loc[open_mask, "cal_date"].astype(str).str.replace(r"[-/]", "", regex=True)
                collected.update(item for item in dates.tolist() if re.fullmatch(r"\d{8}", item))
            cursor = chunk_end + timedelta(days=1)
        return sorted(collected)

    def _filter_universe(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "ts_code" not in frame.columns:
            return frame
        universe = self.get_stock_universe()
        if universe.empty:
            return frame
        return frame[frame["ts_code"].isin(set(universe["ts_code"].astype(str)))].copy()

    @staticmethod
    def _normalize_daily(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame() if frame is None else normalize_trade_date(frame)
        result = normalize_trade_date(frame).rename(columns={"vol": "volume"})
        if "amount" in result.columns:
            result["amount_rmb"] = pd.to_numeric(result["amount"], errors="coerce") * 1000.0
        return result

    @staticmethod
    def _normalize_limit_frame(frame: pd.DataFrame, assume_limit_type: str | None = None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", "is_limit_up", "is_limit_down"])
        result = frame.copy()
        if "close" in result.columns:
            close = pd.to_numeric(result["close"], errors="coerce")
            result = result.loc[close.gt(0)].copy()
            if result.empty:
                return pd.DataFrame(columns=["trade_date", "ts_code", "is_limit_up", "is_limit_down"])
        has_direction_fields = any(
            column in result.columns and result[column].astype("string").str.strip().ne("").any()
            for column in ("limit_type", "up_stat", "down_stat")
        )
        limit_type = result.get("limit_type", pd.Series("", index=result.index, dtype="string")).astype("string").str.upper().fillna("")
        up_stat = result.get("up_stat", pd.Series("", index=result.index, dtype="string")).astype("string").fillna("").str.strip()
        down_stat = result.get("down_stat", pd.Series("", index=result.index, dtype="string")).astype("string").fillna("").str.strip()
        empty_stat = {"", "NAN", "NONE", "NA", "N/A", "0", "0.0"}
        result["is_limit_up"] = limit_type.eq("U") | ~up_stat.str.upper().isin(empty_stat)
        result["is_limit_down"] = limit_type.eq("D") | ~down_stat.str.upper().isin(empty_stat)
        if assume_limit_type and not has_direction_fields:
            if assume_limit_type.upper() == "U":
                result["is_limit_up"] = True
            elif assume_limit_type.upper() == "D":
                result["is_limit_down"] = True
        if "close" in result.columns and "up_limit" in result.columns:
            result["is_limit_up"] |= pd.to_numeric(result["close"], errors="coerce").ge(
                pd.to_numeric(result["up_limit"], errors="coerce") - 1e-8
            )
        if "close" in result.columns and "down_limit" in result.columns:
            result["is_limit_down"] |= pd.to_numeric(result["close"], errors="coerce").le(
                pd.to_numeric(result["down_limit"], errors="coerce") + 1e-8
            )
        result["limit_method"] = "limit_list_d"
        result["is_limit_up"] = result["is_limit_up"].fillna(False).astype(bool)
        result["is_limit_down"] = result["is_limit_down"].fillna(False).astype(bool)
        return normalize_trade_date(result)

    def _checkpoint_path(self, dataset: str, trade_date: str) -> Path:
        root = self.checkpoint_root / dataset
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{trade_date}.csv"

    @staticmethod
    def _read_checkpoint(path: Path) -> pd.DataFrame:
        try:
            return normalize_trade_date(pd.read_csv(path))
        except Exception as exc:
            raise ProviderDataUnavailable(f"Could not read Tushare checkpoint {path}: {exc}") from exc

    @staticmethod
    def _write_checkpoint(frame: pd.DataFrame, path: Path, *, columns: list[str]) -> None:
        output = frame.copy()
        if output.empty:
            output = pd.DataFrame(columns=columns)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temp = Path(temp_name)
        try:
            output.to_csv(temp, index=False, encoding="utf-8-sig")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    def _call(self, endpoint: str, **kwargs: Any) -> pd.DataFrame:
        method: Callable[..., Any] | None = getattr(self.api, endpoint, None)
        if method is None:
            raise ProviderDataUnavailable(f"Tushare API endpoint is not available: {endpoint}")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                self._wait_before_request()
                result = method(**kwargs)
                if result is None:
                    return pd.DataFrame()
                if not isinstance(result, pd.DataFrame):
                    return pd.DataFrame(result)
                return result.copy()
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.retries:
                    break
                if self._live_api:
                    time.sleep(2.0 * (attempt + 1))
        raise ProviderDataUnavailable(f"Tushare {endpoint} request failed: {last_error}") from last_error

    def _wait_before_request(self) -> None:
        if not self._live_api:
            return
        now = time.monotonic()
        if self._last_request_at is not None:
            remaining = self.min_interval - (now - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()
