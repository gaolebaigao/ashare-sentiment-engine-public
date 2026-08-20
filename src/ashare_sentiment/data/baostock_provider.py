"""BaoStock provider for free, anonymous A-share historical data.

BaoStock exposes a socket-based Python API rather than an HTTP API.  The
adapter keeps that dependency optional and normalizes its string-valued result
sets into the same dataframe contract used by the other providers.

BaoStock does not expose a verified historical limit-price table in the fields
used here.  The limit-up/down panel therefore uses board-aware percentage bands
and marks the method explicitly.  It is useful for a free research run, but it
must not be described as exchange-exact limit-price history.
"""

from __future__ import annotations

import atexit
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import time
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .base import DataProvider, ProviderDataUnavailable, format_date, normalize_trade_date


class BaoStockProvider(DataProvider):
    """Free BaoStock adapter with current-universe and board-aware limits."""

    name = "baostock"

    _HISTORY_FIELDS = (
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
        "turn,tradestatus,pctChg,isST"
    )
    _SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
    _SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")

    def __init__(
        self,
        *,
        include_beijing: bool = False,
        exclude_st: bool = True,
        max_symbols: int | None = None,
        min_interval: float = 0.20,
        query_timeout_seconds: float = 15.0,
        progress_every: int = 100,
        workers: int = 1,
        api: Any | None = None,
    ):
        self.include_beijing = include_beijing
        self.exclude_st = exclude_st
        self.max_symbols = max_symbols
        self.min_interval = max(0.0, float(min_interval))
        self.query_timeout_seconds = max(0.0, float(query_timeout_seconds))
        self.progress_every = max(0, int(progress_every))
        self.workers = max(1, int(workers))
        self._last_query_at = 0.0
        self._universe: pd.DataFrame | None = None
        self._panel_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._logged_in = False

        if api is None:
            try:
                import baostock as bs
            except ImportError as exc:
                raise ProviderDataUnavailable(
                    "BaoStock is not installed. Install the data extra: "
                    "pip install -e '.[data]'"
                ) from exc
            self.api = bs
            # BaoStock prints login banners to stdout; keep the CLI output
            # focused on dataset and validation results.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                login_result = self.api.login()
            if getattr(login_result, "error_code", "1") != "0":
                raise ProviderDataUnavailable(
                    f"BaoStock login failed: {getattr(login_result, 'error_msg', login_result)}"
                )
            self._set_socket_timeout()
            self._logged_in = True
            atexit.register(self.close)
        else:
            self.api = api

    def close(self) -> None:
        if not self._logged_in:
            return
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.api.logout()
        finally:
            self._logged_in = False

    def get_stock_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str | None = None,
    ) -> pd.DataFrame:
        if not ts_code:
            raise ProviderDataUnavailable(
                "BaoStock stock history is symbol-oriented; --symbol is required for a single-stock request."
            )
        baostock_code = self._parse_ts_code(ts_code)
        return self._get_history(baostock_code, start_date, end_date, ts_code=ts_code)

    def get_index_daily(
        self,
        start_date: str | date,
        end_date: str | date,
        ts_code: str,
    ) -> pd.DataFrame:
        baostock_code = self._parse_ts_code(ts_code)
        return self._get_history(baostock_code, start_date, end_date, ts_code=ts_code)

    def get_market_breadth(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        cache_key = (format_date(start_date), format_date(end_date))
        if cache_key in self._panel_cache:
            return self._panel_cache[cache_key].copy()

        universe = self.get_stock_universe(end_date=end_date)
        if self.max_symbols is not None:
            universe = universe.head(self.max_symbols).copy()
        if universe.empty:
            raise ProviderDataUnavailable("BaoStock returned no eligible SSE/SZSE stock universe")

        if self.workers > 1 and self._logged_in:
            frames, failures = self._get_market_breadth_parallel(universe, start_date, end_date)
        else:
            frames = []
            failures = []
            for position, row in enumerate(universe.itertuples(index=False), start=1):
                try:
                    frame = self._get_history(
                        row.baostock_code,
                        start_date,
                        end_date,
                        ts_code=row.ts_code,
                    )
                    if not frame.empty:
                        frame["name"] = row.name
                        frame["exchange"] = row.exchange
                        frame["universe_count"] = len(universe)
                        frame["list_date"] = getattr(row, "list_date", pd.NA)
                        frame["delist_date"] = getattr(row, "delist_date", pd.NA)
                        frame["board"] = getattr(row, "board", pd.NA)
                        frames.append(frame)
                except ProviderDataUnavailable:
                    failures.append(row.ts_code)
                if self.progress_every and position % self.progress_every == 0:
                    print(f"BaoStock full-market download progress: {position:,}/{len(universe):,} symbols", flush=True)

        if not frames:
            raise ProviderDataUnavailable("BaoStock failed to retrieve any stock history")
        if failures and self.max_symbols is None:
            raise ProviderDataUnavailable(
                f"BaoStock failed for {len(failures)} of {len(universe)} symbols; "
                "refusing to compute breadth from an unexplained partial panel"
            )
        result = normalize_trade_date(pd.concat(frames, ignore_index=True))
        self._panel_cache[cache_key] = result.copy()
        return result

    def _get_market_breadth_parallel(
        self,
        universe: pd.DataFrame,
        start_date: str | date,
        end_date: str | date,
    ) -> tuple[list[pd.DataFrame], list[str]]:
        records = universe[["ts_code", "baostock_code", "name", "exchange", "list_date", "delist_date", "board"]].to_dict("records")
        chunks = [records[index:index + max(1, (len(records) + self.workers - 1) // self.workers)] for index in range(0, len(records), max(1, (len(records) + self.workers - 1) // self.workers))]
        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(
                    _fetch_baostock_chunk,
                    chunk,
                    self._iso_date(start_date),
                    self._iso_date(end_date),
                    self.min_interval,
                    self.query_timeout_seconds,
                    len(universe),
                )
                for chunk in chunks
            ]
            completed = 0
            for future in as_completed(futures):
                chunk_frames, chunk_failures = future.result()
                frames.extend(chunk_frames)
                failures.extend(chunk_failures)
                completed += 1
                print(f"BaoStock full-market download progress: {completed}/{len(chunks)} worker chunks completed", flush=True)
        return frames, failures

    def get_limit_up_down(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        panel = self.get_market_breadth(start_date, end_date)
        frame = panel.copy()
        frame["pct_chg"] = pd.to_numeric(frame.get("pct_chg"), errors="coerce")
        frame["is_st"] = pd.to_numeric(frame.get("is_st", 0), errors="coerce").fillna(0)
        if self.exclude_st:
            frame = frame[frame["is_st"].ne(1)].copy()
        frame["limit_pct"] = frame["ts_code"].map(self._limit_pct)
        # A 0.20 percentage-point tolerance accounts for rounded percentage
        # fields and one-tick price rounding. This is deliberately labelled as
        # approximate because BaoStock does not provide the exchange limit
        # price schedule in this query.
        tolerance = 0.20
        frame["is_limit_up"] = frame["pct_chg"].ge(frame["limit_pct"] - tolerance)
        frame["is_limit_down"] = frame["pct_chg"].le(-frame["limit_pct"] + tolerance)
        frame = frame[frame["is_limit_up"] | frame["is_limit_down"]].copy()
        if frame.empty:
            raise ProviderDataUnavailable(
                f"BaoStock produced no board-band limit records for {format_date(start_date)} to {format_date(end_date)}"
            )
        frame["limit_method"] = "baostock_board_band_approximation"
        return normalize_trade_date(frame)

    def get_margin_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable(
            "BaoStock free adapter does not expose the unified margin dataset required by v0.1."
        )

    def get_option_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        raise ProviderDataUnavailable("BaoStock free adapter does not provide the v0.1 option dataset.")

    def get_stock_universe(self, *, end_date: str | date | None = None) -> pd.DataFrame:
        if self._universe is not None:
            return self._universe.copy()
        requested = pd.Timestamp(format_date(end_date or date.today())).date()
        raw = pd.DataFrame()
        for offset in range(15):
            day = (requested - timedelta(days=offset)).isoformat()
            raw = self._query_all_stock(day)
            if not raw.empty:
                break
        if raw.empty:
            raise ProviderDataUnavailable("BaoStock returned no stock-universe rows for recent trading days")

        frame = raw.rename(columns={"code": "baostock_code", "code_name": "name"}).copy()
        frame["baostock_code"] = frame["baostock_code"].astype(str).str.lower()
        parsed = frame["baostock_code"].str.extract(r"^(?P<market>[^.]+)\.(?P<code>\d+)$")
        frame = frame.join(parsed)
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["market"] = frame["market"].astype(str).str.lower()
        frame["exchange"] = frame["market"].map({"sh": "SSE", "sz": "SZSE", "bj": "BSE"})
        allowed = (
            ((frame["market"] == "sh") & frame["code"].str.startswith(self._SH_PREFIXES))
            | ((frame["market"] == "sz") & frame["code"].str.startswith(self._SZ_PREFIXES))
        )
        if self.include_beijing:
            allowed |= (frame["market"] == "bj") & frame["code"].str.startswith(("4", "8", "92"))
        frame = frame[allowed].copy()
        trade_status = frame.get("tradeStatus", pd.Series(1, index=frame.index))
        frame["trade_status"] = pd.to_numeric(trade_status, errors="coerce").fillna(1)
        frame = frame[frame["trade_status"].eq(1)].copy()
        if self.exclude_st:
            frame = frame[~frame["name"].astype(str).str.contains(r"\*?ST", case=False, regex=True)]
        suffix = frame["market"].map({"sh": "SH", "sz": "SZ", "bj": "BJ"})
        frame["ts_code"] = frame["code"] + "." + suffix
        frame["list_date"] = pd.NA
        frame["delist_date"] = pd.NA
        if hasattr(self.api, "query_stock_basic"):
            try:
                basic = self._query_stock_basic()
                if not basic.empty:
                    basic = basic.rename(
                        columns={
                            "code": "baostock_code",
                            "code_name": "basic_name",
                            "ipoDate": "list_date_basic",
                            "outDate": "delist_date_basic",
                        }
                    )
                    keep = [column for column in ("baostock_code", "basic_name", "list_date_basic", "delist_date_basic") if column in basic.columns]
                    frame = frame.merge(basic[keep], on="baostock_code", how="left")
                    frame["name"] = frame["name"].fillna(frame.get("basic_name"))
                    frame["list_date"] = frame["list_date_basic"].combine_first(frame["list_date"])
                    frame["delist_date"] = frame["delist_date_basic"].combine_first(frame["delist_date"])
            except ProviderDataUnavailable:
                pass
        frame["board"] = frame["code"].map(self._board_name)
        frame = frame[["ts_code", "baostock_code", "code", "market", "exchange", "name", "list_date", "delist_date", "board"]]
        frame = frame.drop_duplicates("ts_code").reset_index(drop=True)
        if frame.empty:
            raise ProviderDataUnavailable("BaoStock returned no eligible SSE/SZSE stocks after filtering")
        self._universe = frame
        return frame.copy()

    def _get_history(
        self,
        baostock_code: str,
        start_date: str | date,
        end_date: str | date,
        *,
        ts_code: str,
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(2):
            self._throttle()
            try:
                result = self.api.query_history_k_data_plus(
                    baostock_code,
                    self._HISTORY_FIELDS,
                    start_date=self._iso_date(start_date),
                    end_date=self._iso_date(end_date),
                    frequency="d",
                    adjustflag="3",
                )
                raw = self._result_frame(result)
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0 and self._logged_in:
                    self._reconnect()
        else:
            raise ProviderDataUnavailable(
                f"BaoStock history request failed for {ts_code}: {last_error}"
            ) from last_error
        if raw.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", "open", "high", "low", "close"])
        frame = raw.rename(
            columns={
                "date": "trade_date",
                "preclose": "pre_close",
                "pctChg": "pct_chg",
                "vol": "volume",
                "turn": "turnover_rate",
                "isST": "is_st",
            }
        ).copy()
        frame["ts_code"] = ts_code
        for column in (
            "open", "high", "low", "close", "pre_close", "volume", "amount",
            "turnover_rate", "pct_chg", "is_st", "tradestatus",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "amount" in frame.columns:
            frame["amount_rmb"] = frame["amount"]
        if "pct_chg" not in frame.columns or frame["pct_chg"].isna().all():
            frame["pct_chg"] = frame["close"].div(frame["pre_close"]).sub(1.0).mul(100.0)
        return normalize_trade_date(frame)

    def _query_all_stock(self, day: str) -> pd.DataFrame:
        self._throttle()
        try:
            result = self.api.query_all_stock(day=day)
        except Exception as exc:
            raise ProviderDataUnavailable(f"BaoStock stock-universe request failed for {day}: {exc}") from exc
        return self._result_frame(result)

    def _query_stock_basic(self) -> pd.DataFrame:
        self._throttle()
        try:
            result = self.api.query_stock_basic()
        except Exception as exc:
            raise ProviderDataUnavailable(f"BaoStock stock-basic request failed: {exc}") from exc
        return self._result_frame(result)

    @staticmethod
    def _result_frame(result: Any) -> pd.DataFrame:
        error_code = str(getattr(result, "error_code", "1"))
        if error_code != "0":
            raise ProviderDataUnavailable(
                f"BaoStock query failed: {getattr(result, 'error_msg', result)}"
            )
        rows: list[list[Any]] = []
        while result.next():
            rows.append(result.get_row_data())
        return pd.DataFrame(rows, columns=list(getattr(result, "fields", [])))

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_query_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_query_at = time.monotonic()

    def _set_socket_timeout(self) -> None:
        """Bound BaoStock's otherwise-unbounded socket receive operation."""
        if self.query_timeout_seconds <= 0:
            return
        try:
            import baostock.common.context as context

            sock = getattr(context, "default_socket", None)
            if sock is not None:
                sock.settimeout(self.query_timeout_seconds)
        except Exception:
            # A mocked API or a future BaoStock client may not expose its socket.
            return

    def _reconnect(self) -> None:
        """Reset the BaoStock session after a timed-out or malformed response."""
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.api.logout()
                login_result = self.api.login()
            if getattr(login_result, "error_code", "1") != "0":
                raise ProviderDataUnavailable(
                    f"BaoStock reconnect failed: {getattr(login_result, 'error_msg', login_result)}"
                )
            self._logged_in = True
            self._set_socket_timeout()
        except ProviderDataUnavailable:
            raise
        except Exception as exc:
            raise ProviderDataUnavailable(f"BaoStock reconnect failed: {exc}") from exc

    @staticmethod
    def _iso_date(value: str | date) -> str:
        return pd.Timestamp(format_date(value)).strftime("%Y-%m-%d")

    @staticmethod
    def _parse_ts_code(ts_code: str) -> str:
        code, _, exchange = ts_code.upper().partition(".")
        market = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(exchange)
        if market is None:
            raise ProviderDataUnavailable(f"BaoStock expects .SH, .SZ or .BJ codes: {ts_code}")
        return f"{market}.{code.zfill(6)}"

    @staticmethod
    def _limit_pct(ts_code: str) -> float:
        code = str(ts_code).split(".", 1)[0].zfill(6)
        if code.startswith(("300", "301", "688", "689")):
            return 20.0
        if code.startswith(("4", "8", "92")):
            return 30.0
        return 10.0

    @staticmethod
    def _board_name(code: str) -> str:
        code = str(code).zfill(6)
        if code.startswith(("688", "689")):
            return "STAR"
        if code.startswith(("300", "301")):
            return "ChiNext"
        if code.startswith(("4", "8", "92")):
            return "BSE"
        return "MAIN"


def _fetch_baostock_chunk(
    records: list[dict[str, Any]],
    start_date: str,
    end_date: str,
    min_interval: float,
    query_timeout_seconds: float,
    universe_count: int,
) -> tuple[list[pd.DataFrame], list[str]]:
    """Fetch one independent BaoStock slice in a child process."""
    import baostock as bs

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        login_result = bs.login()
    if getattr(login_result, "error_code", "1") != "0":
        raise ProviderDataUnavailable(
            f"BaoStock worker login failed: {getattr(login_result, 'error_msg', login_result)}"
        )
    try:
        try:
            import baostock.common.context as context

            if getattr(context, "default_socket", None) is not None and query_timeout_seconds > 0:
                context.default_socket.settimeout(query_timeout_seconds)
        except Exception:
            pass
        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        last_query_at = 0.0
        for record in records:
            elapsed = time.monotonic() - last_query_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_query_at = time.monotonic()
            ts_code = str(record["ts_code"])
            try:
                result = bs.query_history_k_data_plus(
                    str(record["baostock_code"]),
                    BaoStockProvider._HISTORY_FIELDS,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3",
                )
                raw = BaoStockProvider._result_frame(result)
                if raw.empty:
                    failures.append(ts_code)
                    continue
                frame = _normalize_baostock_history(raw, ts_code)
                frame["name"] = record.get("name")
                frame["exchange"] = record.get("exchange")
                frame["universe_count"] = universe_count
                frame["list_date"] = record.get("list_date")
                frame["delist_date"] = record.get("delist_date")
                frame["board"] = record.get("board")
                frames.append(frame)
            except Exception:
                failures.append(ts_code)
        return frames, failures
    finally:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bs.logout()


def _normalize_baostock_history(raw: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    frame = raw.rename(
        columns={
            "date": "trade_date",
            "preclose": "pre_close",
            "pctChg": "pct_chg",
            "turn": "turnover_rate",
            "isST": "is_st",
        }
    ).copy()
    frame["ts_code"] = ts_code
    for column in (
        "open", "high", "low", "close", "pre_close", "volume", "amount",
        "turnover_rate", "pct_chg", "is_st", "tradestatus",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "amount" in frame.columns:
        frame["amount_rmb"] = frame["amount"]
    if "pct_chg" not in frame.columns or frame["pct_chg"].isna().all():
        frame["pct_chg"] = frame["close"].div(frame["pre_close"]).sub(1.0).mul(100.0)
    return normalize_trade_date(frame)
