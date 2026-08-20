"""Production data-quality gates and daily integrity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


class ProductionDataQualityError(RuntimeError):
    """Base class for blocking production data-quality failures."""


class InsufficientMarketCoverage(ProductionDataQualityError):
    """Raised when a score would be based on too little of the A-share market."""


class UnavailableMarketFactor(ProductionDataQualityError):
    """Raised when a required production factor has no trustworthy input."""


@dataclass(frozen=True)
class CoverageAssessment:
    """Summary of the latest stock-panel day and its daily audit table."""

    known_stocks: int
    minimum_market_universe: int
    latest_eligible_count: int
    minimum_coverage_ratio: float
    latest_coverage_ratio: float
    valid_price_count: int
    valid_amount_count: int
    date_count: int
    expected_eligible_count: int = 0
    observed_universe: int = 0
    daily_coverage: pd.DataFrame | None = None
    warnings: tuple[str, ...] = ()

    @property
    def sufficient_universe(self) -> bool:
        return self.expected_eligible_count >= self.minimum_market_universe

    @property
    def sufficient_coverage(self) -> bool:
        return self.latest_coverage_ratio >= self.minimum_coverage_ratio

    @property
    def valid(self) -> bool:
        return self.sufficient_universe and self.sufficient_coverage


def _quality_thresholds(config: Mapping[str, Any] | None) -> tuple[int, float]:
    quality = (config or {}).get("data_quality", {})
    minimum_universe = int(
        quality.get("minimum_expected_universe", quality.get("minimum_market_universe", 3000))
    )
    minimum_ratio = float(
        quality.get("minimum_market_coverage_ratio", quality.get("minimum_coverage_ratio", 0.90))
    )
    return minimum_universe, minimum_ratio


def assess_market_coverage(
    stock_panel: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> CoverageAssessment:
    """Assess the latest day and retain the full daily coverage evidence."""
    minimum_universe, minimum_ratio = _quality_thresholds(config)
    if stock_panel is None or stock_panel.empty:
        raise InsufficientMarketCoverage(
            f"Market panel is empty; required at least {minimum_universe:,} expected eligible stocks."
        )
    daily = build_market_coverage_daily(stock_panel, config=config)
    if daily.empty:
        raise InsufficientMarketCoverage("Market panel has no valid trade dates.")
    latest = daily.sort_values("trade_date").iloc[-1]
    latest_expected = int(latest["expected_eligible_count"])
    latest_observed = int(latest["observed_universe"])
    latest_ratio = float(latest["coverage_ratio"]) if pd.notna(latest["coverage_ratio"]) else 0.0
    warnings: list[str] = []
    if "list_date" not in stock_panel.columns or stock_panel["list_date"].isna().all():
        warnings.append("HISTORICAL_LIST_DATE_INCOMPLETE")
    if "delist_date" not in stock_panel.columns or stock_panel["delist_date"].isna().all():
        warnings.append("HISTORICAL_DELIST_DATE_INCOMPLETE")
    if "is_st" not in stock_panel.columns:
        warnings.append("HISTORICAL_ST_STATUS_INCOMPLETE")
    if latest_expected < minimum_universe:
        warnings.append(f"EXPECTED_UNIVERSE_BELOW_MINIMUM_{latest_expected}")
    if latest_ratio < minimum_ratio:
        warnings.append(f"MARKET_COVERAGE_BELOW_MINIMUM_{latest_ratio:.3f}")
    return CoverageAssessment(
        known_stocks=latest_expected,
        minimum_market_universe=minimum_universe,
        latest_eligible_count=latest_observed,
        minimum_coverage_ratio=minimum_ratio,
        latest_coverage_ratio=latest_ratio,
        valid_price_count=int(latest["valid_price_count"]),
        valid_amount_count=int(latest["valid_amount_count"]),
        date_count=len(daily),
        expected_eligible_count=latest_expected,
        observed_universe=latest_observed,
        daily_coverage=daily,
        warnings=tuple(warnings),
    )


class ProductionDataQualityGate:
    """Block ordinary scores when the selected/latest day is incomplete."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config

    def assess(self, stock_panel: pd.DataFrame) -> CoverageAssessment:
        return assess_market_coverage(stock_panel, self.config)

    def validate(
        self,
        stock_panel: pd.DataFrame,
        *,
        allow_partial_data: bool = False,
    ) -> CoverageAssessment:
        assessment = self.assess(stock_panel)
        if assessment.valid or allow_partial_data:
            return assessment
        reason = "market coverage too low"
        if not assessment.sufficient_universe:
            reason = "expected eligible stock universe below minimum"
        elif not assessment.sufficient_coverage:
            reason = "daily observed/expected market coverage below minimum"
        raise InsufficientMarketCoverage(
            "MARKET TEMPERATURE INVALID\n"
            f"Reason: {reason}\n"
            f"Observed universe: {assessment.observed_universe:,}\n"
            f"Expected universe: {assessment.expected_eligible_count:,}\n"
            f"Required expected universe: {assessment.minimum_market_universe:,}\n"
            f"Coverage: {assessment.latest_coverage_ratio:.1%}\n"
            f"Required coverage: {assessment.minimum_coverage_ratio:.1%}\n"
            "Hint: run a full-market update; --allow-partial-data is research-only."
        )


def build_market_coverage_daily(
    stock_panel: pd.DataFrame,
    *,
    breadth_coverage: pd.Series | None = None,
    profit_effect_coverage: pd.Series | None = None,
    liquidity_coverage: pd.Series | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the auditable expected-vs-observed daily coverage table."""
    config = config or {}
    if stock_panel is None or stock_panel.empty:
        return pd.DataFrame()
    frame = stock_panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype("string")
    frame = frame.dropna(subset=["trade_date", "ts_code"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "pre_close" in frame:
        frame["pre_close"] = pd.to_numeric(frame["pre_close"], errors="coerce")
    amount = frame.get("amount_rmb", frame.get("amount"))
    frame["_amount"] = pd.to_numeric(amount, errors="coerce")
    frame = frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
    frame["_price_valid"] = _valid_price_mask(frame)
    frame["_amount_valid"] = frame["_amount"].ge(0) & frame["_amount"].notna()
    grouped = frame.groupby("trade_date", sort=True)
    result = grouped.agg(
        observed_universe=("_price_valid", "sum"),
        observed_symbol_count=("ts_code", "nunique"),
        valid_price_count=("_price_valid", "sum"),
        valid_amount_count=("_amount_valid", "sum"),
    ).reset_index()
    result = result.merge(build_expected_universe_daily(frame), on="trade_date", how="left")
    result["known_stocks"] = result["expected_eligible_count"]
    result["eligible_count"] = result["observed_universe"]
    result["missing_close_count"] = result["expected_eligible_count"] - result["valid_price_count"]
    result["missing_amount_count"] = result["expected_eligible_count"] - result["valid_amount_count"]
    result["coverage_ratio"] = result["observed_universe"].div(
        result["expected_eligible_count"].where(result["expected_eligible_count"].gt(0))
    )
    result["breadth_coverage"] = _align_series(result["trade_date"], breadth_coverage, result["coverage_ratio"])
    result["profit_effect_coverage"] = _align_series(result["trade_date"], profit_effect_coverage, result["coverage_ratio"])
    result["liquidity_coverage"] = _align_series(
        result["trade_date"], liquidity_coverage,
        result["valid_amount_count"].div(result["expected_eligible_count"].where(result["expected_eligible_count"].gt(0))),
    )
    minimum_universe, minimum_ratio = _quality_thresholds(config)
    result["expected_universe_below_minimum"] = result["expected_eligible_count"].lt(minimum_universe)
    result["data_quality"] = result.apply(
        lambda row: "VALID" if row["coverage_ratio"] >= minimum_ratio else "INVALID",
        axis=1,
    )
    result["warnings"] = result.apply(
        lambda row: _coverage_warning(row, minimum_ratio, minimum_universe), axis=1
    )
    return result


def build_expected_universe_daily(stock_panel: pd.DataFrame) -> pd.DataFrame:
    """Build an as-of expected universe without future-date lookahead."""
    required = {"trade_date", "ts_code"}
    if stock_panel is None or stock_panel.empty or not required.issubset(stock_panel.columns):
        return pd.DataFrame(columns=["trade_date", "expected_eligible_count", "expected_universe_source"])
    keep = [column for column in ("trade_date", "ts_code", "list_date", "delist_date", "universe_count") if column in stock_panel]
    frame = stock_panel[keep].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype("string")
    frame = frame.dropna(subset=["trade_date", "ts_code"]).drop_duplicates(["trade_date", "ts_code"])
    dates = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
    first_seen = frame.groupby("ts_code")["trade_date"].min()
    meta = frame.sort_values(["ts_code", "trade_date"]).groupby("ts_code", as_index=True).first()
    for column in ("list_date", "delist_date"):
        meta[column] = _parse_vendor_date(meta[column]) if column in meta else pd.NaT
    declared = pd.to_numeric(frame.get("universe_count", pd.Series(index=frame.index)), errors="coerce")
    declared_by_date = declared.groupby(frame["trade_date"]).max().reindex(dates)
    cumulative_distinct: list[int] = []
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for index, day in enumerate(dates):
        seen.update(frame.loc[frame["trade_date"].eq(day), "ts_code"].astype(str).tolist())
        cumulative_distinct.append(len(seen))
        active_codes = first_seen.index[first_seen.le(day)]
        active = meta.loc[active_codes]
        listed = active["list_date"].isna() | active["list_date"].le(day)
        not_delisted = active["delist_date"].isna() | active["delist_date"].ge(day)
        metadata_count = int((listed & not_delisted).sum())
        metadata_available = bool(active["list_date"].notna().any() or active["delist_date"].notna().any())
        if metadata_count > 0 and metadata_available:
            expected, source = metadata_count, "point_in_time_listing_metadata"
        elif pd.notna(declared_by_date.iloc[index]) and float(declared_by_date.iloc[index]) > 0:
            expected, source = int(declared_by_date.iloc[index]), "declared_universe_count"
        else:
            expected, source = cumulative_distinct[-1], "cumulative_observed_symbols"
        rows.append({"trade_date": day, "expected_eligible_count": expected, "expected_universe_source": source})
    return pd.DataFrame(rows)


def ratio_violations(frame: pd.DataFrame) -> pd.Series:
    """Return a boolean row mask for impossible [0, 1] ratios."""
    columns = [
        "adv_ratio", "above_ma20_ratio", "above_ma60_ratio", "limit_up_rate",
        "limit_down_rate", "failed_limit_rate", "breadth_coverage",
        "profit_effect_coverage", "liquidity_coverage",
    ]
    flags = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            flags[column] = values.notna() & (values.lt(0) | values.gt(1))
    return flags.any(axis=1) if not flags.empty else pd.Series(False, index=frame.index)


def _valid_price_mask(frame: pd.DataFrame) -> pd.Series:
    valid = frame["close"].gt(0)
    if "pre_close" in frame:
        valid &= frame["pre_close"].gt(0)
    if "volume" in frame:
        valid &= pd.to_numeric(frame["volume"], errors="coerce").fillna(0).gt(0)
    if "name" in frame:
        valid &= ~frame["name"].astype(str).str.contains(r"\*?ST", case=False, regex=True)
    if "is_st" in frame:
        valid &= pd.to_numeric(frame["is_st"], errors="coerce").fillna(0).ne(1)
    if "recent_ipo_excluded" in frame:
        valid &= ~frame["recent_ipo_excluded"].fillna(False).astype(bool)
    return valid


def _coverage_warning(row: pd.Series, minimum_ratio: float, minimum_universe: int) -> str:
    warnings: list[str] = []
    if row["expected_eligible_count"] < minimum_universe:
        warnings.append(f"EXPECTED_UNIVERSE_BELOW_MINIMUM_{int(row['expected_eligible_count'])}")
    if pd.isna(row["coverage_ratio"]) or row["coverage_ratio"] < minimum_ratio:
        value = "nan" if pd.isna(row["coverage_ratio"]) else f"{row['coverage_ratio']:.3f}"
        warnings.append(f"MARKET_COVERAGE_BELOW_MINIMUM_{value}")
    return ";".join(warnings)


def _parse_vendor_date(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce").dt.normalize()


def _align_series(index: pd.Series, values: pd.Series | None, fallback: pd.Series) -> pd.Series:
    if values is None:
        return fallback
    aligned = pd.Series(values).copy()
    if not isinstance(aligned.index, pd.DatetimeIndex):
        aligned.index = pd.to_datetime(aligned.index, errors="coerce").normalize()
    return pd.to_datetime(index).map(aligned.to_dict()).fillna(fallback)
