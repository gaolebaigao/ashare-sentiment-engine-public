"""MarketTemperature v0.2 orchestration with production data gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..data.quality import (
    ProductionDataQualityGate,
    build_market_coverage_daily,
    ratio_violations,
)
from ..factors import compute_breadth, compute_liquidity, compute_profit_effect, compute_stretch
from .composite import combine_scores
from .degenerate import mask_degenerate_scores
from .percentile import historical_percentile


def calculate_market_temperature(
    stock_panel: pd.DataFrame,
    limit_panel: pd.DataFrame,
    index_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    *,
    production: bool = False,
    allow_partial_data: bool = False,
) -> pd.DataFrame:
    """Calculate factors and the options-excluded temperature.

    ``production=True`` activates the full-market gate.  Direct factor API
    callers can keep ``production=False`` for small deterministic unit tests.
    """
    assessment = ProductionDataQualityGate(config).assess(stock_panel) if production else None
    scoring = config.get("scoring", {})
    percentile = scoring.get("percentile", {})
    lookback = int(percentile.get("lookback", 756))
    min_periods = int(percentile.get("min_periods", 252))
    factor_cfg = config.get("factors", {})
    windows = factor_cfg.get("windows", {})
    quality_cfg = config.get("data_quality", {})
    ipo_days = int(config.get("universe", {}).get("exclude_recent_ipos_days", 0))
    minimum_coverage = float(quality_cfg.get("minimum_market_coverage_ratio", quality_cfg.get("minimum_coverage_ratio", 0.90)))
    degenerate_cfg = quality_cfg.get("degenerate_factor", {})

    coverage = build_market_coverage_daily(stock_panel, config=config)
    expected_counts = coverage.set_index("trade_date")["expected_eligible_count"]
    breadth = compute_breadth(
        stock_panel,
        ma20_window=int(windows.get("ma20", 20)),
        ma60_window=int(windows.get("ma60", 60)),
        new_high_low_window=int(windows.get("new_high_low", 60)),
        exclude_recent_ipos_days=ipo_days,
        minimum_coverage_ratio=minimum_coverage,
        expected_counts=expected_counts,
    )
    profit = compute_profit_effect(
        stock_panel,
        limit_panel,
        eligible_counts=expected_counts,
        exclude_recent_ipos_days=ipo_days,
        minimum_coverage_ratio=minimum_coverage,
    )
    liquidity = compute_liquidity(
        stock_panel,
        breadth,
        zscore_window=int(windows.get("turnover_zscore", 60)),
        exclude_recent_ipos_days=ipo_days,
        minimum_coverage_ratio=minimum_coverage,
        expected_counts=expected_counts,
    )
    stretch = compute_stretch(
        index_frames,
        rsi_window=int(windows.get("rsi", 14)),
        atr_window=int(windows.get("atr", 14)),
        ma_window=int(windows.get("ma20", 20)),
    )

    breadth = _score_breadth(breadth, factor_cfg.get("breadth_weights", {}), lookback, min_periods, degenerate_cfg)
    profit = _score_profit(profit, factor_cfg.get("profit_effect_weights", {}), lookback, min_periods, degenerate_cfg)
    liquidity = _score_liquidity(liquidity, factor_cfg.get("liquidity_weights", {}), lookback, min_periods, degenerate_cfg)
    stretch = _score_stretch(stretch, factor_cfg.get("stretch_weights", {}), lookback, min_periods)
    stretch["stretch_quality"] = stretch["stretch_score"].notna().map({True: "VALID", False: "INVALID"})

    result = breadth.merge(profit, on="trade_date", how="outer", suffixes=("", "_profit"))
    result = result.merge(liquidity, on="trade_date", how="outer", suffixes=("", "_liquidity"))
    result = result.merge(stretch, on="trade_date", how="outer", suffixes=("", "_stretch"))
    result = result.sort_values("trade_date").reset_index(drop=True)

    module_weights = dict(scoring.get("weights", {}))
    result["options_score"] = pd.NA
    result = combine_scores(
        result,
        {
            "breadth_score": float(module_weights.get("breadth", 0.30)),
            "profit_effect_score": float(module_weights.get("profit_effect", 0.25)),
            "liquidity_score": float(module_weights.get("liquidity", 0.15)),
            "options_score": float(module_weights.get("options", 0.15)),
            "stretch_score": float(module_weights.get("stretch", 0.15)),
        },
        score_column="raw_market_temperature",
        metadata_prefix="market_temperature",
    )
    result["available_factors"] = "breadth,profit_effect,liquidity,stretch"
    result["missing_factors"] = result.apply(_missing_factor_summary, axis=1)
    result["data_quality_warnings"] = _warnings(
        config,
        stock_panel=stock_panel,
        limit_panel=limit_panel,
        assessment=assessment,
    )
    result = _apply_integrity_gate(result, coverage, config, production=production)
    for prefix, score_column in (
        ("breadth", "breadth_score"),
        ("profit_effect", "profit_effect_score"),
        ("liquidity", "liquidity_score"),
    ):
        degenerate_column = f"{prefix}_degenerate_factors"
        quality_column = f"{prefix}_quality"
        if degenerate_column in result.columns and quality_column in result.columns:
            degenerate_rows = result[score_column].isna() & result[degenerate_column].astype(str).ne("")
            result.loc[degenerate_rows, quality_column] = "DEGENERATE"
    required_quality = result[["breadth_quality", "profit_effect_quality", "liquidity_quality"]].fillna("INVALID")
    invalid_required = required_quality.isin({"INVALID"}).any(axis=1)
    degraded_required = required_quality.isin({"DEGRADED", "DEGENERATE"}).any(axis=1)
    partial = result["integrity_invalid"]
    result["market_temperature_quality"] = "A"
    result.loc[result["data_quality_warnings"].ne(""), "market_temperature_quality"] = "B"
    result.loc[degraded_required | invalid_required, "market_temperature_quality"] = "C"
    result.loc[partial | invalid_required, "market_temperature_quality"] = "INVALID"
    result["confidence"] = result["market_temperature_quality"].map(
        {"A": "HIGH", "B": "MEDIUM", "C": "LOW", "INVALID": "NONE"}
    )
    result["data_quality_status"] = result["data_quality_warnings"].map(
        lambda value: "WARN" if value else "PASS"
    )
    result.loc[result["market_temperature_quality"].eq("INVALID"), "data_quality_status"] = "INVALID"
    if allow_partial_data:
        result.loc[result["market_temperature_quality"].eq("INVALID"), "data_quality_warnings"] = (
            result.loc[result["market_temperature_quality"].eq("INVALID"), "data_quality_warnings"]
            + ";PARTIAL_MARKET_PANEL_NOT_VALID_FOR_TRADING"
        )
    result["date"] = result["trade_date"]
    # A production score is never emitted for a failed day.  Research mode
    # retains the row for diagnosis but labels it explicitly.
    result.loc[result["integrity_invalid"], "raw_market_temperature"] = pd.NA
    if not allow_partial_data:
        result.loc[result["integrity_invalid"], "data_quality_status"] = "INVALID"
    result["warnings"] = result["data_quality_warnings"]
    return result


def calculate_intraday_market_temperature(
    stock_panel: pd.DataFrame,
    limit_panel: pd.DataFrame,
    index_frames: Mapping[str, pd.DataFrame],
    historical_scores: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Score only the newest live day against persisted daily factor history.

    Intraday refreshes need recent stock bars for rolling MA/turnover inputs,
    but they do not need to recompute those inputs for millions of historical
    stock-day rows. Persisted daily raw factors provide the percentile history;
    the returned frame contains only the live day.
    """
    assessment = ProductionDataQualityGate(config).assess(stock_panel)
    scoring = config.get("scoring", {})
    percentile = scoring.get("percentile", {})
    lookback = int(percentile.get("lookback", 756))
    min_periods = int(percentile.get("min_periods", 252))
    factor_cfg = config.get("factors", {})
    windows = factor_cfg.get("windows", {})
    quality_cfg = config.get("data_quality", {})
    ipo_days = int(config.get("universe", {}).get("exclude_recent_ipos_days", 0))
    minimum_coverage = float(quality_cfg.get("minimum_market_coverage_ratio", quality_cfg.get("minimum_coverage_ratio", 0.90)))
    degenerate_cfg = quality_cfg.get("degenerate_factor", {})

    coverage = build_market_coverage_daily(stock_panel, config=config)
    expected_counts = coverage.set_index("trade_date")["expected_eligible_count"]
    breadth_raw = compute_breadth(
        stock_panel,
        ma20_window=int(windows.get("ma20", 20)),
        ma60_window=int(windows.get("ma60", 60)),
        new_high_low_window=int(windows.get("new_high_low", 60)),
        exclude_recent_ipos_days=ipo_days,
        minimum_coverage_ratio=minimum_coverage,
        expected_counts=expected_counts,
    )
    profit_raw = compute_profit_effect(
        stock_panel,
        limit_panel,
        eligible_counts=expected_counts,
        exclude_recent_ipos_days=ipo_days,
        minimum_coverage_ratio=minimum_coverage,
    )
    liquidity_raw = compute_liquidity(
        stock_panel,
        breadth_raw,
        zscore_window=int(windows.get("turnover_zscore", 60)),
        exclude_recent_ipos_days=ipo_days,
        minimum_coverage_ratio=minimum_coverage,
        expected_counts=expected_counts,
    )
    stretch_raw = compute_stretch(
        index_frames,
        rsi_window=int(windows.get("rsi", 14)),
        atr_window=int(windows.get("atr", 14)),
        ma_window=int(windows.get("ma20", 20)),
    )

    target_date = pd.to_datetime(stock_panel["trade_date"], errors="coerce").max().normalize()
    history = historical_scores.copy()
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
    history = history[history["trade_date"].lt(target_date)]

    def with_history(current: pd.DataFrame) -> pd.DataFrame:
        latest = current[pd.to_datetime(current["trade_date"], errors="coerce").eq(target_date)].copy()
        past = history.reindex(columns=latest.columns)
        return pd.concat([past, latest], ignore_index=True, sort=False).sort_values("trade_date").reset_index(drop=True)

    breadth = _score_breadth(with_history(breadth_raw), factor_cfg.get("breadth_weights", {}), lookback, min_periods, degenerate_cfg)
    profit = _score_profit(with_history(profit_raw), factor_cfg.get("profit_effect_weights", {}), lookback, min_periods, degenerate_cfg)
    liquidity = _score_liquidity(with_history(liquidity_raw), factor_cfg.get("liquidity_weights", {}), lookback, min_periods, degenerate_cfg)
    stretch = _score_stretch(with_history(stretch_raw), factor_cfg.get("stretch_weights", {}), lookback, min_periods)
    breadth = breadth[breadth["trade_date"].eq(target_date)].tail(1)
    profit = profit[profit["trade_date"].eq(target_date)].tail(1)
    liquidity = liquidity[liquidity["trade_date"].eq(target_date)].tail(1)
    stretch = stretch[stretch["trade_date"].eq(target_date)].tail(1)
    stretch["stretch_quality"] = stretch["stretch_score"].notna().map({True: "VALID", False: "INVALID"})

    result = breadth.merge(profit, on="trade_date", how="outer", suffixes=("", "_profit"))
    result = result.merge(liquidity, on="trade_date", how="outer", suffixes=("", "_liquidity"))
    result = result.merge(stretch, on="trade_date", how="outer", suffixes=("", "_stretch"))
    module_weights = dict(scoring.get("weights", {}))
    result["options_score"] = pd.NA
    result = combine_scores(
        result,
        {
            "breadth_score": float(module_weights.get("breadth", 0.30)),
            "profit_effect_score": float(module_weights.get("profit_effect", 0.25)),
            "liquidity_score": float(module_weights.get("liquidity", 0.15)),
            "options_score": float(module_weights.get("options", 0.15)),
            "stretch_score": float(module_weights.get("stretch", 0.15)),
        },
        score_column="raw_market_temperature",
        metadata_prefix="market_temperature",
    )
    result["available_factors"] = "breadth,profit_effect,liquidity,stretch"
    result["missing_factors"] = result.apply(_missing_factor_summary, axis=1)
    result["data_quality_warnings"] = _warnings(config, stock_panel=stock_panel, limit_panel=limit_panel, assessment=assessment)
    current_coverage = coverage[coverage["trade_date"].eq(target_date)]
    result = _apply_integrity_gate(result, current_coverage, config, production=True)
    for prefix, score_column in (("breadth", "breadth_score"), ("profit_effect", "profit_effect_score"), ("liquidity", "liquidity_score")):
        degenerate_column = f"{prefix}_degenerate_factors"
        quality_column = f"{prefix}_quality"
        if degenerate_column in result.columns and quality_column in result.columns:
            degenerate_rows = result[score_column].isna() & result[degenerate_column].astype(str).ne("")
            result.loc[degenerate_rows, quality_column] = "DEGENERATE"
    required_quality = result[["breadth_quality", "profit_effect_quality", "liquidity_quality"]].fillna("INVALID")
    invalid_required = required_quality.isin({"INVALID"}).any(axis=1)
    degraded_required = required_quality.isin({"DEGRADED", "DEGENERATE"}).any(axis=1)
    result["market_temperature_quality"] = "A"
    result.loc[result["data_quality_warnings"].ne(""), "market_temperature_quality"] = "B"
    result.loc[degraded_required | invalid_required, "market_temperature_quality"] = "C"
    result.loc[result["integrity_invalid"] | invalid_required, "market_temperature_quality"] = "INVALID"
    result["confidence"] = result["market_temperature_quality"].map({"A": "HIGH", "B": "MEDIUM", "C": "LOW", "INVALID": "NONE"})
    result["data_quality_status"] = result["data_quality_warnings"].map(lambda value: "WARN" if value else "PASS")
    result.loc[result["market_temperature_quality"].eq("INVALID"), "data_quality_status"] = "INVALID"
    result["date"] = result["trade_date"]
    result.loc[result["integrity_invalid"], "raw_market_temperature"] = pd.NA
    result["warnings"] = result["data_quality_warnings"]
    return result.reset_index(drop=True)


def _apply_integrity_gate(
    result: pd.DataFrame,
    coverage: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    production: bool,
) -> pd.DataFrame:
    """Attach per-day integrity flags; never turn missing data into zeros."""
    output = result.copy()
    coverage = coverage.set_index("trade_date").reindex(output["trade_date"])
    output["expected_eligible_count"] = coverage["expected_eligible_count"].to_numpy()
    output["observed_universe"] = coverage["observed_universe"].to_numpy()
    output["observed_valid_count"] = coverage["observed_universe"].to_numpy()
    output["market_coverage_ratio"] = coverage["coverage_ratio"].to_numpy()
    output["expected_universe_source"] = coverage["expected_universe_source"].to_numpy()
    quality = config.get("data_quality", {})
    minimum_universe = int(quality.get("minimum_expected_universe", quality.get("minimum_market_universe", 3000)))
    minimum_ratio = float(quality.get("minimum_market_coverage_ratio", quality.get("minimum_coverage_ratio", 0.90)))
    maximum_deviation = float(quality.get("max_cross_module_universe_deviation", 0.10))
    collapse_ratio = float(quality.get("minimum_relative_universe_ratio", 0.80))
    warnings: list[list[str]] = []
    invalid: list[bool] = []
    previous_observed: float | None = None
    for index, row in output.iterrows():
        current_warnings: list[str] = []
        expected = float(row.get("expected_eligible_count", 0) or 0)
        observed = float(row.get("observed_universe", 0) or 0)
        coverage_ratio = float(row.get("market_coverage_ratio")) if pd.notna(row.get("market_coverage_ratio")) else float("nan")
        if expected < minimum_universe:
            current_warnings.append(f"EXPECTED_UNIVERSE_BELOW_MINIMUM_{int(expected)}")
        if pd.isna(coverage_ratio) or coverage_ratio < minimum_ratio:
            current_warnings.append("MARKET_COVERAGE_BELOW_MINIMUM")
        if previous_observed is not None and previous_observed > 0 and observed / previous_observed < collapse_ratio:
            current_warnings.append("MARKET_COVERAGE_COLLAPSE")
        previous_observed = observed if observed > 0 else previous_observed
        for module_count in ("eligible_count", "observed_eligible_count", "valid_amount_count"):
            if module_count not in row or expected <= 0 or pd.isna(row[module_count]):
                continue
            deviation = abs(float(row[module_count]) - expected) / expected
            if deviation > maximum_deviation:
                current_warnings.append(f"CROSS_MODULE_UNIVERSE_MISMATCH_{module_count}")
        warnings.append(current_warnings)
        invalid.append(bool(current_warnings))
    output["integrity_warnings"] = [";".join(items) for items in warnings]
    output["integrity_invalid"] = invalid
    output["ratio_violation"] = ratio_violations(output).to_numpy()
    output.loc[output["ratio_violation"], "integrity_warnings"] = output.loc[output["ratio_violation"], "integrity_warnings"].map(
        lambda value: ";".join(item for item in [value, "IMPOSSIBLE_RATIO"] if item)
    )
    output.loc[output["ratio_violation"], "integrity_invalid"] = True
    output["data_quality_warnings"] = output.apply(
        lambda row: ";".join(item for item in [str(row["data_quality_warnings"]), str(row["integrity_warnings"])] if item and item != "nan"),
        axis=1,
    )
    if production and not output.empty:
        output["production_gate"] = output["integrity_invalid"].map({True: "INVALID", False: "PASS"})
    else:
        output["production_gate"] = "RESEARCH"
    return output


def _score_breadth(frame: pd.DataFrame, weights: Mapping[str, float], lookback: int, min_periods: int, degenerate_cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    raw = frame.copy()
    score_frame = pd.DataFrame({"trade_date": raw["trade_date"]})
    for name in ("adv_ratio", "above_ma20_ratio", "above_ma60_ratio", "nhnl_ratio"):
        score_frame[name] = historical_percentile(raw[name], lookback=lookback, min_periods=min_periods)
    score_frame, degenerate = mask_degenerate_scores(score_frame=score_frame, raw=raw, factor_names=list(score_frame.columns[1:]), **_degenerate_kwargs(degenerate_cfg))
    scored = combine_scores(
        score_frame,
        {name: float(weights.get(name, default)) for name, default in {
            "adv_ratio": 0.30, "above_ma20_ratio": 0.25, "above_ma60_ratio": 0.25, "nhnl_ratio": 0.20
        }.items()},
        score_column="breadth_score",
        metadata_prefix="breadth",
    )
    scored["breadth_degenerate_factors"] = degenerate.to_numpy()
    return raw.merge(
        scored.drop(columns=["trade_date", "adv_ratio", "above_ma20_ratio", "above_ma60_ratio", "nhnl_ratio"]),
        left_index=True,
        right_index=True,
    )


def _score_profit(frame: pd.DataFrame, weights: Mapping[str, float], lookback: int, min_periods: int, degenerate_cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    raw = frame.copy()
    score_frame = pd.DataFrame({"trade_date": raw["trade_date"]})
    directions = {
        "limit_up_rate": "bullish",
        "limit_down_rate": "bearish",
        "yesterday_limitup_mean_return": "bullish",
        "failed_limit_rate": "bearish",
    }
    for name, direction in directions.items():
        score_frame[name] = historical_percentile(raw[name], lookback=lookback, min_periods=min_periods, direction=direction)
    configured = {
        name: float(weights.get(name, default)) for name, default in {
            "limit_up_rate": 0.35,
            "limit_down_rate": 0.25,
            "yesterday_limitup_mean_return": 0.40,
            "failed_limit_rate": 0.00,
        }.items()
    }
    score_frame, degenerate = mask_degenerate_scores(score_frame=score_frame, raw=raw, factor_names=list(directions), **_degenerate_kwargs(degenerate_cfg))
    scored = combine_scores(score_frame, configured, score_column="profit_effect_score", metadata_prefix="profit_effect")
    scored["profit_effect_degenerate_factors"] = degenerate.to_numpy()
    return raw.merge(
        scored.drop(columns=["trade_date", "limit_up_rate", "limit_down_rate", "yesterday_limitup_mean_return", "failed_limit_rate"]),
        left_index=True,
        right_index=True,
    )


def _score_liquidity(frame: pd.DataFrame, weights: Mapping[str, float], lookback: int, min_periods: int, degenerate_cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    raw = frame.copy()
    score_frame = pd.DataFrame({"trade_date": raw["trade_date"]})
    score_frame["turnover_percentile"] = historical_percentile(raw["total_market_turnover"], lookback=lookback, min_periods=min_periods)
    score_frame["signed_turnover_intensity"] = historical_percentile(raw["signed_turnover_intensity"], lookback=lookback, min_periods=min_periods)
    configured = {
        "turnover_percentile": float(weights.get("turnover_percentile", 0.50)),
        "signed_turnover_intensity": float(weights.get("signed_turnover_intensity", 0.50)),
    }
    score_frame, degenerate = mask_degenerate_scores(score_frame=score_frame, raw=raw, factor_names=["turnover_percentile", "signed_turnover_intensity"], **_degenerate_kwargs(degenerate_cfg))
    scored = combine_scores(score_frame, configured, score_column="liquidity_score", metadata_prefix="liquidity")
    scored["liquidity_degenerate_factors"] = degenerate.to_numpy()
    return raw.merge(
        scored.drop(columns=["trade_date", "signed_turnover_intensity"]),
        left_index=True,
        right_index=True,
    )


def _score_stretch(frame: pd.DataFrame, weights: Mapping[str, float], lookback: int, min_periods: int) -> pd.DataFrame:
    raw = frame.copy()
    aliases = sorted({column.rsplit("_", 1)[0] for column in raw.columns if column.endswith("_rsi14")})
    score_frame = pd.DataFrame({"trade_date": raw["trade_date"]})
    factor_alias_columns: dict[str, list[str]] = {name: [] for name in ("rsi14", "ma20_atr", "return5d", "return20d")}
    for alias in aliases:
        for factor_name in factor_alias_columns:
            raw_name = f"{alias}_{factor_name}"
            score_name = f"{alias}_{factor_name}_score"
            if raw_name not in raw:
                continue
            score_frame[score_name] = historical_percentile(raw[raw_name], lookback=lookback, min_periods=min_periods)
            factor_alias_columns[factor_name].append(score_name)
    for factor_name, columns in factor_alias_columns.items():
        score_frame[factor_name] = score_frame[columns].mean(axis=1) if columns else pd.NA
    configured = {
        name: float(weights.get(name, 0.25)) for name in ("rsi14", "ma20_atr", "return5d", "return20d")
    }
    scored = combine_scores(score_frame, configured, score_column="stretch_score", metadata_prefix="stretch")
    return raw.merge(scored.drop(columns=["trade_date"] + list(configured) + [column for columns in factor_alias_columns.values() for column in columns]), left_index=True, right_index=True)


def _missing_factor_summary(row: pd.Series) -> str:
    missing = ["options", "failed_limit_rate", "margin_buy_ratio"]
    if pd.isna(row.get("breadth_score")):
        missing.append("breadth_score")
    if pd.isna(row.get("profit_effect_score")):
        missing.append("profit_effect_score")
    if pd.isna(row.get("liquidity_score")):
        missing.append("liquidity_score")
    if pd.isna(row.get("stretch_score")):
        missing.append("stretch_score")
    return ",".join(missing)


def _warnings(
    config: Mapping[str, Any],
    *,
    stock_panel: pd.DataFrame | None = None,
    limit_panel: pd.DataFrame | None = None,
    assessment: Any | None = None,
) -> str:
    warnings = ["OPTIONS_UNAVAILABLE", "MARGIN_OPTIONAL", "FAILED_LIMIT_RATE_UNAVAILABLE"]
    provider = str(config.get("data", {}).get("provider", "")).lower()
    if provider in {"baostock", "bao-stock", "bao_stock"}:
        warnings.append("LIMIT_STATUS_APPROXIMATE_BAOSTOCK_BOARD_BANDS")
    if stock_panel is not None and "ts_code" in stock_panel.columns:
        symbol_count = stock_panel["ts_code"].nunique(dropna=True)
        if symbol_count < 1000:
            warnings.append(f"PARTIAL_STOCK_PANEL_SYMBOLS_{symbol_count}")
    if assessment is not None:
        warnings.extend(item for item in assessment.warnings if item not in warnings)
    if limit_panel is None or limit_panel.empty:
        warnings.append("LIMIT_DATA_UNAVAILABLE")
    elif "limit_method" in limit_panel.columns and limit_panel["limit_method"].astype(str).str.contains("approx", case=False).any():
        warnings.append("LIMIT_RULES_APPROXIMATE_NOT_PRODUCTION_VALID")
    if config.get("data", {}).get("survivorship_bias_warning", True):
        warnings.insert(0, "SURVIVORSHIP_BIAS_WARNING")
    return ";".join(warnings)


def _degenerate_kwargs(config: Mapping[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    return {
        "window": int(config.get("window", 20)),
        "max_unique": int(config.get("max_unique", 2)),
        "minimum_valid": int(config.get("minimum_valid", 5)),
        "variance_floor": float(config.get("variance_floor", 1e-12)),
    }
