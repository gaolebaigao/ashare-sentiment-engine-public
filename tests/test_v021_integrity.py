import numpy as np
import pandas as pd

from ashare_sentiment.data.quality import build_expected_universe_daily, build_market_coverage_daily, ratio_violations
from ashare_sentiment.factors.liquidity import compute_liquidity
from ashare_sentiment.factors.profit_effect import compute_profit_effect
from ashare_sentiment.scoring.market_temperature import _apply_integrity_gate


def _panel(days=3, symbols=100):
    rows = []
    for day_index, day in enumerate(pd.date_range("2026-08-10", periods=days, freq="B")):
        count = symbols if day_index != 1 else 10
        for code_index in range(count):
            code = f"{code_index:06d}.SZ"
            rows.append({
                "trade_date": day,
                "ts_code": code,
                "close": 10.0 + code_index,
                "pre_close": 9.9 + code_index,
                "volume": 100.0,
                "amount_rmb": 1_000_000.0,
                "universe_count": symbols,
                "list_date": 20200101,
            })
    return pd.DataFrame(rows)


def test_expected_universe_does_not_shrink_to_partial_observation():
    coverage = build_market_coverage_daily(
        _panel(),
        config={"data_quality": {"minimum_market_coverage_ratio": 0.90}},
    )
    partial = coverage.iloc[1]
    assert partial["observed_universe"] == 10
    assert partial["expected_eligible_count"] == 100
    assert partial["coverage_ratio"] == 0.10


def test_expected_universe_is_as_of_and_ignores_future_only_symbols():
    before = _panel(days=2, symbols=20)
    future = pd.DataFrame([{
        "trade_date": pd.Timestamp("2026-08-20"), "ts_code": "999999.SZ", "close": 10,
        "pre_close": 9.9, "volume": 1, "amount_rmb": 1, "list_date": 20260820,
    }])
    combined = pd.concat([before, future], ignore_index=True)
    first_only = build_expected_universe_daily(before)
    with_future = build_expected_universe_daily(combined)
    assert with_future.loc[0, "expected_eligible_count"] == first_only.loc[0, "expected_eligible_count"]


def test_limit_rates_use_expected_denominator_and_stay_bounded():
    panel = _panel(days=1, symbols=100)
    limits = pd.DataFrame([{
        "trade_date": panel.trade_date.iloc[0], "ts_code": "000001.SZ",
        "is_limit_up": True, "is_limit_down": False,
    }])
    result = compute_profit_effect(panel, limits, eligible_counts=pd.Series({panel.trade_date.iloc[0]: 100}))
    row = result.iloc[0]
    assert row["limit_up_rate"] == 0.01
    assert 0 <= row["limit_up_rate"] <= 1


def test_real_limit_down_zero_is_distinguished_from_unavailable():
    panel = _panel(days=2, symbols=100)
    limits = pd.DataFrame([{
        "trade_date": day, "ts_code": "000001.SZ", "is_limit_up": True,
        "is_limit_down": False, "limit_down_source": "limit_list_d",
        "limit_down_status": "REAL_ZERO",
    } for day in panel.trade_date.unique()])
    result = compute_profit_effect(panel, limits, eligible_counts=pd.Series({day: 100 for day in panel.trade_date.unique()}))
    assert set(result["limit_down_status"]) == {"REAL_ZERO"}
    assert set(result["profit_effect_quality"]) == {"VALID"}


def test_signed_turnover_intensity_clips_negative_turnover_zscore():
    panel = _panel(days=25, symbols=10)
    breadth = pd.DataFrame({"trade_date": sorted(panel.trade_date.unique()), "adv_ratio": 0.75})
    result = compute_liquidity(panel, breadth, zscore_window=5, zscore_min_periods=5)
    assert (result.loc[result["turnover_zscore"] < 0, "signed_turnover_intensity"] == 0).all()
    positive = result["turnover_zscore"] > 0
    expected = result.loc[positive, "turnover_zscore"] * 0.5
    pd.testing.assert_series_equal(
        result.loc[positive, "signed_turnover_intensity"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_limit_down_degenerate_status_is_explicit():
    panel = _panel(days=25, symbols=100)
    limits = pd.DataFrame([{
        "trade_date": day, "ts_code": "000001.SZ", "is_limit_up": True,
        "is_limit_down": False, "limit_down_source": "limit_list_d",
        "limit_down_status": "REAL_ZERO",
    } for day in panel.trade_date.unique()])
    result = compute_profit_effect(panel, limits, eligible_counts=pd.Series({day: 100 for day in panel.trade_date.unique()}))
    assert result.iloc[-1]["limit_down_status"] == "DEGENERATE"
    assert result.iloc[-1]["profit_effect_quality"] == "DEGRADED"


def test_impossible_ratio_is_rejected():
    assert bool(ratio_violations(pd.DataFrame({"limit_up_rate": [9.4], "adv_ratio": [0.5]})).iloc[0])


def test_coverage_collapse_and_cross_module_mismatch_are_invalid():
    dates = pd.date_range("2026-08-10", periods=2, freq="B")
    result = pd.DataFrame({
        "trade_date": dates,
        "eligible_count": [100, 10],
        "observed_eligible_count": [100, 10],
        "valid_amount_count": [100, 10],
        "data_quality_warnings": ["", ""],
    })
    coverage = pd.DataFrame({
        "trade_date": dates,
        "expected_eligible_count": [100, 100],
        "observed_universe": [100, 10],
        "coverage_ratio": [1.0, 0.1],
        "expected_universe_source": ["declared", "declared"],
    })
    out = _apply_integrity_gate(
        result, coverage,
        {"data_quality": {"minimum_expected_universe": 50, "minimum_market_coverage_ratio": 0.90}},
        production=True,
    )
    assert bool(out.iloc[1]["integrity_invalid"])
    assert "MARKET_COVERAGE_COLLAPSE" in out.iloc[1]["integrity_warnings"]
    assert "CROSS_MODULE_UNIVERSE_MISMATCH_eligible_count" in out.iloc[1]["integrity_warnings"]
