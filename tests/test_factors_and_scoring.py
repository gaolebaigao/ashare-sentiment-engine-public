import unittest

import numpy as np
import pandas as pd

from ashare_sentiment.factors import compute_breadth, compute_liquidity, compute_profit_effect, compute_stretch
from ashare_sentiment.scoring.composite import combine_scores, renormalize_weights
from ashare_sentiment.scoring.market_temperature import calculate_market_temperature
from ashare_sentiment.scoring.percentile import historical_percentile


def stock_fixture(days: int = 180) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2021-01-01", periods=days, freq="B")
    for i, day in enumerate(dates):
        for code, drift in (("000001.SZ", 0.8), ("000002.SZ", -0.2), ("600000.SH", 0.1), ("600001.SH", -0.05)):
            close = 100 + i * drift + (0 if code.startswith("000") else 5)
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "close": close,
                    "pre_close": close - drift,
                    "volume": 1000 + i * 3,
                    "amount_rmb": (1_000_000 + i * 10_000) * (1 if drift >= 0 else 0.8),
                }
            )
    return pd.DataFrame(rows)


def index_fixture(days: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=days, freq="B")
    close = pd.Series(np.linspace(100, 140, days))
    return pd.DataFrame({"trade_date": dates, "close": close, "high": close + 1, "low": close - 1})


class FactorTests(unittest.TestCase):
    def test_advance_ratio_excludes_flat_from_denominator(self):
        panel = pd.DataFrame(
            {
                "trade_date": ["2021-01-01"] * 3,
                "ts_code": ["A", "B", "C"],
                "close": [11, 9, 10],
                "pre_close": [10, 10, 10],
                "volume": [1, 1, 1],
                "amount_rmb": [1, 1, 1],
            }
        )
        result = compute_breadth(panel, ma20_window=1, ma60_window=1, new_high_low_window=1).iloc[0]
        self.assertEqual(result["advancing_count"], 1)
        self.assertEqual(result["declining_count"], 1)
        self.assertEqual(result["flat_count"], 1)
        self.assertEqual(result["adv_ratio"], 0.5)

    def test_ma_breadth_and_new_high_low_use_trailing_windows(self):
        panel = stock_fixture(80)
        result = compute_breadth(panel, ma20_window=20, ma60_window=60, new_high_low_window=60)
        self.assertTrue(result["above_ma20_ratio"].notna().any())
        self.assertTrue(result["above_ma60_ratio"].notna().any())
        self.assertTrue(result["new_high_60"].sum() > 0)
        self.assertTrue(result["new_low_60"].sum() > 0)

    def test_limit_rates_and_next_day_leader_return(self):
        panel = stock_fixture(5)
        dates = sorted(panel["trade_date"].unique())
        limits = pd.DataFrame(
            [
                {"trade_date": dates[0], "ts_code": "000001.SZ", "is_limit_up": True, "is_limit_down": False},
                {"trade_date": dates[0], "ts_code": "000002.SZ", "is_limit_up": False, "is_limit_down": True},
            ]
        )
        result = compute_profit_effect(panel, limits)
        self.assertEqual(result.iloc[0]["limit_up_count"], 1)
        self.assertEqual(result.iloc[0]["limit_down_count"], 1)
        self.assertEqual(result.iloc[0]["limit_up_rate"], 0.25)
        self.assertTrue(pd.notna(result.iloc[1]["yesterday_limitup_mean_return"]))

    def test_liquidity_and_signed_turnover(self):
        panel = stock_fixture(100)
        breadth = compute_breadth(panel)
        result = compute_liquidity(panel, breadth, zscore_window=20)
        self.assertTrue(result["total_market_turnover"].gt(0).all())
        self.assertTrue(result["turnover_zscore"].notna().any())
        expected = result["turnover_zscore"] * (2 * breadth.set_index("trade_date").loc[result["trade_date"], "adv_ratio"].to_numpy() - 1)
        expected = pd.Series(expected, index=result.index)
        pd.testing.assert_series_equal(result["signed_turnover_intensity"], expected, check_names=False)

    def test_wilder_rsi_and_atr_distance(self):
        up = index_fixture(40)
        result = compute_stretch({"hs300": up}, rsi_window=14, atr_window=14, ma_window=20)
        self.assertAlmostEqual(result["hs300_rsi14"].dropna().iloc[-1], 100.0)
        self.assertTrue(result["hs300_ma20_atr"].notna().any())
        self.assertTrue(result["hs300_return5d"].notna().any())

    def test_historical_percentile_and_bearish_inversion(self):
        values = pd.Series([1, 2, 3, 4, 5], dtype=float)
        bullish = historical_percentile(values, lookback=3, min_periods=3)
        bearish = historical_percentile(values, lookback=3, min_periods=3, direction="bearish")
        self.assertGreater(bullish.iloc[-1], bearish.iloc[-1])
        self.assertAlmostEqual(bullish.iloc[-1] + bearish.iloc[-1], 100.0)

    def test_missing_factor_weights_are_renormalized(self):
        self.assertEqual(renormalize_weights({"a": True, "b": False}, {"a": 0.3, "b": 0.7}), {"a": 1.0, "b": 0.0})
        result = combine_scores(pd.DataFrame({"a": [80.0], "b": [np.nan]}), {"a": 0.3, "b": 0.7}, score_column="score")
        self.assertEqual(result.iloc[0]["score"], 80.0)
        self.assertIn("b", result.iloc[0]["score_missing_factors"])

    def test_market_temperature_is_0_to_100_and_options_are_missing(self):
        stocks = stock_fixture(180)
        dates = sorted(stocks["trade_date"].unique())
        limits = pd.DataFrame(
            [{"trade_date": day, "ts_code": "000001.SZ", "is_limit_up": True, "is_limit_down": False} for day in dates]
        )
        config = {
            "data": {"survivorship_bias_warning": True},
            "scoring": {
                "weights": {"breadth": 0.30, "profit_effect": 0.25, "liquidity": 0.15, "options": 0.15, "stretch": 0.15},
                "percentile": {"lookback": 100, "min_periods": 20},
            },
            "factors": {},
        }
        out = calculate_market_temperature(stocks, limits, {"hs300": index_fixture(180), "csi1000": index_fixture(180), "chinext": index_fixture(180)}, config)
        valid = out["raw_market_temperature"].dropna()
        self.assertTrue(valid.between(0, 100).all())
        self.assertIn("options", out.iloc[-1]["missing_factors"])

    def test_no_future_data_changes_history(self):
        values = pd.Series(range(1, 21), index=pd.date_range("2024-01-01", periods=20, freq="D"), dtype=float)
        before = historical_percentile(values.iloc[:15], lookback=10, min_periods=5)
        after = historical_percentile(values, lookback=10, min_periods=5).iloc[:15]
        pd.testing.assert_series_equal(before, after)

    def test_no_future_data_changes_full_temperature_history(self):
        full_stocks = stock_fixture(180)
        short_dates = sorted(full_stocks["trade_date"].unique())[:150]
        short_stocks = full_stocks[full_stocks["trade_date"].isin(short_dates)].copy()
        full_limits = pd.DataFrame(
            [{"trade_date": day, "ts_code": "000001.SZ", "is_limit_up": True, "is_limit_down": False}
             for day in sorted(full_stocks["trade_date"].unique())]
        )
        short_limits = full_limits[full_limits["trade_date"].isin(short_dates)].copy()
        config = {
            "data": {"survivorship_bias_warning": True},
            "scoring": {
                "weights": {"breadth": 0.30, "profit_effect": 0.25, "liquidity": 0.15, "options": 0.15, "stretch": 0.15},
                "percentile": {"lookback": 100, "min_periods": 20},
            },
            "factors": {},
        }
        full_indexes = {alias: index_fixture(180) for alias in ("hs300", "csi1000", "chinext")}
        short_indexes = {alias: frame.iloc[:150].copy() for alias, frame in full_indexes.items()}
        before = calculate_market_temperature(short_stocks, short_limits, short_indexes, config)
        after = calculate_market_temperature(full_stocks, full_limits, full_indexes, config)
        cutoff = short_dates[-5]
        before = before[before["trade_date"] <= cutoff].set_index("trade_date")
        after = after[after["trade_date"] <= cutoff].set_index("trade_date")
        columns = ["breadth_score", "profit_effect_score", "liquidity_score", "stretch_score", "raw_market_temperature"]
        pd.testing.assert_frame_equal(before[columns], after[columns], check_exact=False, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
