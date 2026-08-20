import unittest

import numpy as np
import pandas as pd

from ashare_sentiment.data.quality import InsufficientMarketCoverage, ProductionDataQualityGate, build_market_coverage_daily
from ashare_sentiment.scoring.composite import combine_scores
from ashare_sentiment.scoring.degenerate import degenerate_mask, mask_degenerate_scores


def panel(symbol_count: int, days: int = 2) -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2026-08-14", periods=days, freq="B"):
        for index in range(symbol_count):
            close = 10.0 + index * 0.01
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": f"{index:06d}.SZ",
                    "close": close,
                    "pre_close": close - 0.01,
                    "volume": 100.0,
                    "amount_rmb": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


class V02QualityTests(unittest.TestCase):
    def test_partial_market_panel_rejected(self):
        config = {"data_quality": {"minimum_market_universe": 3000, "minimum_coverage_ratio": 0.90}}
        with self.assertRaises(InsufficientMarketCoverage):
            ProductionDataQualityGate(config).validate(panel(10))

    def test_full_market_panel_accepted(self):
        config = {"data_quality": {"minimum_market_universe": 3000, "minimum_coverage_ratio": 0.90}}
        assessment = ProductionDataQualityGate(config).validate(panel(3001))
        self.assertTrue(assessment.valid)
        self.assertEqual(assessment.latest_eligible_count, 3001)

    def test_degenerate_factor_gets_zero_effective_weight(self):
        raw = pd.DataFrame({"factor": [1.0] * 20})
        score = pd.DataFrame({"factor": [50.0] * 20})
        masked, labels = mask_degenerate_scores(raw, score, ["factor"], window=20, max_unique=2, minimum_valid=5)
        self.assertTrue(bool(degenerate_mask(raw["factor"], window=20).iloc[-1]))
        combined = combine_scores(masked, {"factor": 1.0}, score_column="score")
        self.assertTrue(pd.isna(combined.iloc[-1]["score"]))
        self.assertEqual(combined.iloc[-1]["score_effective_weights"], '{"factor": 0.0}')
        self.assertEqual(labels.iloc[-1], "factor")

    def test_missing_factor_is_not_neutral_50(self):
        result = combine_scores(
            pd.DataFrame({"a": [np.nan], "b": [np.nan]}),
            {"a": 0.5, "b": 0.5},
            score_column="score",
        )
        self.assertTrue(pd.isna(result.iloc[0]["score"]))

    def test_coverage_report_keeps_counts_and_ratio(self):
        frame = panel(10)
        frame.loc[0, "amount_rmb"] = np.nan
        result = build_market_coverage_daily(frame, config={"data_quality": {"minimum_coverage_ratio": 0.90}})
        self.assertEqual(result.iloc[0]["known_stocks"], 10)
        self.assertEqual(result.iloc[0]["valid_price_count"], 10)
        self.assertEqual(result.iloc[0]["valid_amount_count"], 9)
        self.assertEqual(result.iloc[0]["data_quality"], "VALID")


if __name__ == "__main__":
    unittest.main()
