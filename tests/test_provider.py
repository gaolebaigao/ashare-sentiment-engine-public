import unittest

import pandas as pd

from ashare_sentiment.data.base import DataProvider, ProviderDataUnavailable
from ashare_sentiment.data.tushare_provider import TushareProvider


class FakeTushareApi:
    def daily(self, **kwargs):
        fields = kwargs.get("fields", "")
        if fields == "ts_code,trade_date,close,pct_chg":
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["20260817"],
                    "close": [10.0],
                    "pct_chg": [1.0],
                }
            )
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260817"],
                "open": [9.9],
                "high": [10.1],
                "low": [9.8],
                "close": [10.0],
                "pre_close": [9.9],
                "pct_chg": [1.0],
                "vol": [1000],
                "amount": [10000],
            }
        )

    def daily_basic(self, **kwargs):
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260817"],
                "close": [10.0],
                "turnover_rate": [2.0],
                "turnover_rate_f": [2.5],
                "volume_ratio": [1.1],
                "total_mv": [100000],
                "circ_mv": [80000],
                "limit_status": [1],
            }
        )

    def stk_limit(self, **kwargs):
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260817"],
                "pre_close": [9.9],
                "up_limit": [10.89],
                "down_limit": [8.91],
            }
        )

    def index_daily(self, **kwargs):
        return pd.DataFrame({"ts_code": ["000300.SH"], "trade_date": ["20260817"], "close": [4000]})

    def margin(self, **kwargs):
        return pd.DataFrame({"trade_date": ["20260817"], "rzye": [100.0]})


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = TushareProvider(api=FakeTushareApi())

    def test_provider_implements_common_contract(self):
        self.assertIsInstance(self.provider, DataProvider)
        stock = self.provider.get_stock_daily("2026-08-17", "2026-08-17", "000001.SZ")
        self.assertIn("trade_date", stock.columns)
        self.assertEqual(stock.iloc[0]["trade_date"].strftime("%Y%m%d"), "20260817")

    def test_market_breadth_joins_daily_and_daily_basic(self):
        breadth = self.provider.get_market_breadth("2026-08-17", "2026-08-17")
        self.assertIn("turnover_rate", breadth.columns)
        self.assertIn("volume", breadth.columns)

    def test_limit_flags_use_provider_limit_prices(self):
        limits = self.provider.get_limit_up_down("2026-08-17", "2026-08-17")
        self.assertFalse(bool(limits.iloc[0]["is_limit_up"]))
        self.assertFalse(bool(limits.iloc[0]["is_limit_down"]))

    def test_unavailable_options_are_not_silently_filled(self):
        with self.assertRaises(ProviderDataUnavailable):
            self.provider.get_option_data("2026-08-17", "2026-08-17")


if __name__ == "__main__":
    unittest.main()
