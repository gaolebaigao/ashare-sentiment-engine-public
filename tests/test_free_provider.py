import unittest

import pandas as pd

from ashare_sentiment.data.base import DataProvider
from ashare_sentiment.data.baostock_provider import BaoStockProvider
from ashare_sentiment.data.composite_provider import CompositeFreeProvider
from ashare_sentiment.data.eastmoney_provider import EastMoneyProvider
from ashare_sentiment.data.factory import create_provider
from ashare_sentiment.data.tencent_provider import TencentProvider


class FakeJsonClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def get_json(self, url, params, **kwargs):
        self.calls.append((url, dict(params)))
        if "clist" in url:
            return self.payloads["list"]
        if "kline" in url:
            return self.payloads["kline"]
        if "ZTPool" in url:
            return self.payloads["up"]
        return self.payloads["down"]


class FreeProviderTests(unittest.TestCase):
    def test_free_factory_does_not_require_optional_akshare_package(self):
        provider = create_provider({"data": {"provider": "free", "http_cache_root": "data/raw/http"}})
        self.assertIsInstance(provider, CompositeFreeProvider)

    def test_eastmoney_provider_parses_free_json_contract(self):
        client = FakeJsonClient(
            {
                "list": {"rc": 0, "data": {"total": 1, "diff": [{"f12": "000001", "f13": 0, "f14": "测试股", "f26": "20200101"}]}},
                "kline": {"rc": 0, "data": {"klines": ["2026-08-17,10,10.2,10.3,9.9,100,1000,4,2,0.2,1"]}},
                "up": {"rc": 0, "data": {"pool": [{"c": "000001", "m": 0, "n": "测试股", "zdp": 10.0, "amount": 1000}]}},
                "down": {"rc": 0, "data": {"pool": []}},
            }
        )
        provider = EastMoneyProvider(client=client, min_interval=0)
        self.assertIsInstance(provider, DataProvider)
        universe = provider.get_stock_universe()
        self.assertEqual(universe.iloc[0]["ts_code"], "000001.SZ")
        stock = provider.get_stock_daily("2026-08-17", "2026-08-17", "000001.SZ")
        self.assertEqual(stock.iloc[0]["amount_rmb"], 1000)
        limits = provider.get_limit_up_down("2026-08-17", "2026-08-17")
        self.assertTrue(bool(limits.iloc[0]["is_limit_up"]))

    def test_composite_uses_fallback_without_accepting_empty_frame(self):
        class EmptyPrimary(DataProvider):
            name = "primary"

            def _empty(self):
                return pd.DataFrame()

            get_stock_daily = lambda self, *args: self._empty()
            get_index_daily = lambda self, *args: self._empty()
            get_market_breadth = lambda self, *args: self._empty()
            get_limit_up_down = lambda self, *args: self._empty()
            get_margin_data = lambda self, *args: self._empty()
            get_option_data = lambda self, *args: self._empty()

        class Fallback(EmptyPrimary):
            name = "fallback"

            def get_index_daily(self, *args):
                return pd.DataFrame({"trade_date": ["2026-08-17"], "close": [1.0]})

        result = CompositeFreeProvider(primary=EmptyPrimary(), fallback=Fallback()).get_index_daily("2026-08-17", "2026-08-17", "000300.SH")
        self.assertEqual(len(result), 1)

    def test_tencent_provider_parses_public_daily_json(self):
        class Client:
            def get_json(self, url, params, **kwargs):
                self.params = params
                return {
                    "code": 0,
                    "data": {
                        "sh000300": {
                            "day": [["2026-08-17", "10", "10.2", "10.3", "9.9", "100", {}, "2", "1000", "0", "0"]]
                        }
                    },
                }

        provider = TencentProvider(client=Client(), min_interval=0)
        result = provider.get_index_daily("2026-08-17", "2026-08-17", "000300.SH")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["amount_rmb"], 1000)

    def test_baostock_provider_parses_history_and_board_band_limits(self):
        class Result:
            def __init__(self, fields, rows):
                self.error_code = "0"
                self.error_msg = "success"
                self.fields = fields
                self.rows = iter(rows)

            def next(self):
                try:
                    self.current = next(self.rows)
                    return True
                except StopIteration:
                    return False

            def get_row_data(self):
                return self.current

        class Api:
            def query_all_stock(self, day):
                return Result(["code", "tradeStatus", "code_name"], [["sz.000001", "1", "平安银行"], ["sh.000300", "1", "沪深300"]])

            def query_history_k_data_plus(self, code, fields, **kwargs):
                return Result(
                    fields.split(","),
                    [["2026-08-17", code, "10", "11", "9", "11", "10", "100", "1000", "3", "1", "1", "10", "0"]],
                )

        provider = BaoStockProvider(api=Api(), min_interval=0)
        stock = provider.get_stock_daily("2026-08-17", "2026-08-17", "000001.SZ")
        self.assertEqual(stock.iloc[0]["amount_rmb"], 1000)
        limits = provider.get_limit_up_down("2026-08-17", "2026-08-17")
        self.assertTrue(bool(limits.iloc[0]["is_limit_up"]))


if __name__ == "__main__":
    unittest.main()
