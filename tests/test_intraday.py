from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ashare_sentiment.application.intraday import IntradaySnapshotService
from ashare_sentiment.data.base import ProviderDataUnavailable
from ashare_sentiment.data.eastmoney_provider import EastMoneyProvider
from ashare_sentiment.data.tencent_provider import TencentProvider


class _SnapshotClient:
    def get_json(self, url, params, *, refresh=False):
        assert refresh is True
        assert "f2" in params["fields"]
        return {
            "rc": 0,
            "data": {
                "total": 2,
                "diff": [
                    {"f2": 10.2, "f3": 2.0, "f5": 100, "f6": 1000, "f8": 1.2, "f12": "000001", "f13": 0, "f14": "平安银行", "f15": 10.3, "f16": 9.8, "f17": 10.0, "f18": 10.0},
                    {"f2": 20.0, "f3": -1.0, "f5": 200, "f6": 3000, "f8": 0.8, "f12": "600000", "f13": 1, "f14": "浦发银行", "f15": 20.4, "f16": 19.8, "f17": 20.2, "f18": 20.2},
                ],
            },
        }


def test_eastmoney_realtime_snapshot_normalizes_full_market_quotes():
    provider = EastMoneyProvider(client=_SnapshotClient(), exclude_st=True)
    frame = provider.get_realtime_snapshot(trade_date="2026-08-19")
    assert frame["ts_code"].tolist() == ["000001.SZ", "600000.SH"]
    assert frame["trade_date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2026-08-19"]
    assert frame["universe_count"].unique().tolist() == [2]
    assert frame["amount_rmb"].sum() == 4000


def test_market_clock_projects_only_during_active_session():
    tz = ZoneInfo("Asia/Shanghai")
    assert IntradaySnapshotService._market_clock(datetime(2026, 8, 19, 10, 30, tzinfo=tz)) == ("OPEN", 0.25)
    assert IntradaySnapshotService._market_clock(datetime(2026, 8, 19, 12, 0, tzinfo=tz)) == ("LUNCH", 0.5)
    assert IntradaySnapshotService._market_clock(datetime(2026, 8, 19, 15, 30, tzinfo=tz)) == ("CLOSED", 1.0)


def test_tencent_realtime_parser_uses_exact_turnover_amount():
    values = [""] * 90
    values[1:7] = ["平安银行", "000001", "11.27", "11.05", "11.08", "1426496"]
    values[30:36] = ["20260819161412", "0.22", "1.99", "11.27", "11.07", "11.27/1426496/1596598790"]
    rows = TencentProvider._parse_realtime_text(f'v_sz000001="{"~".join(values)}";')
    assert rows[0]["ts_code"] == "000001.SZ"
    assert rows[0]["amount_rmb"] == 1596598790
    assert rows[0]["pct_chg"] == 1.99


def test_intraday_rejects_a_stale_exchange_timestamp_during_open_market():
    service = IntradaySnapshotService.__new__(IntradaySnapshotService)
    service.timezone = ZoneInfo("Asia/Shanghai")
    service.max_staleness_seconds = 180
    snapshot = pd.DataFrame({"quote_time": ["20260820100000"]})
    now = datetime(2026, 8, 20, 10, 4, tzinfo=service.timezone)
    with pytest.raises(ProviderDataUnavailable, match="行情已延迟 240 秒"):
        service._validate_quote_freshness(snapshot, now, "OPEN", "tencent-realtime")


def test_intraday_reports_the_exchange_timestamp_not_the_calculation_time():
    service = IntradaySnapshotService.__new__(IntradaySnapshotService)
    service.timezone = ZoneInfo("Asia/Shanghai")
    service.max_staleness_seconds = 180
    snapshot = pd.DataFrame({"quote_time": ["20260820100130", "20260820100155"]})
    now = datetime(2026, 8, 20, 10, 2, tzinfo=service.timezone)
    result = service._validate_quote_freshness(snapshot, now, "OPEN", "tencent-realtime")
    assert result.isoformat(timespec="seconds") == "2026-08-20T10:01:55+08:00"
