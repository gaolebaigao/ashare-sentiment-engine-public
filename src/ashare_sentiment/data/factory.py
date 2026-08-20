"""Create a provider from project configuration."""

from __future__ import annotations

import os
from typing import Any, Mapping

from .akshare_provider import AkShareProvider
from .baostock_provider import BaoStockProvider
from .base import DataProvider, ProviderDataUnavailable
from .composite_provider import CompositeFreeProvider
from .eastmoney_provider import EastMoneyProvider
from .tencent_provider import TencentProvider
from .tushare_provider import TushareProvider


def create_provider(config: Mapping[str, Any]) -> DataProvider:
    data = config.get("data", {})
    provider_name = str(data.get("provider", "tushare")).lower()
    provider_kwargs = {
        "cache_root": data.get("http_cache_root", "data/raw/http"),
        "timeout": float(data.get("request_timeout_seconds", 15)),
        "min_interval": float(data.get("request_min_interval_seconds", 0.20)),
        "retries": int(data.get("request_retries", 3)),
        "include_beijing": bool(data.get("include_beijing", False)),
        "exclude_st": bool(data.get("exclude_st", True)),
        "max_symbols": data.get("max_symbols"),
    }
    http_kwargs = {
        "cache_root": provider_kwargs["cache_root"],
        "timeout": provider_kwargs["timeout"],
        "min_interval": provider_kwargs["min_interval"],
        "retries": provider_kwargs["retries"],
    }
    if provider_name in {"free", "free-composite", "composite-free"}:
        return CompositeFreeProvider(
            primary=EastMoneyProvider(**provider_kwargs),
            fallbacks=[TencentProvider(**http_kwargs)],
        )
    if provider_name in {"eastmoney", "eastmoney-free"}:
        return EastMoneyProvider(**provider_kwargs)
    if provider_name in {"tencent", "tencent-free"}:
        return TencentProvider(**http_kwargs)
    if provider_name in {"baostock", "bao-stock", "bao_stock"}:
        return BaoStockProvider(
            include_beijing=bool(data.get("include_beijing", False)),
            exclude_st=bool(data.get("exclude_st", True)),
            max_symbols=data.get("max_symbols"),
            min_interval=float(data.get("request_min_interval_seconds", 0.20)),
            query_timeout_seconds=float(data.get("request_timeout_seconds", 15.0)),
            progress_every=int(data.get("progress_every_symbols", 100)),
            workers=int(data.get("download_workers", 1)),
        )
    if provider_name == "tushare":
        return TushareProvider(
            token=os.getenv("TUSHARE_TOKEN"),
            endpoint=str(data.get("tushare_endpoint", "https://api.tushare.pro")),
            min_interval=max(float(data.get("request_min_interval_seconds", 0.20)), 0.20),
            retries=int(data.get("request_retries", 3)),
            checkpoint_root=data.get("tushare_checkpoint_root", "data/raw/tushare_checkpoints"),
            include_beijing=bool(data.get("include_beijing", False)),
            exclude_st=bool(data.get("exclude_st", True)),
            include_delisted=bool(data.get("tushare_include_delisted", False)),
            max_symbols=data.get("max_symbols"),
            progress_every=int(data.get("progress_every_symbols", 50)),
        )
    if provider_name == "akshare":
        return AkShareProvider()
    raise ProviderDataUnavailable(f"Unknown data provider: {provider_name}")
