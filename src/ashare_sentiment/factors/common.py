"""Shared panel preparation helpers."""

from __future__ import annotations

import pandas as pd


def prepare_stock_panel(
    panel: pd.DataFrame,
    *,
    exclude_recent_ipos_days: int = 0,
) -> pd.DataFrame:
    """Normalize a stock panel and add a conservative eligibility flag.

    ``exclude_recent_ipos_days`` is measured in observed trading rows per
    security.  It is a conservative fallback when a free source has no
    point-in-time IPO calendar; the panel still records the limitation.
    """
    required = {"ts_code", "trade_date", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"stock panel is missing columns: {sorted(missing)}")
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    for column in ("close", "pre_close", "volume", "amount", "amount_rmb", "pct_chg", "is_st"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )
    if "pre_close" not in result.columns:
        result["pre_close"] = result.groupby("ts_code")["close"].shift(1)
    else:
        result["pre_close"] = result["pre_close"].fillna(result.groupby("ts_code")["close"].shift(1))
    result["is_eligible"] = result["close"].gt(0) & result["pre_close"].gt(0)
    if "volume" in result.columns:
        result["is_eligible"] &= result["volume"].fillna(0).gt(0)
    if "name" in result.columns:
        result["is_eligible"] &= ~result["name"].astype(str).str.contains(r"\*?ST", case=False, regex=True)
    if "is_st" in result.columns:
        result["is_eligible"] &= result["is_st"].fillna(0).ne(1)
    if "list_date" in result.columns:
        list_dates = pd.to_datetime(result["list_date"], errors="coerce").dt.normalize()
        result["is_eligible"] &= list_dates.isna() | result["trade_date"].ge(list_dates)
    if "delist_date" in result.columns:
        delist_dates = pd.to_datetime(result["delist_date"], errors="coerce").dt.normalize()
        result["is_eligible"] &= delist_dates.isna() | result["trade_date"].le(delist_dates)
    if exclude_recent_ipos_days > 0:
        observed_age = result.groupby("ts_code", sort=False).cumcount()
        result["recent_ipo_excluded"] = observed_age < int(exclude_recent_ipos_days)
        result["is_eligible"] &= ~result["recent_ipo_excluded"]
    else:
        result["recent_ipo_excluded"] = False
    if "pct_chg" not in result.columns:
        result["pct_chg"] = (result["close"] / result["pre_close"] - 1.0) * 100
    else:
        result["pct_chg"] = result["pct_chg"].fillna((result["close"] / result["pre_close"] - 1.0) * 100)
    return result.reset_index(drop=True)


def date_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.to_datetime(frame["trade_date"].dropna().unique())))
