# Data sources and availability (V0.2)

调查日期：2026-08-18。这里区分三件事：接口是否存在、账户是否有权限、数据是否适合 point-in-time 研究。只有第一件事成立，并不等于可以直接用于无偏回测。

## V0.2 Full-Market Free Data Audit

本轮重新做了小规模实际连接测试。BaoStock 在 2026-08-18 的独立测试中匿名登录成功：`query_all_stock(day="2026-08-17")` 返回 7,344 条原始证券记录，`query_stock_basic()` 返回 8,903 条基本资料记录；项目 Provider 使用其中的 SSE/SZSE A 股过滤结果，并尝试合并 `ipoDate` / `outDate`。BaoStock 历史日线仍是逐股票查询，因此完整 2024+ 面板需要长时间下载。

东方财富公开 JSON 在当前运行环境的重复测试遇到代理端 `RemoteDisconnected`，因此不能把它的历史涨跌停池说成当前已验证可用。AkShare 未安装，本轮仅核对其公开文档，不把文档存在当成实时可用性证明。上交所/深交所公开页面可以作为官方校验入口，但未发现可直接替代免费批量历史 API 的稳定无授权下载路径。

| Candidate | Actual test | Free / auth | Historical / batch | V0.2 decision |
| --- | --- | --- | --- | --- |
| BaoStock | Anonymous login and universe/basic probes passed; project full-panel run is symbol-oriented and long-running | Free; anonymous login | Daily history; one security per request; current snapshot plus basic IPO/delist fields | Selected for full-market OHLCV, Breadth and Liquidity |
| Eastmoney public JSON | Current repeated Python probe failed at proxy connection; prior fixture/parser tests pass | Free; no token | Daily K-line and date-based limit pools; one symbol/day request pattern | Selected as the preferred limit-pool source only when live download succeeds; no silent fallback |
| AkShare | Not installed and no live call completed in this run | Free wrapper; no project token | Symbol-level history and date-based pool methods documented | Optional adapter; not evidence for production availability |
| SSE / SZSE public pages | Public pages reachable in web research; bulk historical automation/licensing not verified | Public pages; data-service terms apply | Official market-data products exist, but not a verified free bulk API here | Cross-check/reference only |

Production consequence: if BaoStock full-panel download or the configured Eastmoney limit-pool download cannot complete, V0.2 stops with an explicit error. It must not use the previous 10-stock cache or BaoStock approximate board bands as a formal Profit Effect input.

## Free Data Research 2026

本轮实际测试了无需登录、无需 token 的东方财富公开 JSON 接口。测试时间为 2026-08-17，使用保守的单次 HTTP 请求、标准 User-Agent 和 15 秒 timeout；没有绕过登录、验证码、付费墙或访问控制。

| Dataset | Candidate Source | Endpoint / Library | Free? | Login? | Historical? | Frequency | Coverage | Stability | Current Test Result | Chosen? | Fallback | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stock OHLCV | 东方财富公开 JSON | `push2his.eastmoney.com/api/qt/stock/kline/get` | Yes | No | Yes | Daily | Symbol-level SSE/SZSE | Medium | Stock-specific live probe was blocked by an intermittent network/proxy failure in the final verification attempt; endpoint schema is documented and parser is covered by fixture tests | Yes, pending live retry | AkShare adapter | One request per symbol; full-universe history is slow and provider-derived |
| Index OHLCV | 东方财富公开 JSON | Same K-line endpoint, e.g. `secid=1.000300` | Yes | No | Yes | Daily | 沪深300 and compatible index secids | Medium | HTTP 200 / `rc=0`, 635 rows for 2024-01-01 to 2026-08-17 in a successful probe; later requests were intermittently blocked by the network path | Yes | AkShare | Index secid mapping must be validated per benchmark |
| Stock/index OHLCV fallback | 腾讯公开 JSON | `proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get` | Yes | No | Yes | Daily | Symbol-level `.SH` / `.SZ` history | Medium | Verified live on 2026-08-18 for 沪深300, 平安银行 and 创业板指; a 2,000-row request returned 2018-05-23 to 2026-08-18, with 2021-01-01 onward cached for four configured benchmarks | Yes | None required | Symbol-oriented only; does not provide the full breadth panel or limit-up/down pool |
| Stock/index OHLCV and universe | BaoStock | `baostock` Python package / `query_history_k_data_plus` | Yes | Anonymous login | Yes | Daily | SSE/SZSE stocks and indices | Medium | Verified live on 2026-08-18: anonymous login succeeded; `sz.000001` and `sh.000300` returned 2026-08-03 to 2026-08-17 history; `query_all_stock` returned the current security list | Yes, now default | None required | Current-universe snapshot; full panel is symbol-oriented and limit-up/down is board-band approximation |
| Stock universe | 东方财富公开 JSON | `push2.eastmoney.com/api/qt/clist/get` | Yes | No | Current snapshot | Intraday snapshot | SSE/SZSE A-share filters | Medium | HTTP 200 / `rc=0`, `total=5549`, 5 rows returned in the smoke request | Yes | AkShare stock list | Current universe is not a historical point-in-time universe |
| Limit-up pool | 东方财富公开 JSON | `push2ex.eastmoney.com/getTopicZTPool` | Yes | No | Daily historical-by-date endpoint | Daily | Limit-up pool | Medium | HTTP 200 / `rc=0`, 5 rows returned for 2026-08-17 with `pool` records | Yes | AkShare `stock_zt_pool_em` | Requires one request per date; not a substitute for intraday touched-limit history |
| Limit-down pool | 东方财富公开 JSON | `push2ex.eastmoney.com/getTopicDTPool` | Yes | No | Daily historical-by-date endpoint | Daily | Limit-down pool | Medium | HTTP 200 / `rc=0`, 1 record reported for 2026-08-17 | Yes | AkShare `stock_zt_pool_dtgc_em` | Pool definition and historical coverage should be rechecked |
| Failed-limit rate | 东方财富 / AkShare炸板池 | `stock_zt_pool_zbgc_em` candidate | Yes | No | Candidate historical-by-date | Daily | Failed-limit pool candidate | Not verified | Not used in this round; no reliable field-level probe was completed | No | None | Never filled with zero; remains unavailable |
| Market turnover | Sum of stock OHLCV amount | Derived from stock K-line `amount` | Yes | No | Yes if stock panel exists | Daily | Eligible SSE/SZSE stocks | Depends on stock panel | Amount field is available in K-line payload; provider normalizes it to RMB | Yes | AkShare stock history | Thousands of symbol requests for broad historical coverage |
| Margin financing | SSE/SZSE public data candidates | Exchange pages / AkShare candidates | Yes | No | Partial/candidate | Daily | Exchange-specific | Not verified | No stable no-key full-history path was verified in this round | Optional | None | Omitted from v0.1 rather than blocking or fabricating |

The default OHLCV path selected in code is `TushareProvider`: it uses the compatible endpoint described in the supplied guide, keeps the token in the environment, serializes requests at a minimum 0.2 second interval and saves per-trading-date checkpoints. Market breadth is built from `daily` plus `stock_basic` metadata; Tushare amount values are normalized from thousand RMB to RMB in `amount_rmb`. V0.2.1 does not accept a partial day as a valid production score: expected eligible universe, observed valid universe, coverage collapse, cross-module count deviation and impossible ratios are audited per date. `limit_list_d` is queried separately for `limit_type=U` and `limit_type=D`, with `stk_limit` plus `daily` as a permission-aware fallback; every cached row carries source/status metadata so a real zero is not confused with unavailable data. The previous `CompositeFreeProvider` and BaoStock path remain available for public/free fallback tests. Final normalized data remains Parquet with metadata; failed provider calls raise `ProviderDataUnavailable`, and an empty or non-atomic fallback frame is never treated as a valid factor input.

## Recommended source map

| Engine dataset | Preferred source | Current adapter | Availability / permission | Research caveat |
| --- | --- | --- | --- | --- |
| Stock daily OHLCV | Tushare Pro `daily` | Implemented | Historical daily data; account points and rate limits apply | Raw/unadjusted vs adjusted prices must be chosen explicitly for each factor |
| Index daily OHLCV | Tushare Pro `index_daily` | Implemented | Historical index data; index endpoint has its own permission | Index codes and return conventions must be kept in metadata |
| Breadth panel | Tushare `daily` + `stock_basic` | Implemented as resumable per-symbol panel | `daily` is queried by `ts_code`; current compatible endpoint may truncate all-symbol date queries | Current listed universe is not point-in-time; survivorship warning is mandatory |
| Limit-up/down prices | Tushare `limit_list_d`, fallback `stk_limit` + `daily` | Implemented | `limit_list_d`/`stk_limit` permissions vary by account | Price-limit regime changes, ST and IPO rules need provider fields; do not use a universal 9.8% threshold |
| Failed-limit / intraday limit ecosystem | Tushare `kpl_list` or an authorized intraday source | Not yet wired | Higher permission tier for `kpl_list`; intraday historical coverage must be checked | Required for robust `FailedLimitRate`; V0.1 does not fabricate this factor |
| Margin financing | Tushare `margin` / `margin_detail` | Implemented | Margin data has historical coverage but is permissioned | Aggregate fields and stock-level fields are different datasets and must not be mixed |
| Option daily observations | Tushare option daily endpoint | Implemented as raw optional adapter | Requires option data access | Raw option quotes are not automatically IV300/IV1000/PCR; contract selection and derivation come later |
| Official exchange cross-check | SSE / SZSE market data services | Not yet wired | Distribution and licensing conditions apply | Good validation/authorization baseline, but not assumed to be a free bulk API |
| CFFEX index options | CFFEX official product/data pages | Not yet wired | Contract and market-data access must be confirmed | Useful for CSI 300 / CSI 1000 option universe definition; official data is preferred where accessible |

## Source notes and links

### Tushare Pro

Tushare's own interface catalog lists stock daily data, index daily data, daily indicators, margin data, daily limit prices and option daily data. The platform also documents per-endpoint point thresholds and request limits, so the adapter treats permission failures as `ProviderDataUnavailable` rather than returning misleading partial data.

- [Tushare data and permission overview](https://tushare.pro/document/1?doc_id=108)
- [Tushare market data overview](https://tushare.pro/document/1?doc_id=15)
- [`daily_basic` daily indicators](https://tushare.pro/document/2?doc_id=32)
- [`stk_limit` daily limit prices](https://tushare.pro/document/2?doc_id=183)
- [`margin_detail` margin financing detail](https://tushare.pro/document/2?doc_id=59)
- [Tushare option data catalog](https://tushare.pro/document/2?doc_id=157)
- [`kpl_list` limit-up/down and failed-limit board data](https://tushare.pro/document/2?doc_id=347)

### AkShare

AkShare documents many public-web adapters and is useful as an exploratory fallback. Its source endpoints, fields and availability can change, and a real-time/spot endpoint is not automatically a historical point-in-time panel. The project therefore only exposes the simple symbol-oriented stock and index paths in V0.1.

- [AkShare stock data documentation](https://akshare.akfamily.xyz/data/stock/stock.html)
- [AkShare option data documentation](https://akshare.akfamily.xyz/data/option/option.html)

### Eastmoney public JSON endpoints used by the free adapter

- [Historical K-line endpoint](https://push2his.eastmoney.com/api/qt/stock/kline/get)
- [A-share list endpoint](https://push2.eastmoney.com/api/qt/clist/get)
- [Limit-up pool endpoint](https://push2ex.eastmoney.com/getTopicZTPool)
- [Limit-down pool endpoint](https://push2ex.eastmoney.com/getTopicDTPool)

### Official exchange sources

- [Shanghai Stock Exchange historical daily stock data](https://www.sse.com.cn/market/stockdata/overview/day/index_his.shtml)
- [Shenzhen Stock Exchange market data services](https://investor.szse.cn/English/services/dataServices/index.html)
- [CFFEX CSI 300 index options](https://www.cffex.com.cn/cn/hs300gzqq.html)

## What is deliberately not claimed yet

1. No free source in this repository is currently declared sufficient for fully point-in-time A-share universe reconstruction from 2017 onward.
2. No option source is currently declared sufficient for a ready-made CSI 300/CSI 1000 IV percentile or put-call ratio series; those require an explicit contract universe, settlement/quote conventions and missing-date policy.
3. The adapter does not use future data to fill an unavailable history, and it does not treat a failed provider request as a zero factor.
4. Official exchange feeds may be the correct production choice, but licensing, access and bulk-download terms must be resolved before wiring them into automated research.
