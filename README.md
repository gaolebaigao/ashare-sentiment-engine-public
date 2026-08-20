# A-Share Sentiment Timing Engine

一个面向 A 股的市场情绪温度计、市场状态 Regime Engine 和拐点研究框架。项目目标是回答“市场有多冷/多热、情绪是否正在转向、当前是否值得开始关注或减少追高”，服务人工投资决策辅助，而不是预测明天涨跌或机械寻找绝对顶部/底部。

当前稳定数据层为 `MarketTemperature v0.2.1`：全市场面板下载、expected-vs-observed 覆盖率、逐日生产数据门禁、真实 LimitUp/LimitDown 状态、退化因子检测和研究诊断图。`v0.1` 与 `v0.2` 仍保留为审计基线。

## MarketTemperature v0.3 状态机

v0.3 在 v0.2.1 数据合同之上增加独立的 `src/ashare_sentiment/regime/` 层：

```text
Raw Temperature
  -> causal EMA(span=3)
  -> Slope1 / Slope3
  -> valid-observation RollingHigh10 / RollingLow10
  -> RecoveryFromLow / DropFromHigh
  -> deterministic State Machine
```

状态机只使用 `SmoothedTemperature` 做核心状态判断，同时保留 Raw Temperature 和 `TemperatureShock` 作为诊断。`EUPHORIA_RISING` 不等于卖出，`PANIC_FALLING` 不等于冰点反转；`ICE_REVERSAL` 与 `HOT_ROLLOVER` 均需要斜率、滚动极值距离和至少 3/4 模块确认，其中 Breadth 或 Profit Effect 必须参与确认。

EMA、Slope、RollingHigh 和 RollingLow 使用有效的 MarketTemperature 观测序列。`market_temperature_quality == INVALID` 的日期不会被 forward fill、interpolate 或 backfill：该日 Raw/Smoothed/派生温度字段为 `NaN`，State 为 `DATA_INVALID`，Signal 为 `NONE`，并重置确认 streak。INVALID 后的下一有效日可以从上一个有效 EMA 继续计算，但确认必须重新从 WATCH 开始。

`config/default.yaml` 中的 `state_machine` 参数是 research defaults，不是历史收益优化出来的交易参数。本轮不实现 Options、QVIX、IV、收益 Event Study、Portfolio Backtest、Tactical Overlay、Stock Heat 或 Dashboard。

状态优先级固定为：`DATA_INVALID` → `ICE_REVERSAL` → `HOT_ROLLOVER` → `ICE_REVERSAL_WATCH` → `HOT_ROLLOVER_WATCH` → `PANIC_FALLING` → `EUPHORIA_RISING` → `EXTREME_PANIC` → `COLD` → `HOT` → `NORMAL`。同一有效观测满足多个候选条件时按此顺序确定，确认信号与描述性温度状态分开保存。

生成 v0.3 daily state table、统计、敏感性分析、七月案例和研究图：

```bash
PYTHONPATH=src python research/market_regime_v03.py
python -m ashare_sentiment regime
python -m ashare_sentiment regime --date 2026-07-17
```

主要产物为 `data/processed/market_state_daily.parquet`、三张 `reports/state_machine_v03_*.png`、`reports/july_2026_state_machine_v03.md`、状态/信号统计和 `reports/state_machine_sensitivity_v03.csv`。敏感性只比较状态频率、持续时间、拐点日期稳定性和 WATCH-to-CONFIRM，不使用未来收益。

## MarketTemperature v0.3.1 Episode Anchoring & State Stability

v0.3.1 在 v0.3 的因果 EMA 和状态机上增加了极端事件锚点与生命周期记忆，用来避免“Raw Temperature 很冷但 EMA 尚未跌到阈值”时丢失尖锐恐慌，也避免单日 Raw spike 在没有内部模块确认时自动触发状态。默认锚点为：Raw panic `<=20` 且至少 2/4 模块 `<=20`；Raw euphoria `>=85` 且至少 2/4 模块 `>=80`。锚点在最近 10 个有效观测内保持记忆，连续锚点属于同一个 episode；确认、超时、重新锚定或 INVALID 会关闭/重置 episode。

`ICE_REVERSAL` 现在必须同时满足：最近存在 panic episode、相对于 episode 内最低有效 EMA 有足够恢复、Slope3 转正、至少 3/4 模块改善，并且 Breadth 或 Profit Effect 改善。`HOT_ROLLOVER` 使用对称的 euphoria episode、episode 高点回落、负 Slope3 和内部恶化确认。`EUPHORIA_RISING` 不等于卖出，`PANIC_FALLING` 不等于冰点反转。

EMA、Slope、RollingHigh/RollingLow 与 episode memory 只沿有效 MarketTemperature 观测推进。INVALID 日期保持 Raw/Smoothed/派生温度为 `NaN`、State 为 `DATA_INVALID`、Signal 为 `NONE`，并重置确认 streak 和活动 episode；INVALID 之后可以从之前的有效 EMA 继续，但不能跨日直接完成 WATCH 确认。状态机仅对 HOT/NORMAL/COLD 使用默认 `zone_hysteresis: 3`，不对 ICE/HOT turning-point confirmation 添加隐含 hysteresis。

v0.3.1 生成独立产物，不覆盖 v0.3 基线：

```bash
PYTHONPATH=src python research/market_regime_v031.py
python -m ashare_sentiment regime
python -m ashare_sentiment regime --date 2026-07-17
```

主要产物包括 `data/processed/market_state_daily_v031.parquet`、`reports/july_2026_state_machine_v031.md`、`reports/v03_vs_v031_state_comparison.md`、`reports/episode_statistics_v031.csv`、`reports/whipsaw_diagnostics_v031.csv`、`reports/state_machine_sensitivity_v031.csv` 和三张 `reports/state_machine_v031_*.png`。敏感性只比较 episode 数量、状态持续时间、信号日期稳定性、WATCH-to-CONFIRM 与 zone whipsaw，不使用未来收益。

v0.3.1 的参数仍是 research defaults，不是历史收益最优参数。本轮不实现 Event Study、forward returns、Portfolio Backtest、Tactical Overlay 或交易提醒；完成稳定性审查后才进入 v0.4 Event Study。

## MarketTemperature v0.4.1 Advisory Signal Layer

v0.4.1 在冻结的 v0.3.1 状态机之上增加独立的 Advisory Layer：

```text
MarketTemperature
  -> Frozen v0.3.1 State Machine / Episode
  -> Advisory Layer
  -> Human-readable market guidance
```

这是一个市场情绪决策支持系统，不是自动交易系统、组合策略、仓位管理引擎、执行引擎或券商接口。`BUY_REFERENCE` / `SELL_REFERENCE` 是市场环境参考，不是必须执行的买卖指令；`PANIC_WAIT` 和 `HOT_CAUTION` 明确表达“等待”和“谨慎追高”。特别地，`PANIC_FALLING` 不会映射成 `BUY_REFERENCE`，`EUPHORIA_RISING` 也不会映射成 `SELL_REFERENCE`。

当前 Advisory Signal 为：

```text
PANIC_WAIT / BUY_WATCH / BUY_REFERENCE / NEUTRAL
HOT_CAUTION / SELL_WATCH / SELL_REFERENCE
DATA_INVALID
```

`AdvisorySignal` 是唯一的最高层人工参考字段。`Buy Reference` / `Sell Reference` 只是方向性辅助强弱，不构成第二套信号词汇；`Signal Confidence` 与 `Research Evidence` 也严格分离。`DATA_INVALID` 是硬阻断，不生成正常市场建议。

`SignalConfidence` 与 `ResearchEvidence` 分开：前者回答“今天状态识别是否清楚”，后者回答“历史上该类状态是否存在稳定统计规律”。目前 ICE 的短期历史证据为 `WEAK`、中期为 `MODERATE`，更适合作为 20–60 个交易观测的中期恢复参考，而不是精确底部或短线买点；HOT 更适合作为数周至数月的中期风险提醒，而不是精确顶部。历史样本数仍有限。

生成最新或指定日期的 Advisory：

```bash
PYTHONPATH=src python -m ashare_sentiment advisory
PYTHONPATH=src python -m ashare_sentiment advisory --date 2026-07-23
```

命令只使用当日及此前的状态和诊断字段，不读取 Event Study 的 future returns 或 outcome。主要产物为 `data/processed/market_advisory_daily_v041.parquet`、`reports/advisory_history_v041.csv`、`reports/current_market_advisory_v041.png`、`reports/advisory_timeline_v041_2026.png`、`reports/july_2026_advisory_v041.md` 和 `reports/advisory_design_v041.md`。

## MarketTemperature Local Web App

本地浏览器版是 Advisory Layer 的 localhost 视图。交易日 09:30 后，Overview 会优先通过东方财富全市场快照、失败时自动切换腾讯全市场批量行情，生成非持久化的盘中估算，并默认每 60 秒检查更新；盘中累计成交额按已完成交易时段折算，页面会明确标记“实时 / 盘中估算”。正式 History 仍只保存完整收盘数据，避免用未收盘快照污染日线研究结果。实时源不可用时，页面会显示原因并安全回退到最近一次有效缓存；其他页面的 Refresh 仍复用 `update → score → regime → advisory` 收盘流水线。

界面包含 Overview、History、Episodes、Diagnostics 和 Settings：

- Overview 直接展示最新有效交易日、Raw / Smoothed Temperature、状态、唯一 Advisory Signal、风险、观察期限、信号置信度、历史证据、Why、What To Watch Next 和四个内部模块。
- History 支持日期选择、前后交易日跳转和 3M / 6M / 1Y / All 趋势范围。
- Episodes 只展示 Panic / Euphoria 的开始、极值、观察、确认、结束和状态序列，不展示未来收益、CAGR、Sharpe 或最佳入场点。
- DATA_INVALID 为硬阻断：显示数据不可用、原因和恢复提示，不显示正常 Buy / Sell Reference。
- Overview 只显示 `Data Notes: N`，点击后查看每条提示的含义及是否影响正常建议；模型参数不在 Settings 暴露。

启动本地 Web 服务并自动打开浏览器：

```bash
python -m ashare_sentiment web
```

默认地址为 `http://127.0.0.1:8765`；也可以使用兼容别名：

```bash
python -m ashare_sentiment gui
```

不自动打开浏览器：

```bash
python -m ashare_sentiment web --no-open --port 8765
```

仍可生成无服务启动的视觉审核快照：

```bash
python -m ashare_sentiment web --snapshot
```

截图输出为 `reports/gui_overview_light.png`、`reports/gui_overview_dark.png`、`reports/gui_history_july_2026.png`、`reports/gui_episodes.png` 和 `reports/gui_diagnostics.png`。服务只监听本机，不上传本地数据，也不需要登录或云端服务。

## 设计边界

`MarketTemperature v0.2` 仍使用四个模块，不在本轮调权重：

```text
0.30 Breadth
0.25 Profit Effect
0.15 Liquidity
0.15 Stretch
Options unavailable
```

Options 不用 50 分填充，而是在四个可用模块之间重新归一化。权重已经放在 [`config/default.yaml`](config/default.yaml)，不在研究代码中硬编码。所有历史 percentile 必须只使用当日及之前的数据；未来数据、centered rolling、backward fill 和按未来表现重定义历史状态都禁止。

## 安装

建议使用 Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,dev,research]"
cp .env.example .env
```

默认 provider 是 Tushare：把 `TUSHARE_TOKEN` 只放在本地 `.env`；如需使用兼容服务，可通过本地 `TUSHARE_ENDPOINT` 覆盖请求地址。适配器按交易日串行下载全市场，最小请求间隔强制为 0.2 秒，并在 `data/raw/tushare_checkpoints/` 保存可恢复的断点。BaoStock、东方财富、腾讯和 AkShare 仍保留为可选免费适配器。

公开仓库不包含任何真实 token、个人配置、本地缓存、原始/处理后数据或生成报告。运行数据更新和研究命令后，这些内容会在本地的 `data/` 和 `reports/` 目录生成；请勿将包含凭据的 `.env` 提交到 Git。

桌面和 Web 集成测试依赖本地生成的研究产物；在干净克隆中会自动跳过，完成数据更新、状态机和 Advisory 流程后即可运行。

## MarketTemperature v0.2.1 命令

首次运行建议下载足够长的历史数据，再通过 Parquet/HTTP 缓存做增量更新：

```bash
python -m ashare_sentiment update --provider tushare --dataset full-market --start-date 2024-01-01 --end-date 2026-08-17

# 分别请求 Tushare limit_list_d 的 U/D 涨跌停池；权限失败才尝试 stk_limit 推导
python -m ashare_sentiment update --provider tushare --dataset limit-up-down --start-date 2024-01-01 --end-date 2026-08-17
```

下载指数或单只股票：

```bash
python -m ashare_sentiment update --dataset index --symbol 000300.SH --start-date 2024-01-01 --end-date 2025-12-31
python -m ashare_sentiment update --dataset stock --symbol 600519.SH --start-date 2024-01-01 --end-date 2025-12-31
```

计算最新或历史某一天：

```bash
python -m ashare_sentiment score
python -m ashare_sentiment score --date 2026-07-20

# 仅用于开发诊断；结果会标为 INVALID，不是交易用温度
python -m ashare_sentiment score --allow-partial-data
```

可选数据集：`limit-up-down`、`margin`、`options`。如果 provider 没有可靠覆盖，会明确报错，不会以空值或伪造数据冒充成功。

数据和元数据位于：

```text
data/raw/
data/processed/
data/cache/*.parquet
data/cache/*.metadata.json
```

生成 MarketTemperature v0.2.1 全量 CSV、覆盖率/完整性 CSV、LimitDown 诊断和研究图：

```bash
PYTHONPATH=src python research/market_temperature_v021.py --start-date 2026-01-01 --end-date 2026-08-18
```

生成 `reports/market_temperature_v021_2026.csv`、`reports/data_coverage_v021_2026.csv`、`reports/limit_status_diagnostics_v021.csv`、七月案例和三张 v0.2.1 PNG。完整性失败日会保留审计行，但生产温度为 `NaN`、质量为 `INVALID`。

```bash
PYTHONPATH=src python research/market_temperature_v02.py
```

输出：

```text
data/processed/full_market_daily.parquet
data/processed/market_coverage_daily.parquet
data/processed/market_sentiment_daily.parquet
reports/market_temperature_v02_2026.csv
reports/market_temperature_v02_2026.png
reports/market_temperature_v02_modules_2026.png
reports/breadth_diagnostics_2026.png
reports/profit_effect_diagnostics_2026.png
reports/liquidity_diagnostics_2026.png
reports/data_coverage_2026.csv
```

全市场历史数据按交易日逐个请求，首次 Tushare 下载可能需要较长时间；`--max-symbols` 仅用于 smoke test，不用于正式结果。适配器会先查 `stock_basic`，再对每个交易日调用一次带日期范围的 `daily`，每个交易日完成后保存 CSV 断点；重复运行会从已完成文件继续。正式 `score` 要求默认至少 3,000 只有效股票且每日价格覆盖率至少 90%；不足时以非零状态失败。`--allow-partial-data` 只允许研究诊断，并明确标记 `INVALID`。

## 数据源

数据源、权限、覆盖范围和风险记录在 [`DATA_SOURCES.md`](DATA_SOURCES.md)。当前选择和 V0.2 审计状态是：

- BaoStock：默认无 Token 数据源，提供匿名历史日线、当前股票快照和基本资料；适合全市场 Breadth、Liquidity 和 Stretch。其逐股票历史查询可以形成全市场面板，但当前 universe 仍有生存者偏差风险。

- 东方财富公开 JSON：配置为涨跌停池的免费来源，股票/指数历史 K 线、股票列表、涨停池和跌停池均无需 token；若当前网络不可达，V0.2 生产路径不会静默回退到 BaoStock 近似。
- 腾讯公开 JSON：东方财富 K 线不可达时的历史股票/指数 K 线 fallback；接口按证券逐个请求，当前适配器使用稳定的最多约 2,000 行窗口。
- AkShare：免费 fallback 与探索适配器；它聚合公开网页接口，因此字段与稳定性仍需持续验证。
- Tushare Pro 兼容接口：当前默认数据源；`daily`、`index_daily`、`stock_basic`、`trade_cal`、`limit_list_d` 和 `stk_limit` 已接入，密钥只从环境变量读取。当前 universe 仍以现行 `stock_basic` 为准，生存者偏差警告保留。
- 上交所、深交所、中金所：作为官方数据源和后续授权/直连接口的校验基准。

## 数据质量与回测风险

缓存写入前会检查重复 key、无法解析日期、非正价格、负成交量和异常收益。周末/节假日不被误判为缺失交易日；交易日历检查会在接入统一交易日历后启用。

V0.1 是 prototype：曾经使用 10-stock partial panel，不能用于交易。V0.2 默认拒绝这种输入。最重要的剩余限制是 point-in-time universe：免费股票列表不能完全还原历史当日真实股票集合。因此 breadth 与报告必须保留 `SURVIVORSHIP_BIAS_WARNING`，不能宣称无生存者偏差。

另外，BaoStock 路径不使用统一的 `return >= 9.8%` 规则，而按股票代码所属板块使用 10%/20%/30% 区间并记录 `LIMIT_STATUS_APPROXIMATE_BAOSTOCK_BOARD_BANDS`；该近似不被 V0.2 正式 Profit Effect 接受。`FailedLimitRate`、Margin 和 Options 仍保持缺失，不用 0 或 50 伪造。

## 测试

```bash
pytest
```

如果尚未安装开发依赖，也可以直接运行标准库测试：

```bash
python -m unittest discover -s tests -v
```

新增测试覆盖生产 partial-panel gate、full-panel acceptance、退化因子、缺失不等于 50、覆盖率报告和 no-lookahead；v0.3 进一步覆盖因果 EMA、有效观测窗口、状态优先级、ICE/HOT WATCH 确认、INVALID gap、timeout、falling knife、Euphoria 和确定性。

本地 Web 端还覆盖最新 HOT → `HOT_CAUTION`、`DATA_INVALID` 无 Buy/Sell Reference、2026-07-17 / 2026-07-23 / 2026-07-24 历史查询、Episode 生命周期、Warning Center 和核心 Advisory hash 防漂移；验证结果见 [`reports/web_validation_v041.yaml`](reports/web_validation_v041.yaml)。

## 后续开发顺序

1. 验证 point-in-time universe 与免费数据授权/稳定性。
2. 接入经明确合同选择和缺失处理的 Options / IV / Put-Call 层。
3. 使用 v0.3 状态机结果设计不含收益反向调参的 Event Study。
4. 实现 next-open、commission、slippage 的回测与 event study。
5. 加入 Tactical Overlay、Stock Heat 和 Streamlit Dashboard。

本项目是市场状态与风险管理工具，不保证预测顶部或底部，也不构成投资建议。
