# Macro Alpha Pro — Master Backlog（整合版 v2）

> **作者视角**：资深量化金融分析师 + 严谨学术标准  
> **整合来源**：project_report / project_roadmap / trading_agent_pivot / spec_improvement_plan / spec_simulated_execution / 4.21.pdf（FactorMAD + 宇宙扩展 + 编排层 + 新闻升级）  
> **更新日期**：2026-04-26 v3  
> **原则**：已完成项不重复开发；被移除项附理由；冲突显式解决；所有待办以实证价值排序

---

## 零、文档间冲突裁定

| 冲突 | 来源 | 裁定 |
|------|------|------|
| roadmap P4 建议引入 HMM | project_roadmap.md | **已废除** — `engine/regime.py` 已实现 Hamilton MSM |
| pivot 双轨架构 / Markov Switching | trading_agent_pivot.md | **已实现** |
| spec_simulated_execution portfolio_monitor.py | spec_simulated_execution.md | **已废除** — UI 并入 live_dashboard.py |
| roadmap Brier Score / 二项检验 | project_roadmap.md §P0 | 未覆盖，**归入 P2-7** |
| pivot BHY 多重检验校正 | trading_agent_pivot.md | 未覆盖，**归入 P1-5** |
| roadmap Prompt 版本管理 | project_roadmap.md §P2 | 未覆盖，**归入 P2-8** |
| roadmap γ/λ 框架 | project_roadmap.md §P3 | 仍为 stub，**归入 Defer-6** |
| QA-2 权重重设计 vs P2-B TSMOM 连续化 | spec 内部 | 顺序依赖，QA-2 先于 P2-B |
| P1-3 risk_conditions vs P1-4 连续权重 | spec 内部 | 顺序依赖，P1-3 先于 P1-4 |
| **4.21.pdf "量化先验锚"提案** | 4.21.pdf 方向四 | **废除** — 向 prompt 注入 `tsmom_signal=+1` 和 `composite_score=75` 等方向性结论，直接违反 P0-4 确立的注入规则，破坏双轨独立性，引入锚定偏差 |
| **4.21.pdf Phase 1 GICS 11 行业标准化** | 4.21.pdf §1.2 | **修正** — GICS 11 将宇宙从 18 缩减到 11，损失截面分散度，方向反了；改为保留现有 18 个 ETF，向上扩展 |
| **4.21.pdf Track B 四层架构** | 4.21.pdf §二 | **归入 Defer-9** — 框架合理，但 BL 融合层依赖未校准的 LLM 置信度；等 Phase 2 宇宙扩展 + Clean Zone n≥30 后实施 |

---

## 一、已完成模块（不重复开发）

### 基础设施与 UI（截至 2026-04-19）
- 5 组 st.navigation() 导航；Admin 4-Tab 工作流
- LangGraph 研究 Agent 图（researcher→red_team→auditor→translator/reflection）
- SQLite ORM 全表（17 张表）；知识反馈闭环：known_failures 注入 prompt + SkillLibrary 自动更新

### 量化引擎（截至 2026-04-19）
- `engine/signal.py`：TSMOM（12-1月动量）、CSMOM 截面排序、年化波动率，skip_months=1 无前视
- `engine/regime.py`：Hamilton MSM，filtered probability only
- `engine/portfolio.py`：vol-parity 权重 + 制度条件约束 + ρ=0.5 上界警告
- `engine/quant_agent.py`：QuantAssessment，复合评分统一，ATR 批量下载，期限匹配
- `engine/daily_batch.py`：6 步状态机，幂等，盘中 as_of 防护，trailing_high 止损

### 已修复（QA / P0）
- QA-1 ✅ 复合评分统一　QA-3 ✅ Sharpe 单位　QA-4 ✅ carry 缓存　QA-6 ✅ 零相关警告
- QA-7 ✅ ATR 期限匹配　QA-8 ✅ ATR 批量下载　QA-9 ✅ 盘中污染　QA-10 ✅ confidence→weight
- P0-A ✅ trailing_high 止损　P0-B ✅ 状态机命名　P0-C ✅ SimulatedTrade 写入
- ES(CVaR 5%)、FOMC/NFP 事件警告、预估买卖价差展示

### 自动执行架构（截至 2026-04-25）
- `engine/config.py`：部署级配置读取器，桥接 secrets.toml ↔ engine
- `.streamlit/secrets.toml [trading]`：`auto_execute_stops` / `stop_max_weight_auto` 等开关
- `engine/daily_batch.py`：Layer 2 自动止损（`_auto_execute_stop()`），含 shares/fill_price/notional
- `engine/orchestrator.py`：`approve_gate()` 返回执行结果，持久化到 `CycleState.result_summary`
- `engine/memory.py`：`resolve_pending_approval()` Layer 3 战术执行器；`expire_stale_approvals()`；`get_pending_approvals_by_priority()`
- `engine/portfolio_tracker.py`：`execute_rebalance()` 计算 shares_held/cost_basis；`get_current_positions()` 返回完整持仓字段
- `engine/memory.py` `SimulatedTrade`：新增 shares / fill_price / notional 字段 + DB 迁移

### P3 已完成（截至 2026-04-25）
- **P3-1** ✅ Carry 净收益率修正：`dividend_yield − ^IRX`，Koijen et al. (2018)
- **P3-2** ✅ Reversal 因子：60月SMA z-score，截面 winsorise，Poterba & Summers (1988)
- **P3-3** ✅ 净敞口约束：`max_net=0.40 / min_net=-0.10`，Step 5c 等比例压缩
- **P3-6** ✅ IC-Weighted 因子权重：SystemConfig 存储，月度 Spearman IC 自动更新

---

## 二、已移除提案（不再引入）

| 提案 | 移除原因 |
|------|---------|
| FOMC 文本注入 | macro_context 已覆盖，边际价值低 |
| FinArena 风险偏好注入 | 仅改 prompt 语气，不改分析质量 |
| ATLAS 分层 Prompt | 无 OPRO 反馈闭环，只是代码重构 |
| roadmap P4 HMM | 已实现 |
| pivot 双轨架构 / Markov Switching | 已实现 |
| portfolio_monitor.py | 已并入 live_dashboard |
| RL 强化学习组合 | reward signal 不稳定，样本量不支撑 |
| 深度学习价格预测 | 与定位冲突，回测过拟合风险 |
| 实盘交易接口 | 超出项目边界 |
| 向量数据库（Pinecone/Weaviate） | SQLite 完全够用 |
| **量化先验锚（4.21.pdf 方向四）** | 向 LLM 注入 `tsmom_signal`/`composite_score` 等方向性结论，引入锚定偏差，破坏双轨独立性，与 P0-4 注入规则根本冲突 |
| **GICS 11 标准化（4.21.pdf Phase 1）** | 从 18 ETF 缩减到 11，损失截面分散度，与 FactorMAD 四分位验证的扩展方向相反 |
| **Regime Reconciliation Agent** | 依赖 Track B Layer 1（尚未实现），Defer 至 Layer 1 稳定后 |

---

## 三、待办任务

---

### P0 — 立即执行（阻塞研究有效性）

#### P0-1｜DSR 试验次数校准 ✅
~~`backtest.py` hardcode `n_trials=2`~~。`EFFECTIVE_N_TRIALS = 6`（保守估计，用户确认未系统扫描参数）。已全局替换，UI 注释已更新。

#### P0-2｜Survivorship Bias 审计 ✅
新建 `engine/universe_audit.py`，记录 18 个 ETF 成立日期（最晚：XLC 2018-06-18），在 `run_backtest()` 入口调用 `audit_universe()`，超限警告写入 `BacktestResult.warnings` 并在 UI expander 展示。

#### P0-3｜归因实验设计（双轨并行）✅
`SimulatedPosition` 新增 `track` 字段（`"main"` / `"quant"`）；UniqueConstraint 更新为 `(snapshot_date, sector, track)`；`_auto_link_position()` 同时写 main（LLM 调整后）和 quant（纯 TSMOM 信号）两条记录；`_migrate_db()` 加迁移代码含索引重建。`compute_llm_alpha()` 待 P1-E 实现（依赖 verify_pending_decisions）。

#### P0-4｜双轨接通：QuantAssessment → LLM Prompt ✅
`trading_schema.py` 新增 `QuantAssessment.to_prompt_context_raw()`，只输出原始数值（`tsmom_raw_return`、`ann_vol`、`atr_14/63`、`price_vs_sma_200`、`p_risk_on`、`csmom_rank`），永久屏蔽 `tsmom_signal`/`gate_status`/`composite_score`；`AgentState` 加 `quant_context_raw` 字段；`tabs.py` 调用点计算后注入；`red_team_node` 读取并嵌入 `audit_prompt`。

---

### P1 — 本周完成（工程完整性）

#### P1-A｜regime_caps vs WEIGHT_LIMITS 统一 ✅
`regime_caps` 变量实际不存在，代码已统一使用 `WEIGHT_LIMITS`。修复了 `daily_batch.py` 中 fallback 硬编码 0.15/0.12 与实际值不一致的 bug（改为直接 key 索引）。

#### P1-B｜Daily Batch 状态反馈 ✅
`trading_desk.py` 顶部新增 4-metric 状态条（信号/制度/止损触发/入场触发）+ 状态文字说明，错误时展示 `st.error`。

#### P1-C｜分析→Watchlist 流向 ✅
`save_decision()` 已返回 `log.id`；`tabs.py` sector save 路径捕获返回值，入库成功后展示引导横幅指向 Watchlist 和 Decision Monitor。

#### P1-D｜Transition 制度仓位巡检 ✅
`transition` 已在 patrol 循环内覆盖；新增 `track="main"` 过滤，防止 quant 基线轨触发止损。

#### P1-E｜归因去偏（LLM 贡献度量）✅
`llm_weight_alpha = actual_return_20d × (main_weight - quant_weight)`，写入 `DecisionLog.llm_weight_alpha`；`verify_pending_decisions()` 查询 `SimulatedPosition` 双轨权重计算；`_migrate_db()` 迁移；`agent_decisions.py` 展示 alpha 指标。

#### P1-F｜UX-2 Decision Monitor 跳转链接 ✅
每张 sector 类型决策卡片顶栏新增"重新分析"按钮；点击后 `st.session_state["audit_target_sync"] = sector_name` + `st.switch_page("pages/trading_desk.py")`，利用现有 `audit_target_sync` 机制自动预选板块。同时在验证详情内展示 `llm_weight_alpha` 贡献指标。

#### P1-1｜Regime-Conditional 指标完整性 ✅
`BacktestMetrics` 新增 `drawdown_risk_on/off`、`hit_rate_risk_on/off`、`avg_holding_months`，计算逻辑在 `_compute_metrics` 内实现，表格输出列同步更新。

#### P1-2｜持仓状态注入（FinPos）✅
`build_position_context()` helper 新增于 `agent.py`（查询 main 轨最新仓位）；`AgentState` 加 `position_context` 字段；`tabs.py` 调用点注入；`red_team_node` 嵌入 `audit_prompt`。

#### P1-3｜risk_conditions Schema ✅
`trading_schema.py` 新增 `RiskCondition` dataclass（vol_spike/drawdown/regime_cap）；`TradeRecommendation` 加 `risk_conditions` 字段及序列化；`WatchlistEntry` 加 `risk_conditions_json` 列；`_migrate_db()` 加迁移；`_patrol_positions` 实现 vol_spike + drawdown 两种条件评估（regime_cap 已由 5.3 覆盖）；`_add_risk_approval` 加 `suggested_weight` 参数。

#### P1-4｜连续权重输出 ✅
`_add_risk_approval` 加 `suggested_weight` 参数（P1-3 已实现）；regime compression 触发时 `suggested_weight=cap` 而非硬零，让 Supervisor 看到建议目标权重。

#### P1-5｜BHY 多重检验校正 ✅
`engine/backtest.py` 新增 `bhy_correction()` + `sharpe_pvalue()`；`pages/backtest.py` 结果页加 BHY expander，对 TSMOM 和 TSMOM+Regime 两个策略 Sharpe 做 FDR 校正并展示是否显著。

#### P1-6｜QA-2 复合评分权重重设计 ✅
~~`signal.py.compute_composite_scores()` 权重改为：**TSMOM 50% + Sharpe 30% + Regime 20%**，删除 CSMOM 独立权重（改为 TSMOM 截面强度修正）、删去 Carry。~~
已实现：代码已按 P1-6 规格落地，docstring 标注 revised；CSMOM 改为方向-排名冲突截断修正（bottom-tercile cap 70 / top-tercile floor 30）；gate 阈值 < 35 分已更新。

#### P1-7｜QA-5 独立 Vol 窗口 ✅
`get_signal_dataframe()` 新增 `ann_vol_21d` 列（独立 21 日窗口）；`portfolio.py` 优先用 `ann_vol_21d` 做 inverse-vol 定权，fallback 到 12M vol。

---

### P2 — 下周起（研究深度 + 系统扩展）

#### P2-1｜时间衰减记忆检索（λ=ln(2)/90）✅
~~`get_historical_context()` Python 侧加权 `exp(-λ × days_old) × quality`，取 top 5 注入 prompt。~~
已实现：获取候选池 n×4 条，应用 `exp(-ln(2)/90 × days_old) × accuracy_score` 排序后取 top n，header 显示候选池大小。

#### P2-2｜TSMOM 连续化（P2-B）✅
~~`raw_return/ann_vol` 替代 binary sign，min-max 归一化到 0-50 分。依赖 P1-6 权重重设计先完成。~~
已实现：`compute_composite_scores()` 中 `tsmom_norm` 改用 `raw_return/ann_vol` 跨截面 min-max 归一化 [0,100]；零信号资产锚定 50；CSMOM 截断修正保留。

#### P2-3｜Ledoit-Wolf 收缩协方差 ✅
~~`sklearn.covariance.LedoitWolf` 替换对角协方差，代码改动 < 20 行。尤其在 Phase 2 跨资产扩展后必要。~~
已实现：`construct_portfolio()` 新增 `returns_matrix` 参数，≥3 资产且≥60 天时使用 LW 估计组合波动率；`backtest.py` 预取全程日收益率并逐期传入。

#### P2-4｜GARCH(1,1) 条件波动率 ✅
~~`arch` 库前向条件 vol 替换历史波动率，用于仓位缩放。依赖 P1-7 独立 vol 窗口先完成。~~
已实现：P1-7 下载窗口从 30 扩展到 280 BDays；同一批数据尾部 252 天拟合 GARCH(1,1)，前向一步条件 vol 写入 `ann_vol_garch`；`portfolio.py` sizing vol 优先链：GARCH > 21d > 12M；arch 未安装或拟合失败时透明降级。

#### P2-5｜宏观预期差数据源（Economic Surprise）✅
~~新建 `engine/macro_fetcher.py`，从 FRED/econdb/本地 CSV 获取 consensus vs actual；prompt 中 `CPI=3.2%（预期3.0%，超预期+0.2%）`。~~
已实现：`engine/macro_fetcher.py` 新建；`get_economic_surprises()` 通过 FRED 公开 API 拉取 CPI/核心CPI/PCE/失业率/非农/10Y收益率/利差/密歇根信心 9 个系列最近 3 期；格式化为环比变化字符串（诚实注明非市场共识预期差）；3 小时内存缓存；`_run_macro_analysis()` 优先注入为 augmented_ctx 第一段；`FRED_API_KEY` 可选（无 key 时 FRED 公开限速仍可访问）。

#### P2-6｜Horizon 分层绩效报告 ✅
~~Admin Performance Tab 增加 GROUP BY horizon 分层表格（季度/半年/中期 × 胜率/准确率/持仓天数）。实现成本约半天。~~
已实现：`_tab_perf` 新增分层表格，按 horizon 分组展示 n/胜率/均分/平均持仓天数/Brier Score。

#### P2-7｜Brier Score + 自动二项检验 ✅
已有实现：逻辑已内嵌于 `get_clean_zone_stats()._zone_stats()`；Admin Performance Tab 已展示 Brier Score。无需额外代码。

#### P2-8｜Prompt 版本管理 ✅
~~`save_decision()` 新增 `model_version` 和 `prompt_version`（prompt 内容 SHA-256 前 8 位）；写入 DecisionLog。~~
已实现：`save_decision()` 扩展两个参数，`DecisionLog` 新增两列，迁移脚本已加入 `dl_extra_columns`。

#### P2-9｜学习阶段状态机（S5）✅
~~`memory.py` 新增 `get_learning_stage() -> LearningStageInfo`，实时推导（不持久化）；Admin System Tab 顶部显示阶段进度条。~~
已实现：`LearningStageInfo` dataclass + `get_learning_stage()` 四段状态机；Admin System Tab 顶部色带进度条显示当前阶段和解锁门控。

#### P2-10｜实验日志哈希链（S6）✅
~~SHA-256 哈希链，`DecisionLog` 新增 `chain_hash` 字段；Admin System Tab "验证链完整性"按钮。~~
已实现：`save_decision()` 中 flush→计算哈希→commit；`verify_chain_integrity()` 函数；Admin System Tab 验证按钮展示 ok/broken/total。

#### P2-14｜双变量 MSM（信用利差）✅
~~`_fit_and_filter()` 接收 `credit_series` 参数，statsmodels 双变量 MS 模型；FRED BAMLC0A4CBBB 或 LQD-IEF 代理。~~
已实现：`_credit_spread_proxy()` + `_get_monthly_credit_spread()`；双变量失败时回退单变量；版本兼容 try/except。

#### P2-11｜动态 Universe 管理框架 ✅
~~施工前必读 [docs/blueprint_p2.md § BP-A](blueprint_p2.md)~~
已实现：新建 `engine/universe_manager.py`（UniverseETF ORM + init_universe_db + seed_batch_a/b + get_active_universe + get_universe_by_class + universe_health_check）；`get_active_sector_etf()` 改为优先读 UniverseETF 表，静态 SECTOR_ETF 作回退；`get_sector_momentum()` 和 `backtest.py` 改为动态宇宙；`memory.py` 的 `_SECTOR_ETF_MAP` 改为懒加载；Admin System Tab 新增 Universe 管理面板（状态表格 + 月度健康检查按钮）；app.py 启动时自动 `init_universe_db() + seed_batch_a()`。

```python
# engine/universe_manager.py（新建）
UNIVERSE_RULES = {
    "min_adv_usd": 10_000_000,      # 日均成交额 > $10M
    "min_history_years": 3,          # 成立 ≥ 3 年
    "hard_exclude_types": [          # 硬排除（不可覆盖）
        "leveraged",                 # 杠杆 ETF（TQQQ/SOXL 等）
        "inverse",                   # 反向 ETF
        "money_market",              # 货币市场
    ],
}

class UniverseEntry:
    ticker: str
    asset_class: str      # equity / fixed_income / commodity / real_estate / currency
    sub_class: str        # us_large_cap / us_sector / em_equity / hy_bond / tips / gold ...
    inception_date: date
    added_to_universe: date   # 显式记录纳入日期，防止前视偏差
    status: str           # active / inactive / delisted
    inactivation_reason: str  # icir_decay / low_liquidity / strategy_drift / delisted
```

**Universe 维护规则**：
- 新 ETF 纳入：成立 ≥ 3 年 + ADV > $10M + 人工分类确认
- ETF 移出：ADV 连续 20 日低于阈值 → 自动标记 inactive
- 历史数据永久保留（防止幸存者偏差）
- 每月第一个交易日运行 `universe_health_check()`

#### P2-12｜Phase 2 跨资产 ETF 扩展 ✅
~~施工前必读 [docs/blueprint_p2.md § BP-A A-4/A-5](blueprint_p2.md)~~

**批次 A ✅**：小盘价值(IWN)/小盘成长(IWO)/动量因子(MTUM)/低波动因子(USMV)/质量因子(QUAL)/日本(EWJ)/印度(INDA) 已通过 `seed_batch_a()` 写入 universe_etfs 表，app.py 启动时自动纳入。Universe 从 18 → 25 ETF。

**批次 B ✅**：美国综合债(AGG)/美国中期国债(IEF)/通胀保值债(TIP)/黄金矿业(GDX)/农产品(DBA)/波动率(VXX)/抵押贷款信托(REM) 已通过 `seed_batch_b()` 写入 universe_etfs 表。Universe 从 25 → 32 ETF。

PRE-8 验证结果（yfinance Adj Close 含票息再投资）：AGG PASS −0.17% / IEF PASS −0.36% / TIP PASS −0.38%（误差在票息分派节点前向填充范围内，验收通过）。

Within-class CSMOM 已在 `engine/signal.py` 激活（`get_universe_by_class()` 分类内排序，全局排序作 fallback）；VXX 极性翻转已实现（`_REVERSE_MOMENTUM_TICKERS = {"VXX"}`，上涨动量 → 做空信号）；`engine/universe_audit.py` 已补录 Batch A/B 全部成立日期（IWN 2000-07-24 最早，MTUM 2013-04-16 最晚）。

~~分两批纳入，信号层无需改造的先做：~~

~~**批次 A（直接兼容 TSMOM，立即可纳入）：**~~

~~**批次 B（需验证 yfinance Adj Close 含票息后纳入）：**~~

**关键约束**：批次 B 纳入后必须切换到 **within-class CSMOM**（类别内部排序），不能把债券 ETF 和权益 ETF 放入同一个截面排序。✅ 已实现。

扩展后实际宇宙规模：18（行业）+ 批次 A（7）+ 批次 B（7）= **32 个 ETF**，四分位分析每档约 6-8 个，初步可用（FactorMAD 四分位验证在 ≥30 后激活）。

#### P2-13｜FactorMAD — Alpha 因子自动挖掘引擎 ✅
~~施工前必读 [docs/blueprint_p2.md § BP-B](blueprint_p2.md)~~
已实现：新建 `engine/factor_mad.py`（四层防御：Layer 1 MI污染扫描 + Layer 3 符号回归审计 + 生产因子 IC/ICIR 月度监控 + 候选因子审批流 approve/reject/defer）；`FactorDefinition`/`FactorICIR`/`DiscoveredFactor` 三张表写入 `engine/memory.py`（Base.metadata.create_all 自动创建）；`engine/signal.py` 的 `compute_composite_scores()` 在 FactorMAD 活跃（≥3因子）时用 FactorMAD 20% 替换 regime_score；`pages/factor_dashboard.py` 新增"🧬 FactorMAD"Tab（生产因子状态表 + ICIR 更新按钮 + 候选因子三态裁决面板）。

**定位**：Track A 扩展，纯量化模块，与 Track B LLM 零交互。

**核心工作流（四层防御）**：

```
触发（月度/季度）
  → 准备：数据字典 + 基准因子池 + 训练集/验证集/测试集严格划分
  → Proposer Agent：提出因子（自然语言逻辑 + 可执行代码 + 金融解释）
  → 【Layer 1】MI 污染扫描（纯统计，无 LLM）
      候选因子 MI > 基准因子均值 × 2 → 直接终止，记录 rejection_reason，不浪费 LLM 调用
  → 【Layer 2】Critic Agent + 回测执行器（验证集）：IC/ICIR + 静态审查
      迭代循环（最多 5 轮，或连续 2 轮无改善停止）
  → 【Layer 2】测试集最终验证（辩论过程完全不可见）
      通过阈值（ICIR ≥ 0.3 + 与现有因子相关性 < 0.7）→ DiscoveredFactor 表（状态=pending）
  → 【Layer 3】符号回归结构审计（gplearn，生成侦探报告）
      输出三态信号：正面旁证 / 中性 / 危险信号（非二值门控，仅供 Supervisor 参考）
  → 【Layer 4】人工审核：Supervisor 综合所有证据
      批准 → active，加入 compute_composite_scores()，权重上限 10%
      驳回 → rejected
      要求补充 → pending_further_review
```

> 实现细节见 `docs/blueprint_p2.md § BP-B B-1.5（Layer 1）和 B-4.5（Layer 3）`

**四分位验证**：Universe ≥ 30 ETF 后激活，作为 IC/ICIR 的补充验证层；Universe < 30 时仅用 IC/ICIR + DSR 校正。

**硬约束（工程保障）**：
- Critic Agent 不能检测所有代码层面的前视偏差，**每个候选因子代码必须人工代码审查**（重点检查是否使用了 t+1 数据）
- 测试集在整个辩论过程中完全隔离
- ICIR 阈值配合 DSR 校正，不用裸阈值

**数据库新增**：

```python
class DiscoveredFactor(Base):
    __tablename__ = "discovered_factors"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    description = Column(Text)         # 金融逻辑解释
    code_snippet = Column(Text)        # 因子计算代码
    debate_log = Column(Text)          # 完整辩论记录
    ic_train = Column(Float)
    icir_train = Column(Float)
    ic_test = Column(Float)            # 测试集，辩论后才填
    icir_test = Column(Float)
    correlation_with_existing = Column(Float)  # 与现有因子最高相关性
    status = Column(String(30))        # pending / active / rejected / pending_further_review
    rejection_reason = Column(Text, nullable=True)    # Layer 1 拦截原因 / Supervisor 驳回原因
    inactivation_reason = Column(Text, nullable=True) # 上线后退化原因（ICIR 衰减等）
    mi_ratio = Column(Float, nullable=True)           # Layer 1：候选MI / 基准MI均值
    audit_signal_type = Column(String(10), nullable=True)  # Layer 3："positive"/"neutral"/"danger"
    audit_report = Column(Text, nullable=True)        # Layer 3：符号回归侦探报告全文
    discovered_at = Column(DateTime)
    activated_at = Column(DateTime, nullable=True)
    weight_cap = Column(Float, default=0.10)          # 单一挖掘因子权重上限
```

**因子生命周期**：每月初重新计算 ICIR，连续 2 个月 ICIR < 0.15 自动标记 inactive，保留历史记录。

#### P2-14｜双变量 MSM（加入信用利差）
> ⚠️ **施工前必读 [docs/blueprint_p2.md § BP-C](blueprint_p2.md)**（FRED BAMLC0A4CBBB 获取、_fit_and_filter 双变量改法、statsmodels 版本兼容、5条回归测试清单）

当前单变量 MSM 仅用 10Y-2Y 利差，制度错分率约 15-25%（尤其 2022-2024 加息周期）。

加入 IG/HY 信用利差作为第二变量：
- FRED `BAMLC0A0CM`（IG 利差）或 `BAMLH0A0HYM2`（HY 利差）
- 利率曲线捕获货币政策维度，信用利差捕获风险偏好维度，两者互补

实现：`engine/regime.py` 的 `MarkovRegression` 输入改为双变量序列，其余逻辑不变（filtered probability only 原则保持）。

跨资产扩展后尤其必要：单变量 10Y-2Y 无法刻画商品/债券的宏观状态（例如 2022 年利率倒挂但商品暴涨）。

#### P2-15｜动态交易成本模型 ✅
~~ATR-based spread estimate replacing fixed 10bps~~
已实现：`_atr_transaction_cost(w_prev, w_new, daily_ret_window)` 函数；cost = Σ|Δw_i| × max(floor=3bps, vol_14d_i × 0.15)；walk-forward 循环追踪前期权重 `_w_*_prev`；每期切取 19 天日收益率窗口计算 ATR；无数据时回退到 turnover × 5bps；benchmark 同样使用动态 TC。


`execute_rebalance` 中替换固定 10bps：

```python
def estimate_transaction_cost(ticker, trade_size_usd, asset_class) -> float:
    adv = get_adv(ticker)  # 日均成交额
    participation_rate = trade_size_usd / adv
    impact_bps = 10 * math.sqrt(participation_rate)  # 市场冲击：Kyle's lambda 简化版
    spread_bps = ASSET_CLASS_SPREAD[asset_class]     # 买卖价差：equity=2, bond=5, em=8
    multiplier = {"equity": 1.0, "fixed_income": 1.5, "em_equity": 2.0}.get(asset_class, 1.0)
    return (impact_bps + spread_bps) * multiplier
```

记录估算成本 vs 固定成本对比，写入 `SimulatedTrade`，作为策略容量代理指标。

#### P2-16｜熔断机制（CircuitBreaker）✅
~~`TradingCycleOrchestrator` 新增 `CircuitBreaker` 模块~~
已实现：新建 `engine/circuit_breaker.py`；三级状态机：LIGHT（数据源失效，inline check_data_source()）/ MEDIUM（RPD >80%，自动重置）/ SEVERE（VIX 单日涨幅 >30%，持久化到 `.streamlit/circuit_breaker.json`，需人工 manual_reset()）；`run_daily_chain()` 入口调用 `evaluate()`，SEVERE 直接中止链，MEDIUM 强制 `run_sectors=False`；Admin System Tab 显示状态色带 + 手动恢复表单（需填写理由才能解除）。

#### P2-17｜新闻来源升级 ✅
~~施工前必读 [docs/blueprint_p2.md § BP-D](blueprint_p2.md)~~
已实现：新建 `engine/news_fetcher.py`（`NewsItem` dataclass + `fetch_finnhub_news` Layer1 + `fetch_gnews` Layer2 + `fetch_yfinance_news` Layer3 备用 + `fetch_sector_news` 三层编排 + `build_weighted_news_summary` 时效衰减摘要）；API Key 读自 `st.secrets["FINNHUB_KEY"]` / `st.secrets["GNEWS_KEY"]`，未配置时自动降级；`ui/tabs.py` 板块分析路径优先用 `fetch_sector_news + build_weighted_news_summary`，失败时回退至原 `NewsPerceiver.build_context`；P0-4 合规：情绪分数作为原始数值 `[情绪: ±x.xx]` 注入，禁止任何方向性文字。待完成：PRE-9（Finnhub Key）/ PRE-10（GNews Key）申请后配置到 secrets.toml。

**当前问题**：新闻注入无可信度过滤，来源权重相同，无时效性衰减，无结构化情绪数据。

**升级方案（三层数据源）**：

**第一层：结构化情绪数据（主力）**

- **Finnhub**：提供 `company_news` + 内置情绪分数（`sentiment` 字段），60次/分钟免费额度；用于 Track B 第一层宏观叙事，输出已量化的正负情绪而非原始文本

```python
# engine/news_fetcher.py 新增
def get_finnhub_news_sentiment(ticker: str, from_date: str, to_date: str) -> list[dict]:
    """
    返回每条新闻的 headline + sentiment_score + source + datetime
    sentiment_score: -1 (极负面) 到 +1 (极正面)
    """
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
    ...
```

**第二层：定向关键词检索（按需）**

- **GNews API**：按行业关键词精准检索（`"semiconductor industry"`, `"Federal Reserve"`），100次/天免费，每次最多 10 篇；仅当 Debate Engine 触发时按需调用，节省配额

```python
def get_targeted_news(query: str, language: str = "en", max_articles: int = 5) -> list[dict]:
    """Debate Engine 触发时调用，返回最相关的 N 篇文章摘要"""
    ...
```

**第三层：宏观预期数据（月频）**

- **World Bank Indicators API**（免费，无需 key）：全球宏观指标，覆盖 200+ 国家，月频更新
- **IMF DataMapper API**：WEO 预测数据，用于"预期差"基准（实际 vs IMF 预测）

**过滤规则**：

```python
NEWS_QUALITY_RULES = {
    "time_decay_hours": 48,      # 超过 48 小时的新闻降权 50%
    "source_credibility": {
        "tier_1": ["Reuters", "Bloomberg", "FT", "WSJ", "CNBC"],   # 权重 1.0
        "tier_2": ["Seeking Alpha", "Benzinga", "MarketWatch"],     # 权重 0.6
        "tier_3": ["其他"],                                          # 权重 0.3
    },
    "dedup_similarity_threshold": 0.85,  # 余弦相似度 > 0.85 的新闻合并为一条
}
```

**注意**：Finnhub 免费层的情绪分数基于 VADER/TextBlob 类规则方法，非 LLM 级别，用于辅助输入而非独立决策依据。

#### ✅ P2-20｜FRED 数据层扩展 — 已完成 2026-04-26

**背景**：P2-5 已接入 FRED（9 个宏观预期差系列），但当前三层降级架构的宏观输入仍偏窄：缺完整收益率曲线（1Y/2Y/5Y/10Y/30Y）、Fed Funds 有效利率（SOFR）、全球 PMI 历史序列。ETF 官网爬虫不必要（yfinance 已覆盖主要数据）；SEC EDGAR 13F 因 45 天延迟信号价值有限；新闻 RSS 等 LLM 管道完成后再接。

**要做的事**：在 `engine/macro_fetcher.py` 基础上扩展 FRED 系列覆盖：

```python
FRED_SERIES_EXTENDED = {
    # 收益率曲线全段
    "DGS1":  "1Y Treasury",
    "DGS2":  "2Y Treasury",    # 已有 10Y-2Y 利差，补全曲线
    "DGS5":  "5Y Treasury",
    "DGS30": "30Y Treasury",
    # 货币政策
    "SOFR":       "SOFR",
    "FEDFUNDS":   "Fed Funds Effective Rate",
    # 通胀预期
    "T5YIE":  "5Y Breakeven Inflation",
    "T10YIE": "10Y Breakeven Inflation",
    # 全球 PMI 代理（美国 ISM 制造业）
    "NAPM": "ISM Manufacturing PMI",
}
```

新增 `get_yield_curve_snapshot(as_of) → dict`：返回当日各期限收益率 + 曲线形态判断（正常/平坦/倒挂），注入 macro_context 的 Supervisor 叙述层。

**实现成本**：`macro_fetcher.py` 新增 ~40 行；无新依赖（fredapi 已在 P2-5 引入）。  
**ETF 官网**：用一次性静态 CSV 替代爬虫（ticker → inception_date 映射，约 35 行，见 P3-10 需求）。

#### P2-18｜TradingCycleOrchestrator 正式化 ✅
~~将现有 `engine/orchestrator.py` 升级为完整的 `TradingCycleOrchestrator`~~
已实现：`CycleState` ORM 表（memory.py）+ 迁移；`TradingCycleOrchestrator` 类含四个周期方法（`run_daily/weekly/monthly/verification`）；四个人工闸门常量（`GATE_ANALYSIS_DRAFT/RISK_APPROVAL/MONTHLY_REBALANCE/COVARIANCE_OVERRIDE`）；`approve_gate()` 执行层触发；Admin System Tab 展示运行历史表格 + 待审批闸门操作面板；`run_daily_chain()` 保持向后兼容。

#### P2-19｜定量 Sector Risk Attribution

**背景**：当前 `sector_risks` 字段是 LLM 叙事 memo，已并入 Macro Brief Tab 3，不是真正的风险指标。`sector_deep_dive.py` 页面已废弃（功能重复，可删除）。

**要做的事**：组合级别的板块风险分解，回答"当前组合有多少风险来自板块押注"。

**实现规格**：

```python
# engine/portfolio.py — construct_portfolio() 返回值新增字段
@dataclass
class SectorAttribution:
    sector: str                  # 板块名（如 "Technology"）
    weight: float                # 该板块合计权重（含多/空轧差）
    contribution_to_vol: float   # 该板块对组合年化波动率的贡献（%）
    marginal_var_5pct: float     # 边际 CVaR(5%)：移除该板块后组合 ES 变化量

# PortfolioResult 新增：
sector_attribution: list[SectorAttribution]  # 按 |contribution_to_vol| 降序
sector_herfindahl: float                     # HHI 板块集中度指数（0=完全分散，1=单板块）
```

**数据来源**：`ETF_TAGS["sector_gics"]`（已在 universe_manager.py 实现）做板块分组。

**展示位置**：`pages/live_dashboard.py` — 组合风险监控区，条形图展示各板块 vol contribution。

**前置条件**：无（ETF_TAGS 已就位，LW 协方差已实现）。

**优先级**：P2，但不阻塞其他任何项，可独立实施。

---

### P3 — 学术展示级强化（2026-04-25 设计）

> **设计背景**：系统定位升级为学术展示级（D4）+ 长偏策略（D1: net ∈ [-10%, +40%]）+ 双因子扩展（D2）+ LLM 结构化输出（D3）。P3 批次修复现有方法论漏洞，并将所有参数选择升级为数据驱动或文献有据。

#### ✅ P3-1｜Carry 信号修正（净收益率）— 已完成 2026-04-25

`compute_carry()` 改为 `dividend_yield − risk_free_rate`，风险利率取 `^IRX`（13周T-bill，yfinance）。  
Koijen et al. (2018) 定义落地；加息周期下无红利 ETF 自动得到空信号。  
`carry_norm` 接入 `compute_composite_scores()`，权重由 P3-6 IC-weighted 系统管理。

#### ✅ P3-2｜Reversal 因子（Price-to-5Y-SMA）— 已完成 2026-04-25

新增 `compute_reversal(as_of)`：60月月频收盘价 → z-score → 截面 winsorise 5%/95% → 取反。  
Poterba & Summers (1988) 均值回归；低于历史均值 = 买入信号。  
`reversal_norm` 接入 `compute_composite_scores()`，与 TSMOM 负相关提供因子分散化。

#### ✅ P3-3｜Net Exposure 约束 — 已完成 2026-04-25

`construct_portfolio()` 新增 `max_net=0.40 / min_net=-0.10` 参数，Step 5c 等比例缩减越界方向。  
不改变信号方向，仅缩放幅度；越界时追加 warning 日志。

#### ✅ P3-4｜Beta-adjusted Alpha 与多基准对比 — 已完成 2026-04-26

**问题**：长偏策略的 Sharpe 包含市场 Beta 贡献，不能直接与文献比较。

**改动 A**：`BacktestMetrics` 新增：
- `market_beta`：策略月收益对 SPY 月收益的 OLS 斜率
- `alpha_annualized`：截距项年化（Jensen's Alpha）
- `sharpe_vs_60_40`：超额收益 / 超额波动率（基准 = 60% SPY + 40% AGG）

**改动 B**：Backtest 页结果展示四条净值曲线：
1. 策略（含制度过滤 + 因子）
2. 纯 TSMOM 无风险管理版本（验证制度过滤是否增值）
3. 等权 Buy-and-Hold
4. 60/40（SPY + AGG）

超额收益归因表：制度过滤贡献 / Carry 贡献 / Reversal 贡献 / 残差。

#### ✅ P3-5｜LLM 结构化输出（JSON Schema 强制）— 已完成 2026-04-26

**问题**：所有 LLM 调用返回自由文本，通过 string parsing 提取方向/置信度，脆弱且不可审计。

**改动 A**：`engine/trading_schema.py` 新增 `StructuredTradeOutput` Pydantic 模型：
```python
class StructuredTradeOutput(BaseModel):
    direction: Literal["超配", "标配", "低配", "拦截"]
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: Literal["short", "medium", "long"]
    key_thesis: str = Field(max_length=200)
    primary_risk: str = Field(max_length=100)
    macro_regime_view: Literal["risk-on", "neutral", "risk-off"]
    contradicts_quant: bool   # True = LLM方向与TSMOM信号相反

class StructuredMacroBrief(BaseModel):
    regime_assessment: Literal["risk-on", "neutral", "risk-off"]
    key_driver: str = Field(max_length=150)
    tail_risk: str = Field(max_length=150)
    confidence: float = Field(ge=0.0, le=1.0)
```

**改动 B**：所有 `model.generate_content(prompt)` 调用点改为：
```python
response = model.generate_content(
    prompt,
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=StructuredTradeOutput.schema(),
    ),
)
result = StructuredTradeOutput.parse_raw(response.text)
```

**改动 C**：`save_decision()` 写入 `StructuredTradeOutput` 的所有字段（已有 `confidence_score`、`direction`，补充 `horizon`、`key_thesis`、`primary_risk`、`macro_regime_view`、`contradicts_quant`）。

**影响范围**：`ui/tabs.py` 所有解析点、`engine/daily_batch.py` LLM 简报、`pages/orchestrator.py` 叙述层。

#### ✅ P3-6｜因子权重 IC-Weighted 自动校准 — 已完成 2026-04-25

`compute_composite_scores()` 从 `SystemConfig("factor_ic_weights")` 读取权重，默认值：  
`{tsmom:0.40, sharpe:0.20, carry:0.15, reversal:0.15, factor_mad:0.10}`。  
新增 `update_factor_ic_weights(as_of)`：Spearman IC 滚动 12 个月，结果写回 SystemConfig。  
每月首个交易日 ICIR 更新时自动触发（`daily_batch.py` 月度守卫内）。  
首次运行无历史 IC 时使用文献均值估计作为初始权重。

#### ✅ P3-7｜参数敏感性分析 — 已完成 2026-04-26

**目标**：让审稿人/面试官相信参数选择不是 cherry-picked。

**改动**：Backtest 页新增 "稳健性检验" Tab，包含：

1. **lookback × skip 热力图**（4×2 = 8格）：lookback ∈ {6, 9, 12, 15} × skip ∈ {1, 2}，展示各组合 Sharpe / MaxDD。用颜色深浅标注，当前参数用红框标出。
2. **Gate 阈值敏感性**：{25, 30, 35, 40, 45} 五档，柱状图展示策略 Sharpe 和命中率变化。
3. **结论文字**：自动生成 "当前参数（lookback=12, skip=1）在8种组合中排名 X/8，Sharpe 稳健性 p=…"

#### ✅ P3-8｜Live Dashboard 修复 — 已完成 2026-04-26

**① 真实价差**：用 ATR-TC 模型（P2-15 已实现）替换硬编码 `last * 0.0005`。

**② 统一 P&L 路径**：删除 weight-only 分支（第 171-177 行），强制使用 `shares_held + entry_price` 模式。`execute_rebalance()` 在建仓时必须记录 `shares_held`（当前是可选字段）。

**③ 持仓信息增强**：持仓表新增三列：
- `建仓日期`：从 `SimulatedPosition.created_at`
- `持有天数`：`(today - entry_date).days`
- `距止损`：`(last_price - trailing_stop) / last_price`，红色显示 < 5%

#### ✅ P3-9｜Score-weighted 仓位调节（高影响·低复杂）— 已完成 2026-04-26

**问题**：当前 `construct_portfolio()` 的 vol-parity 权重与 composite_score 完全脱耦——一个 score=40 的信号和 score=85 的信号拿到相同的仓位规模，只要波动率相同。这直接压缩了策略的 alpha。

**改动**：在 `engine/portfolio.py` Step 5（归一化之后）插入 score 调节乘数：

```python
# Step 5d — score-weighted scaling（紧跟 net exposure 约束之后）
if scores is not None:
    score_vec = df["ticker"].map(scores).fillna(50) / 100   # 归一化到 [0,1]
    df["w_scaled"] = df["w_scaled"] * score_vec
    # 重新归一化保持总敞口不变
    total_abs = df["w_scaled"].abs().sum()
    if total_abs > 1e-9:
        df["w_scaled"] /= total_abs * (1 / gross_exposure_target)
```

`construct_portfolio()` 新增 `scores: dict[str, float] | None = None` 参数；调用方（`daily_batch.py`）从 `compute_composite_scores()` 结果传入。

**注意**：gate 阈值（< 35 分 blocked）仍作为二值门控，score-weighting 只在通过 gate 后的连续调节阶段生效，不引入前视。

**预期影响**：高置信度头寸放大 20-40%，低置信度头寸自动压缩，换手率基本不变（权重调节幅度小）。

#### ✅ P3-10｜幸存者偏差动态过滤（高影响·低复杂）— 已完成 2026-04-26

**问题**：P0-2 实现了启动时警告（"XLC 2018年后才存在"），但回测 walk-forward 循环并未动态剔除——2015 年的回测仍使用了 XLC 数据（当时不存在），构成方法论硬伤。

**当前状态**：`engine/universe_audit.py` 记录成立日期并写入 `BacktestResult.warnings`，但 `run_backtest()` 的信号计算未按日期过滤宇宙。

**改动 A**：`engine/universe_manager.py` 的 `UniverseETF` 已有 `inception_date` 字段，新增工具函数：

```python
def get_universe_as_of(as_of_date: date, min_history_years: int = 3) -> list[str]:
    """返回 as_of_date 时已成立满 min_history_years 的活跃 ETF 列表"""
    cutoff = as_of_date - relativedelta(years=min_history_years)
    return [e.ticker for e in get_active_universe() if e.inception_date <= cutoff]
```

**改动 B**：`engine/backtest.py` 的 walk-forward 循环每期调用 `get_universe_as_of(t_month)` 替换静态宇宙列表，信号计算和权重构建只使用当期合规 ETF。

**改动 C**：一次性静态 CSV 补录所有 ETF 的 inception_date（P2-20 中提及），写入 `universe_etfs` 表（`seed_inception_dates()`）。

**副作用**：2013-2016 年样本内回测结果会轻微变化（当期合规 ETF 较少），这是正确的行为，不是 bug。

#### ✅ P3-11｜Supervisor 叙述质量升级（高影响·中复杂）— 已完成 2026-04-26

**问题**：当前 `DailyBriefSnapshot.agent_brief` / `Section B` 展示的是事件代码列表（`ATR_STOP: XLC` / `ENTRY_BLOCKED: reason=low_score`），不具备交易台备忘录的可读性，且 Section A 闸门审批缺乏背景叙述支撑决策。

**改动 A**：新增 `engine/narrative_builder.py`，将 `BatchResult` + `DailyBriefSnapshot` 的事件字段转换为结构化段落：

```python
class NarrativeBuilder:
    def build_daily_brief(self, result: BatchResult, snap: DailyBriefSnapshot) -> str:
        """
        输出格式（三段式）：
        
        【市场状态】制度=risk-on（P=0.82），10Y-2Y利差=-12bps，…
        【今日动作】止损触发 2 笔（XLC -8.3%，HACK -5.1%），自动执行；新入场 0 笔。
        【待审批事项】月度再平衡建议换手率=34%，预估成本=14bps，长仓4/空仓2，建议审批。
        """
```

**改动 B**：`pages/orchestrator.py` Section B 的叙述层改为调用 `NarrativeBuilder.build_daily_brief()`，Section A 的闸门卡片顶部展示一行叙述摘要。

**改动 C**：`pages/live_dashboard.py` 横幅区域改用 NarrativeBuilder 的单行摘要（当前是 f-string 硬拼接）。

**实现方式**：纯 Python 模板渲染，不调用 LLM，避免延迟和成本；LLM-generated brief（`agent_brief` 字段）保留为可折叠的深度分析区。

#### ✅ P3-12｜LLM/Quant 分歧仲裁（中影响·低复杂）— 已完成 2026-04-26

**问题**：P3-5 的 `StructuredTradeOutput.contradicts_quant=True` 已能识别 LLM 与 TSMOM 信号相反的情况，但当前无后续处理——分歧信号与无分歧信号走相同的审批流程，架构不完整。

**改动 A**：`engine/memory.py` 的 `resolve_pending_approval()` 在 `entry` 类型审批时检查 `contradicts_quant`：

```python
if approval.approval_type == "entry":
    trade_rec = TradeRecommendation.from_json(approval.payload_json)
    if trade_rec.contradicts_quant:
        # 仲裁规则：需要更高置信度门槛
        if trade_rec.confidence < CONTRARIAN_MIN_CONFIDENCE:  # 默认 0.75
            # 自动驳回并记录原因
            approval.status = "auto_rejected"
            approval.rejection_reason = f"contradicts_quant=True，置信度 {trade_rec.confidence:.2f} < {CONTRARIAN_MIN_CONFIDENCE}"
            return {"status": "auto_rejected", "reason": approval.rejection_reason}
        else:
            # 高置信度反向信号：升级为 GATE_COVARIANCE_OVERRIDE（需要更高级别审批）
            approval.approval_type = "contrarian_override"
```

**改动 B**：`pages/orchestrator.py` Section A 对 `contrarian_override` 类型的待审批项用橙色警示框标注"⚠️ LLM方向与TSMOM相反，置信度已过仲裁阈值，需明确确认"。

**配置项**：`CONTRARIAN_MIN_CONFIDENCE = 0.75` 写入 `SystemConfig`，可调。

#### ✅ P3-13｜Alpha 衰减测量（中影响·中复杂）— 已完成 2026-04-26

**问题**：当前系统无法回答"信号有效期多长？"——这是持仓期限选择（short/medium/long horizon）的根本依据，也是面试和学术评审的高频问题。

**改动 A**：`engine/backtest.py` 新增 `compute_ic_decay(returns_df, signal_df) → dict`：

```python
def compute_ic_decay(returns_df, signal_df, horizons=(1, 3, 6, 12)):
    """
    对每个 horizon h（月），计算 IC(h) = Spearman(signal_t, return_{t→t+h})
    输出 {1: ic_1m, 3: ic_3m, 6: ic_6m, 12: ic_12m}
    """
```

IC 显著正值期 = 信号有效期；IC 趋近零点 = 最佳持仓上限。

**改动 B**：Backtest 页 "稳健性检验" Tab（P3-7）新增 IC 衰减曲线图，x 轴=持仓月数，y 轴=IC 值，附 95% bootstrap 置信带。

**改动 C**：`DailyBriefSnapshot` 或 Admin Performance Tab 展示当前因子的滚动 IC 衰减斜率（`ic_decay_slope_3m`），作为"信号是否仍有效"的实时监控指标。

**学术价值**：IC 衰减图是投资组合管理课程和学术论文的标准图表，展示后立即提升系统可信度。

---

### P4 — 日频动态调仓（2026-04-25 设计，详细规格见 docs/spec_daily_tactical.md）

#### ✅ P4-1｜TSMOM-Fast 信号（3-1月）

新增 `get_fast_signal_dataframe(lookback=3, skip=1)` 或扩展现有接口。  
Fast 信号**只输出方向 flag（+1 / -1 / 0）**，不生成权重。权重永远由 Slow（12-1月）决定。

#### ✅ P4-2｜扩展 engine/config.py 战术配置项

新增 6 个配置项读取（详见 spec）：`auto_execute_regime_compress`、`auto_execute_high_conf_entry`、`tactical_entry_max_weight`、`tactical_entry_daily_limit`、`regime_jump_threshold_ppt`、`entry_composite_score_min`、`entry_momentum_zscore_min`。

#### ✅ P4-3｜`_patrol_daily_tactical()` 主函数

检测 5 类事件（优先级顺序）：
1. **制度跃变** — P(risk-off) 单日变化 > threshold → Layer 2 全组合压缩
2. **高置信度新入场** — Fast+Slow 一致 + risk-on + composite_score ≥ 60 + 5日动量 z≥1.5σ → Layer 2（上限 5%/笔，每日 ≤2笔）
3. **Fast 信号翻转** — Fast 与持仓方向相反且 Slow 尚未翻转 → Layer 3 减仓 50%
4. **浮亏扩大预警** — unrealized loss > -8%（未触发 ATR 止损）→ Layer 3 减仓建议
5. **波动率尖峰压缩** — 持仓 ATR/price > 3% 且超配 → Layer 3 压缩

制度跃变触发时跳过 2-5。

#### ✅ P4-4｜BatchResult 扩展

新增字段：`tactical_entries: list[str]`、`tactical_reduces: list[str]`、`regime_jump: bool`。

#### ✅ P4-5｜Orchestrator 战术事件展示（2026-04-26 完成）

Section B 展示当日战术动作（新入场 / 减仓 / 制度跃变），制度跃变显示为红色横幅（最高优先级），并正确区分 Layer 2 / Layer 3 事件。

#### ✅ P4-6｜Live Dashboard 战术横幅（2026-04-26 完成）

顶部横幅区分：自动执行（Layer 2）/ 待审批（Layer 3）/ 无事件。新增 `tactical_entries_json`、`tactical_reduces_json`、`regime_jump_today` 持久化到 `DailyBriefSnapshot`。

#### ✅ P4-7｜回测验证（2026-04-26 完成）

`validate_tactical_patrol()` 在历史月度数据上模拟 Fast Flip + Regime Compress 叠加层，输出：
- 年换手率（目标 < 150%，防止成本吃掉 alpha）
- 战术入场 vs 月度再平衡的增量 Sharpe

**方法论红线**：回测 Sharpe 提升 < 0.05 → 不上线；换手率 > 200% → 必须收紧阈值。  
UI 展示于 `pages/backtest.py` "战术巡逻验证（P4-7）" 区块。

---

### P5 — 自动化脊柱闭合（2026-04-26 全部 ✅）

| 任务 | 内容 |
|------|------|
| P5-A-1 ✅ | Circuit Breaker 接入日批 Step 0（SEVERE 中止，MEDIUM 清空 model） |
| P5-A-2 ✅ | FinDebate 自动触发（入场候选 + 制度切换 top-3，≤4次/日） |
| P5-A-3 ✅ | 再平衡 4 门自动执行逻辑 |
| P5-A-4 ✅ | FactorMAD ICIR 月度步骤强化（deactivated 记录） |
| P5-B ✅ | 流水线状态时间轴（Daily Brief 7步视图，`_pl_row()`） |
| P5-C ✅ | LLM 宏观简报自动生成（`get_model()` → `_load_batch()` → `ensure_daily_batch_completed(model=m)`） |

---

### P6 — 多频架构精炼施工蓝图（2026-04-27 设计）

> **设计背景**：P0-P5 完成后，系统具备基本自动化骨架，但频率层次混乱（日/月/季混写于同一函数），信号持久化缺失（每日信号不写库，无法做 flip/decay 检测），组合构建有矛盾（换手惩罚阻塞战略再平衡；Reversal 权重在非 transition 制度自动重分配），以及缺少外部现实验证层（ERA）和宇宙自动审查。本批次系统性修复以上问题，不引入任何前视偏差，不改变已有的 filtered-probability-only / ATR-trailing-stop 等核心原则。

---

#### 多频架构总览

```
频率层         触发条件                      执行方式        核心函数
─────────────────────────────────────────────────────────────────────────
每日同步层      页面加载 / ensure_daily()     同步，幂等      _step1_data_quality()
                                                             _step2_signal_snapshot()
                                                             _step3_patrol_tactical()   ← P4已有
                                                             _step4_debate()            ← P5已有
                                                             _step5_verify_learn()      ← P5已有

月首同步层      is_first_trading_day()        同步追加        _monthly_M1_carry_update()
                                                             _monthly_M2_strategic_rebalance()
                                                             _monthly_M3_icir_calibration()

季首异步层      is_first_trading_day_of_Q()   后台线程        _quarterly_universe_review()  ← 新建
                                                             _quarterly_factor_mining()    ← 新建
                                                             → SystemConfig bridge 跨会话持久化

事件驱动层      CB SEVERE / regime jump       优先中断        circuit_breaker.evaluate()    ← P5已有
```

**时间来源（论文 timi 对应）**：
- 日层：`TSMOM(3-1)` fast signal，ATR(21) vol spike
- 月层：`TSMOM(12-1)` slow rebalance，ICIR 更新
- 季层：Universe AUM/ADV 筛查，BH 校正因子挖掘
- 事件：CB SEVERE（VIX +30%），制度跃变（P(risk-off) Δ > threshold）

---

#### Phase 1｜信号持久化 + 数据质量层（施工基础）

> **为何优先**：SignalSnapshot 是所有其他改进（flip 检测、decay 巡逻、ERA 验证、宇宙审查）的数据基础。无此表，后续改进无数据源。

**新增 ORM 表（engine/memory.py，Base.metadata.create_all 自动创建）：**

```python
class SignalSnapshot(Base):
    __tablename__ = "signal_snapshots"
    id            = Column(Integer, primary_key=True)
    as_of_date    = Column(Date, nullable=False)
    ticker        = Column(String(20), nullable=False)
    tsmom_raw     = Column(Float)       # raw_return / ann_vol（连续值，非符号）
    tsmom_fast    = Column(Integer)     # +1/0/-1  Fast(3-1)
    csmom_rank    = Column(Float)       # 0-1 截面排名
    carry_raw     = Column(Float)       # 净收益率（div_yield - rf）
    reversal_z    = Column(Float)       # 60M z-score（transition 制度时才有意义）
    composite     = Column(Float)       # 最终复合评分 0-100
    regime_label  = Column(String(20))  # risk-on / transition / risk-off
    __table_args__ = (UniqueConstraint("as_of_date", "ticker"),)

class RegimeSnapshot(Base):
    __tablename__ = "regime_snapshots"
    id            = Column(Integer, primary_key=True)
    as_of_date    = Column(Date, unique=True, nullable=False)
    filtered_prob_risk_on  = Column(Float)
    filtered_prob_risk_off = Column(Float)
    regime_label  = Column(String(20))
    vix_level     = Column(Float, nullable=True)

class SignalFlipLog(Base):
    __tablename__ = "signal_flip_logs"
    id            = Column(Integer, primary_key=True)
    flip_date     = Column(Date, nullable=False)
    ticker        = Column(String(20), nullable=False)
    signal_type   = Column(String(20))  # tsmom_fast / composite_gate
    prev_value    = Column(Float)
    new_value     = Column(Float)
    flip_direction = Column(String(10))  # long→short / short→flat / etc.

class DataQualityLog(Base):
    __tablename__ = "data_quality_logs"
    id            = Column(Integer, primary_key=True)
    check_date    = Column(Date, nullable=False)
    check_type    = Column(String(50))   # ohlcv_freshness / fred_freshness
    ticker        = Column(String(20), nullable=True)
    passed        = Column(Boolean)
    detail        = Column(Text, nullable=True)

class CircuitBreakerLog(Base):
    __tablename__ = "circuit_breaker_logs"
    id            = Column(Integer, primary_key=True)
    event_time    = Column(DateTime, nullable=False)
    level         = Column(String(10))   # LIGHT / MEDIUM / SEVERE
    trigger_reason = Column(Text)
    auto_reset    = Column(Boolean, default=False)
    manual_reset_by = Column(String(100), nullable=True)
    notes         = Column(Text, nullable=True)
```

**新增函数（engine/daily_batch.py）：**

```python
def _step1_data_quality(as_of: date) -> bool:
    """OHLCV 和 FRED 数据新鲜度检查，失败写 DataQualityLog，返回是否通过。"""
    pass  # 检查最新数据日期，超过 2 个交易日则降级

def _write_signal_snapshot(signals_df, regime_label, as_of: date):
    """将当日各 ticker 信号写入 SignalSnapshot（幂等，已有则跳过）。"""
    pass

def _detect_signal_flips(as_of: date):
    """比较今日与昨日 SignalSnapshot，写入 SignalFlipLog。"""
    pass
```

---

#### Phase 2｜信号层精炼

**F6 — 资产类别专属 Carry 信号（engine/signal.py）**

```python
CARRY_BY_CLASS = {
    "equity":        lambda row: row["dividend_yield"] - row["DFF"] / 100,
    "fixed_income":  lambda row: row["implied_yield_proxy"],   # IEF: ytm proxy
    "commodity":     lambda _: 0.0,
    "alternatives":  lambda _: 0.0,
    "thematic":      lambda _: 0.0,
}
# 未知资产类别返回 0，不抛异常
```

**F7 — 制度条件 Reversal（engine/signal.py）**

```python
def compute_reversal(as_of, regime_label) -> pd.Series:
    if regime_label != "transition":
        return pd.Series(0.0, index=universe)  # 非 transition 返回零序列
    # 60M 月频 SMA z-score，取反，截面 winsorise 5%/95%
    # 基于 Poterba & Summers (1988) 均值回归
```

**F8 — 固定 COMPOSITE_WEIGHTS（engine/signal.py）**

```python
COMPOSITE_WEIGHTS = {
    "tsmom":     0.40,   # 主动量因子
    "csmom":     0.25,   # 截面排名修正
    "carry":     0.20,   # 净收益率
    "factormad": 0.10,   # 已激活挖掘因子（≥3个时生效，否则 0.0 不重分配）
    "reversal":  0.05,   # 过渡制度均值回归（非 transition 时为 0，但权重不重分配）
}
# 关键：Reversal=0 时其 5% 权重保留为零贡献，不向其他因子重分配
# 目的：防止制度切换时权重突变引发伪信号
```

**矛盾解决 M2（固定权重方案）**：原实现 N 信号等权，Reversal 不生效时自动重归一化，导致 risk-on 到 transition 切换时其他因子权重跳变 +5.9%，产生伪换手。固定权重消除此问题。

**F9 — Signal Decay 巡逻（engine/daily_batch.py）**

```python
def _patrol_signal_decay(as_of: date):
    """
    检查持仓中 TSMOM raw_return 较入场峰值衰减 > 60% 的标的。
    触发：写 PendingApproval(gate_type="REVIEW_SIGNAL_DECAY")，Layer 3 人工审批。
    替代原 90 天年龄规则（age rule 与信号强度无关，是时间而非质量指标）。
    """
```

**F10 — Vol Spike 巡逻（engine/daily_batch.py）**

```python
def _patrol_vol_spike(as_of: date):
    """
    ATR(21)/price > 3% → Layer 3 减仓建议
    ATR(21)/price > 5% → Layer 2 自动压缩（auto_execute_vol_spike=True 时）
    ATR 窗口用 21 日（当前波动率快照），区别于 5a 止损的 ATR(63)（月度持仓期）
    """
```

---

#### Phase 3｜组合构建精炼

**F11 — 制度条件仓位上限（engine/portfolio.py）**

```python
REGIME_POSITION_LIMITS = {
    "risk-on":     {"max_long": 10, "max_short": 6},
    "transition":  {"max_long": 7,  "max_short": 7},
    "risk-off":    {"max_long": 5,  "max_short": 8},
}

def _get_position_limits(regime_label: str) -> dict:
    return REGIME_POSITION_LIMITS.get(regime_label, {"max_long": 8, "max_short": 6})
```

**F12 — 换手惩罚分离（矛盾解决 M1）**

```python
# construct_portfolio() 新增参数
def construct_portfolio(
    signals_df,
    prev_weights: dict | None = None,
    turnover_penalty: float = 0.0,
    apply_turnover_penalty: bool = False,   # 战略再平衡传 False；战术调仓传 True
    ...
):
```

**关键规则**：月度战略再平衡 `apply_turnover_penalty=False`（惩罚=0），确保权重完全收敛到 vol-parity 目标；日频战术调仓 `apply_turnover_penalty=True, turnover_penalty=0.3`，控制日内换手。

原问题：若月度再平衡也用 turnover_penalty=0.3，则权重永远无法完全到达目标，出现系统性持仓漂移。

**F13 — 制度调整协方差（engine/portfolio.py）**

```python
def _regime_adjusted_cov(cov_matrix, regime_label: str):
    if regime_label == "risk-off":
        # 30% 收缩向完全相关矩阵（危机时相关性趋向 1）
        # Longin & Solnik (2001): 危机期间资产类别相关性显著上升
        perfect_corr = np.outer(np.sqrt(np.diag(cov_matrix)), np.sqrt(np.diag(cov_matrix)))
        return 0.70 * cov_matrix + 0.30 * perfect_corr
    return cov_matrix
```

**F14 — Track B LLM 权重叠加层（新建 engine/track_b.py）**

```python
TRACK_B_BUDGET = 0.05          # 总预算：5% 绝对权重
MAX_RELATIVE_DELTA = 0.25      # 单标的相对调整上限：±25%

def run_track_b(quant_weights: dict, llm_views: dict, regime_label: str) -> dict:
    """
    约束：
    1. 只调整量化已持有的标的（不新建零权重到 Track B 超配）
    2. 每个标的调整幅度 ≤ |w_quant| × MAX_RELATIVE_DELTA
    3. 所有调整绝对值之和 ≤ TRACK_B_BUDGET
    4. risk-off 制度下 Track B 预算自动缩减 50%
    
    输出：写 PendingApproval(gate_type="GATE_TRACK_B") + AlphaMemory
    """
```

---

#### Phase 4｜外部现实验证层（ERA）

> **ERA vs ILA 区分**：ILA（Internal Logic Audit，engine/lcs.py）测试信号内部一致性（Mirror/Noise/Cross-Cycle）；ERA（External Reality Audit，engine/era.py）验证 LLM 分析逻辑链是否与实际宏观数据吻合。

**新建 engine/era.py：**

```python
ERA_TEMPERATURE = 0.1   # 低于决策 agent（0.7-0.9），减少随机性和同源偏差

ERA_MACRO_SERIES = [
    "DGS10",        # 10Y 美债收益率
    "DGS2",         # 2Y 美债收益率
    "T10Y2Y",       # 10Y-2Y 利差（直接）
    "BAMLC0A4CBBB", # BBB 信用利差
    "T5YIE",        # 5Y 盈亏平衡通胀
    "VIXCLS",       # VIX（月均值，非日内）
]
# ERA 只看宏观数据，不看价格（防止事后诸葛亮，后视偏差）

def run_era(decision_log_id: int, verification_period_days: int = 20) -> dict:
    """
    流程：
    1. 读取 DecisionLog 的 key_thesis + 宏观假设
    2. 获取验证期实际 FRED 宏观数据
    3. ERA agent（temp=0.1）判断逻辑链：
       - "logic_correct"：thesis 预测方向与实际宏观变化一致
       - "lucky_guess"：结果正确但逻辑链与数据不吻合
       - "logic_wrong"：结果错误或方向相反
    4. 写入 AlphaMemory，可供 Proposer 参考历史逻辑质量
    """
```

**新增 ORM 表：**

```python
class AlphaMemory(Base):
    __tablename__ = "alpha_memory"
    id              = Column(Integer, primary_key=True)
    decision_log_id = Column(Integer, ForeignKey("decision_logs.id"))
    era_verdict     = Column(String(20))   # logic_correct / lucky_guess / logic_wrong
    era_report      = Column(Text)
    macro_data_snapshot = Column(Text)     # JSON: FRED 验证期实际值
    created_at      = Column(DateTime)

class QuantOnlySnapshot(Base):
    __tablename__ = "quant_only_snapshots"
    id          = Column(Integer, primary_key=True)
    snapshot_date = Column(Date, unique=True, nullable=False)
    nav         = Column(Float)            # 纯量化 NAV（不含 Track B 调整）
    weights_json = Column(Text)            # 各标的权重快照
    # alpha = primary_NAV - quant_only_NAV（在 UI 计算，不存库）
    # 展示累计超额收益，不展示 IR（n<30 时统计无意义）
```

---

#### Phase 5｜月度/季度循环分离

**月度三步骤（月首同步，延续现有月度守卫模式）：**

```python
def _monthly_M1_carry_update(as_of: date):
    """更新无风险利率缓存（DFF），刷新 Carry 信号基准。"""

def _monthly_M2_strategic_rebalance(as_of: date, nav: float):
    """
    月度战略再平衡：
    - vol-parity 权重 + composite score 调节乘数
    - apply_turnover_penalty=False（不惩罚战略换手）
    - 结果写 SimulatedPosition + QuantOnlySnapshot（双轨）
    """

def _monthly_M3_icir_calibration(as_of: date):
    """
    滚动 12 个月 Spearman IC 更新因子权重（COMPOSITE_WEIGHTS 中 factormad 占比）。
    Harvey-Liu t-value = ICIR × √n_months（写库展示，不作门控，矛盾解决 M3）。
    """
```

**季度异步层（新增）：**

```python
def _is_first_trading_day_of_quarter(as_of: date) -> bool:
    """判断 as_of 是否为当季第一个交易日（NYSE 日历）。"""

def _launch_quarterly_jobs(as_of: date):
    """
    在后台线程中启动：
    1. engine/universe_review.run_universe_review()
    2. engine/factor_mad.run_quarterly_factor_mining()
    
    完成后通过 SystemConfig("quarterly_job_status") 桥接跨会话状态：
    {"status": "completed", "last_run": "2026-04-01", "universe_changes": [...]}
    """
    import threading
    t = threading.Thread(target=_quarterly_worker, args=(as_of,), daemon=True)
    t.start()
```

**依赖说明**：季度层写 `PendingApproval`（GATE_UNIVERSE_CHANGE / GATE_FACTORMAD_CANDIDATE），不自动执行，闸门审批流程与现有 P5 架构完全兼容。

---

#### Phase 6｜宇宙审查 + FactorMAD 增强

**新建 engine/universe_review.py：**

```python
EXIT_AUM     = 500_000_000   # AUM < $5亿 → 退出候选
ENTRY_AUM    = 1_000_000_000 # AUM > $10亿 → 入选候选
MIN_ADV      = 5_000_000     # 日均成交额 < $5M → 退出候选
MAX_CORR_EXIT  = 0.92        # 与宇宙内任意现有 ETF 相关性 > 0.92 → 冗余退出
MAX_CORR_ENTRY = 0.75        # 新 ETF 入选：与现有最高相关性 < 0.75（多样性门槛）
UNIVERSE_MAX_SIZE = 40       # 宇宙硬上限，防止过度分散

def run_universe_review(as_of: date):
    """
    季度执行（异步）：
    1. 计算全宇宙 AUM / ADV / 相关性
    2. 标记退出候选（EXIT）和入选候选（ENTRY）
    3. 生成 PendingApproval(gate_type="GATE_UNIVERSE_CHANGE")
    4. 不自动更改 UniverseETF 状态，必须人工审批后 approve_gate() 执行
    """
```

**FactorMAD 增强（engine/factor_mad.py）：**

```python
# FactorICIR 表新增字段
regime    = Column(String(20))   # risk-on / transition / risk-off
t_stat    = Column(Float)        # Harvey-Liu t = ICIR × √n_months（展示用）
n_months  = Column(Integer)      # 用于计算 t_stat

def compute_harvey_liu_t(icir: float, n_months: int) -> float:
    """Harvey & Liu (2015): t = ICIR × √n_months。仅用于 UI 展示，不作门控。"""
    return icir * math.sqrt(n_months)

def _bh_correction(p_values: list[float], alpha: float = 0.10) -> list[bool]:
    """Benjamini-Hochberg FDR 校正，alpha=10%。季度因子挖掘批次中应用。"""
    # 多重检验校正，控制发现率
    ...

def _check_factormad_budget() -> tuple[bool, int]:
    """返回 (has_space, current_count)；factormad-source 因子上限=8。"""
    count = session.query(DiscoveredFactor).filter_by(
        status="active", source="factormad"
    ).count()
    return count < 8, count
```

---

#### 矛盾解决汇总表

| 编号 | 矛盾描述 | 解决方案 |
|------|---------|---------|
| M1 | 换手惩罚阻塞战略再平衡 | `apply_turnover_penalty` 参数：月度=False（惩罚=0），战术=True（惩罚=0.3） |
| M2 | Reversal 不激活时权重自动重分配，制度切换产生伪换手 | 固定 5-权重方案（TSMOM 40%/CSMOM 25%/Carry 20%/FactorMAD 10%/Reversal 5%）；Reversal=0 时其 5% 保留为零贡献不重分配 |
| M3 | Harvey-Liu t>3.0 与 ICIR≥0.3 门控在 24 月数据下不兼容（ICIR=0.3→t≈1.47） | t-value 为展示指标，Layer 2 自动通过门控保持 ICIR≥0.3；Supervisor 看 t-value 做知情判断 |
| M4 | ERA 验证用同一决策模型存在同源偏差 | ERA 专用 `temperature=0.1`，且只接收宏观数据（无价格），与决策 agent 解耦 |
| M5 | WatchlistEntry 状态机与优化器入场逻辑冲突 | 删除 WatchlistEntry 状态机；纯量化优化器隐式处理入场；LLM 超配只创建 watching 状态入口，由优化器决定是否实际建仓 |
| M6 | 90 天年龄规则与信号质量无关（时间替代质量指标） | 用信号强度衰减替代：TSMOM raw_return 较入场峰值下降 >60% → 触发 REVIEW_SIGNAL_DECAY，Layer 3 审批 |

---

#### 新增文件清单

| 文件 | 功能 | 关键常量 |
|------|------|---------|
| `engine/track_b.py` | Track B LLM 权重叠加层 | `TRACK_B_BUDGET=0.05`, `MAX_RELATIVE_DELTA=0.25` |
| `engine/era.py` | 外部现实验证审计 | `ERA_TEMPERATURE=0.1`, `ERA_MACRO_SERIES` (6个FRED系列) |
| `engine/universe_review.py` | 季度宇宙筛查 | `EXIT_AUM=$500M`, `ENTRY_AUM=$1B`, `UNIVERSE_MAX_SIZE=40` |

#### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `engine/memory.py` | 新增 7 张 ORM 表（SignalSnapshot/RegimeSnapshot/SignalFlipLog/DataQualityLog/CircuitBreakerLog/AlphaMemory/QuantOnlySnapshot）；`_migrate_db()` 补迁移 |
| `engine/signal.py` | `compute_carry()` 资产类别分支；`compute_reversal()` 制度条件；`COMPOSITE_WEIGHTS` 固定 5 权重常量 |
| `engine/portfolio.py` | `_get_position_limits(regime)` 替换固定 MAX_LONG/MAX_SHORT；`apply_turnover_penalty` 参数；`_regime_adjusted_cov()` risk-off 30% 收缩 |
| `engine/daily_batch.py` | 新增 `_step1_data_quality()`/`_write_signal_snapshot()`/`_detect_signal_flips()`/`_patrol_signal_decay()`/`_patrol_vol_spike()`；月度 M1/M2/M3 分离；`_is_first_trading_day_of_quarter()`；`_launch_quarterly_jobs()` 后台线程 |
| `engine/factor_mad.py` | `compute_harvey_liu_t()`；`_bh_correction()`；`_check_factormad_budget()`；`FactorICIR` 新增 `regime/t_stat/n_months` 字段 |

#### 实施顺序建议

```
Phase 1（数据基础）→ Phase 2（信号精炼）→ Phase 3（组合构建）
  ↓
Phase 4（ERA 验证层，可并行 Phase 3）
  ↓
Phase 5（频率分离，依赖 Phase 1-3 完成）
  ↓
Phase 6（宇宙审查 + FactorMAD 增强，依赖 Phase 5）
```

Phase 1 是硬前置（SignalSnapshot 表是所有后续改进的数据基础）。Phase 4 的 ERA 与 Phase 3 无依赖，可并行。Phase 6 的宇宙审查依赖季度异步框架（Phase 5）。

---

### Defer — 等待 Clean Zone n≥30

#### Defer-1｜FinDebate 置信度校准（n≥30）
Dispersion Index 需先用 n≥30 验证数据画 calibration curve，确认与 `accuracy_score` 统计负相关（p<0.05）后再部署。

#### Defer-2｜TradingGPT 情节记忆（n≥30 + P2-1 验证）
新建 `EpisodicMemory` 表，每条 verified 决策压缩为结构化摘要 + 时间衰减权重。

#### Defer-3｜cvxpy 组合优化器（n≥30）
BL 风格优化：`minimize ||w - quant_baseline||² - λ × opinion^T w`，受 `WEIGHT_LIMITS` 约束。

#### Defer-4｜Kelly 仓位建议（n≥30）
n=0 时任何 Kelly 计算均为无效假精确。

#### Defer-5｜S4 反事实情景测试（n≥20）
Admin Post-Mortem Tab 人工触发的反事实测试，`engine/adversarial.py`。

#### Defer-6｜γ/λ 量化框架（n≥15）
`γ = μ / σ_target²` 基于历史 Clean Zone 决策推导；`λ` 稳定性准则（bootstrap 80% 非零）。

#### Defer-7｜PSR（n_months ≥ 50）
Probabilistic Sharpe Ratio，置信区间过宽时无意义。

#### Defer-8｜多模型真实辩论（n≥30 + Phase 2 完成）
引入 Claude Sonnet 或 GPT-4o 作红队，解决同源模型偏见；需有对照组才能判断是否改善判断质量。

#### Defer-9｜Track B 四层分层架构（Phase 2 + n≥30）
Layer 1（宏观资产类别观点）→ Layer 2（行业/风格偏好）→ Layer 3（ETF 验证）→ Layer 4（组合压力测试）。Black-Litterman 融合层依赖已校准的 LLM 置信度（Defer-1 完成后），在此之前 BL 融合挂起。

#### Defer-10｜Within-Class CSMOM（Phase 2 批次 B 完成后）
债券 ETF 和权益 ETF 独立截面排序；组合层再做资产类别间配置决策。

#### Defer-11｜FactorMAD 四分位验证（Universe ≥ 30 ETF）
18 个 ETF 每档仅 3-4 个，统计无意义；Phase 2 完成后（≥33 ETF）激活，每档 6-7 个初步可用。

#### Defer-12｜4 变量 MSM（双变量 MSM 验证后）
纳入利率曲线 + 信用利差 + VIX + 通胀预期，EM 参数估计不稳定性随维度升高，先验证双变量效果。

---

### 需重设计 — RAEiD 在线学习

当前 LangGraph 图只有 `red_team` 有方向性投票，其余节点无独立 vote。需先重构图（每节点输出 `{direction, confidence}`）→ 建 `AgentVoteLog` 表 → 实现 Bayesian 权重更新。工作量大，P0-P2 全部完成后再启动。

---

## 四、实施前置问题

| 编号 | 问题 | 阻塞任务 | 状态 |
|------|------|---------|------|
| PRE-1 | `memory.py` 是否有 `_migrate_db()` / `ALTER TABLE`？本地 DB 有真实数据？ | P0-3、P1-3、P2-11 | ⏳ |
| PRE-2 | `SignalSnapshot` 字段是否包含 `to_prompt_context_raw()` 所需全部字段？ | P0-4 | ⏳ |
| PRE-3 | 全库搜索 `build_agent_graph` 所有调用点 | P0-4、P1-2 | ⏳ |
| PRE-4 | `engine/history.py` 的 `SECTOR_ETF` 完整列表 + `backtest.py` 默认 `start_date` | P0-2 | ⏳ |
| PRE-5 | 实际调参历史（决定 `EFFECTIVE_N_TRIALS`） | P0-1 | ✅ 未系统扫描 → `EFFECTIVE_N_TRIALS = 6` |
| PRE-6 | 当前 Clean Zone n 值（verified=True + lcs_passed=True） | 所有 Defer 项 | ⏳ |
| PRE-7 | 根目录 / `tests/` 是否有测试套件？ | 所有 memory.py 改动 | ⏳ |
| **PRE-8** | **yfinance `Adj Close` 对债券 ETF（AGG/IEF/TIP）是否含票息再投资？验证后才能纳入批次 B** | P2-12 批次 B | ⏳ |
| **PRE-9** | **Finnhub API Key 是否已申请？** | P2-17 | ⏳ |
| **PRE-10** | **GNews API Key 是否已申请？** | P2-17 | ⏳ |

---

## 五、里程碑

| 里程碑 | 解锁条件 | 意义 |
|--------|---------|------|
| M0（当前） | — | P0 执行中，双轨刚接通 |
| M1 | P0-P1 全部完成 | 信号质量、归因框架就位；可开始 P2 |
| M2 | P2-11 + P2-12 批次 A 完成 | Universe ≥ 26 ETF，FactorMAD 框架可搭建 |
| M3 | P2-12 批次 B + P2-14 双变量 MSM + Universe ≥ 30 ETF | 四分位验证激活；Defer-1/2/3 解锁 |
| M4 | Clean Zone n≥50，Brier Score < 0.25 | PSR 激活，γ/λ 可校准，置信度校准有意义 |
| M5 | Track B 四层 + RAEiD 重设计 + FactorMAD 稳定运行 | 架构完整，具备学术论文发表条件 |

**当前位置：P5 全部 ✅（截至 2026-04-26）；P6 施工蓝图已设计（2026-04-27）；下一步：Phase 1 信号持久化层（SignalSnapshot + DataQualityLog）**

---

## 六、关键设计原则（不可随意修改）

1. **filtered probability only**：`regime.py` 只用 filtered，不用 smoothed
2. **ATR 止损用 trailing high**：stop = trailing_high - 2×ATR(21)，不用 cost_basis
3. **复合评分 gate**：< 35 分或 risk-off → blocked（P1-6 修改权重后需重新校准此阈值）
4. **幂等性**：DailyBatchJob 以 SignalSnapshot.as_of_date 为唯一 guard
5. **分析-操作缓冲**：LLM 超配分析只创建 WatchlistEntry(watching)，不直接触发交易
6. **自动学习门槛**：n<30 时任何自动参数更新 = 过拟合
7. **置信度≠准确率**：n<50 calibration 验证前 LLM confidence_score 只是标签
8. **两轨独立验证**：`quant_baseline_weight` 和 `llm_adjustment_pct` 必须始终分开记录
9. **注入规则硬约束**：`tsmom_signal`、`gate_status`、`composite_score` 永远不注入 LLM prompt
10. **杠杆/反向 ETF 硬排除**：路径依赖衰减使 TSMOM 信号失真，永久排除出 Universe
11. **FactorMAD 四层防御**：Layer 1 MI 扫描拦截统计前视 → Layer 2 Critic 辩论 → Layer 3 符号回归旁证 → Layer 4 人工裁决；Critic Agent 不能替代人工代码审查，候选因子代码必须人工确认无 t+1 数据使用

---

## 七、任务依赖关系图

```
P0-1 (DSR)                独立
P0-2 (Survivorship)       独立  →  P2-11 Universe管理前置
P0-3 (归因双轨)            独立  →  阻塞 P1-E
P0-4 (接通QuantAssessment) 独立  →  阻塞 P1-2

P1-A (WEIGHT_LIMITS统一)   →  阻塞 P1-D
P1-3 (risk_conditions)    →  阻塞 P1-4
P1-6 (QA-2权重重设计)      →  阻塞 P2-2 (TSMOM连续化)
P1-7 (QA-5独立vol)         →  阻塞 P2-4 (GARCH)

P2-11 (Universe框架)       →  阻塞 P2-12 (跨资产扩展)
P2-12 批次A                →  Universe ≥ 26，FactorMAD框架可建
P2-12 批次B                →  Universe ≥ 33，Defer-10/11激活
P2-14 (双变量MSM)          →  阻塞 Defer-12 (4变量MSM)
P2-1 (时间衰减检索)         →  阻塞 Defer-2 (EpisodicMemory)
P2-9 (学习阶段状态机)       →  统一所有 Defer 项的 n 门控
P2-17 (新闻升级)            →  阻塞 Defer-9 Track B Layer1的新闻输入质量

Defer-1 (置信度校准)        →  阻塞 Defer-9 (Track B BL融合层)
Defer-9 (Track B四层)      →  依赖 Phase 2完成 + n≥30

RAEiD 需重设计             独立大型任务，不阻塞其他项
FactorMAD 四分位验证        需 Universe ≥ 30 ETF（M3 解锁）
```

---

*本文档是 2026-04-27 v4 权威版本。整合来源已归档，此后所有改进开发以本文档为准。P6 施工蓝图（多频架构精炼）2026-04-27 写入，含矛盾解决 M1-M6 全部闭合。*
