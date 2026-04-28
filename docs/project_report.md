# Macro Alpha Pro — 项目技术报告

> **版本**：v3.0（2026-04-24）  
> **作者视角**：资深量化金融工程师 + 严谨学术标准  
> **定位**：NUS MSBA 研究项目，模拟盘优先，关键节点人工审批，不用于实盘交易

---

## 目录

1. [项目定位与研究问题](#1-项目定位与研究问题)
2. [系统总体架构](#2-系统总体架构)
3. [ETF Universe 管理](#3-etf-universe-管理)
4. [量化信号引擎](#4-量化信号引擎)
5. [宏观制度识别](#5-宏观制度识别)
6. [组合构建层](#6-组合构建层)
7. [回测引擎与性能评估](#7-回测引擎与性能评估)
8. [FactorMAD Alpha 因子挖掘](#8-factormad-alpha-因子挖掘)
9. [LLM Agent 层](#9-llm-agent-层)
10. [Alpha Memory 记忆与学习系统](#10-alpha-memory-记忆与学习系统)
11. [风险控制基础设施](#11-风险控制基础设施)
12. [新闻数据源架构](#12-新闻数据源架构)
13. [数据库设计](#13-数据库设计)
14. [UI 架构与各页面 Tab 详解](#14-ui-架构与各页面-tab-详解)
15. [合规与完整性机制](#15-合规与完整性机制)
16. [技术栈总览](#16-技术栈总览)
17. [当前状态与里程碑](#17-当前状态与里程碑)
18. [已知局限性与学术诚信披露](#18-已知局限性与学术诚信披露)

---

## 1. 项目定位与研究问题

### 1.1 核心研究问题

**宏观制度条件能否提升跨资产动量策略的风险调整收益？**

系统对比三个投资组合的历史表现：
- **Portfolio A**：无条件 TSMOM（时序动量，纯量化）
- **Portfolio B**：制度条件 TSMOM × 宏观制度滤波器（Hamilton MSM）
- **Benchmark**：1/N 等权重月度再平衡

研究工具：walk-forward 回测 + Deflated Sharpe Ratio（DSR，López de Prado 2018）+ BHY 多重检验 FDR 校正（Benjamini-Hochberg-Yekutieli）。

### 1.2 双轨独立验证框架

系统核心设计是 **Track A（量化轨）与 Track B（LLM 轨）的严格分离**：

```
Track A（纯量化）：TSMOM / CSMOM / FactorMAD
    → composite_score → quant baseline weight
    ↓ 仅注入原始数值（绝不注入方向性结论）
Track B（LLM）：macro_context + news + quant_context_raw
    → LLM 方向建议 → 人工闸门审批

双轨归因：
    llm_weight_alpha = actual_return_20d × (main_weight - quant_weight)
    main_weight  : LLM 调整后持仓（track="main"）
    quant_weight : 纯 TSMOM 基线（track="quant"）
```

**P0-4 注入规则（硬约束）**：`tsmom_signal`、`gate_status`、`composite_score`、`regime_label` 等方向性结论永远不注入 LLM prompt。只注入原始数值（`tsmom_raw_return`、`ann_vol`、`atr_14/63`、`price_vs_sma_200`、`p_risk_on`、`csmom_rank`），消除 Goodhart 效应与锚定偏差。

---

## 2. 系统总体架构

### 2.1 分层架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        UI Layer（Streamlit）                      │
│  app.py + st.navigation()  │  8 Pages  │  ui/tabs.py 6-Tab 套件  │
└──────────────────┬───────────────────────────┬────────────────────┘
                   │                           │
┌──────────────────▼───────────┐  ┌────────────▼──────────────────┐
│      Track B — LLM 层        │  │    Track A — 量化引擎层        │
│  LangGraph 5节点图           │  │  signal.py / regime.py        │
│  researcher → red_team       │  │  portfolio.py / backtest.py   │
│  → auditor → translator      │  │  factor_mad.py                │
│  → reflection（拦截时）      │  │  universe_manager.py          │
│  Gemini 2.5 Flash（号池）    │  │  circuit_breaker.py           │
└──────────────────┬───────────┘  └────────────┬──────────────────┘
                   │                           │
┌──────────────────▼───────────────────────────▼──────────────────┐
│                   Data / State Layer                              │
│  SQLite（SQLAlchemy ORM，17+ 张表）                              │
│  yfinance │ FRED API │ Finnhub │ GNews │ CBOE VIX               │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 启动流程（app.py）

```python
init_theme()                    # 深色/浅色主题初始化
get_pool() → model              # Gemini Key 池轮转
init_db()                       # SQLite schema + _migrate_db() 迁移
init_universe_db()              # ETF 注册表初始化（幂等）
seed_batch_a() / seed_batch_b() # 批次 A/B ETF 幂等写入
restore_today_from_db()         # 会话状态从 DB 恢复
ensure_daily_batch_completed()  # 每日批处理 6步状态机（幂等）
build_agent_graph(model, ...)   # LangGraph 图构建（闭包注入）
```

### 2.3 核心模块依赖关系

```
universe_manager ← history ← signal ← portfolio ← backtest
                                ↑           ↑
                           regime.py    factor_mad.py
                                ↑
                     macro_fetcher / news_fetcher
memory ← daily_batch ← quant_agent ← trading_schema
       ← agent (LangGraph 5节点图)
       ← orchestrator (TradingCycleOrchestrator)
```

---

## 3. ETF Universe 管理

### 3.1 设计动机

原系统使用 `history.py` 静态字典 `SECTOR_ETF`（18 ETF）。P2-11 替换为 SQLite 动态注册表：
1. 批量扩展无需改代码
2. 携带成立日期（survivorship bias audit 前提）
3. `asset_class` 分类支持 within-class CSMOM
4. `universe_health_check()` 月度流动性监控

向后兼容：`get_active_sector_etf()` 优先读 `UniverseETF` 表，失败时回退静态 `SECTOR_ETF` 字典。

### 3.2 ORM 模型

```python
class UniverseETF(Base):
    __tablename__ = "universe_etfs"
    sector         : String(100), unique   # 与 DecisionLog.sector_name 关联键
    ticker         : String(20)
    asset_class    : String(30)            # equity_sector / equity_factor /
                                           # fixed_income / commodity / volatility
    batch          : Integer               # 0=初始18 / 1=批次A / 2=批次B
    inception_date : Date                  # 上市日期，用于 survivorship audit
    active         : Boolean
    added_at       : Date
    removed_at     : Date
    notes          : String(200)
```

### 3.3 当前 Universe（32 个 ETF）

**Batch 0 — 初始 18 个 ETF**

| Ticker | 板块 | 资产类别 | 成立日期 |
|--------|------|---------|---------|
| SMH | AI算力/半导体 | equity_sector | 2000-05-05 |
| QQQ | 科技成长(纳指) | equity_sector | 1999-03-10 |
| XBI | 生物科技 | equity_sector | 2006-02-06 |
| XLF | 金融 | equity_sector | 1998-12-16 |
| XLE | 能源 | equity_sector | 1998-12-16 |
| XLI | 工业 | equity_sector | 1998-12-16 |
| XLV | 医疗 | equity_sector | 1998-12-16 |
| XLP | 消费必需 | equity_sector | 1998-12-16 |
| XLY | 可选消费 | equity_sector | 1998-12-16 |
| XLC | 传播服务 | equity_sector | 2018-06-18 ⚠️ 最年轻 |
| VNQ | 房地产 | equity_sector | 2004-09-23 |
| ICLN | 清洁能源 | equity_sector | 2008-06-24 |
| EWS | 新加坡 | equity_sector | 1996-03-12 |
| ASHR | 沪深300 | equity_sector | 2013-11-06 |
| KWEB | 中国互联网 | equity_sector | 2013-07-31 |
| GLD | 黄金 | commodity | 2004-11-18 |
| TLT | 美国长债 | fixed_income | 2002-07-22 |
| HYG | 高收益债 | fixed_income | 2007-04-04 |

**Batch A — 7 个因子/国际权益 ETF（P2-12 Batch A）**

| Ticker | 板块 | 成立日期 |
|--------|------|---------|
| IWN | 小盘价值 | 2000-07-24 |
| IWO | 小盘成长 | 2000-07-24 |
| MTUM | 动量因子 | 2013-04-16 |
| USMV | 低波动因子 | 2011-10-18 |
| QUAL | 质量因子 | 2013-07-16 |
| EWJ | 日本 | 1996-03-12 |
| INDA | 印度 | 2012-02-02 |

**Batch B — 7 个跨资产 ETF（P2-12 Batch B，PRE-8 验证通过）**

| Ticker | 板块 | 资产类别 | 特殊处理 |
|--------|------|---------|---------|
| AGG | 美国综合债 | fixed_income | Adj Close 含票息，误差 -0.17%（PRE-8 验收） |
| IEF | 美国中期国债 | fixed_income | Adj Close 含票息，误差 -0.36% |
| TIP | 通胀保值债 | fixed_income | Adj Close 含票息，误差 -0.38% |
| GDX | 黄金矿业 | commodity | — |
| DBA | 农产品 | commodity | — |
| VXX | 波动率 | volatility | **TSMOM 信号极性翻转** |
| REM | 抵押贷款信托 | equity_sector | — |

> **VXX 极性翻转**：VXX 跟踪 VIX 期货，正向动量代表市场恐慌上升而非趋势延续，TSMOM 信号取反。实现：`_REVERSE_MOMENTUM_TICKERS = {"VXX"}`，`raw_return > 0 → tsmom = -1.0`。

### 3.4 Survivorship Bias Audit（`engine/universe_audit.py`）

维护所有 32 个 ETF 的上市日期（来源：ETF 发行商招募说明书 / SEC EDGAR N-1A），在 `run_backtest()` 入口调用 `audit_universe(tickers, backtest_start)`。警告写入 `BacktestResult.warnings`，UI expander 展示，不中断回测。

### 3.5 纳入规则

```python
UNIVERSE_RULES = {
    "min_adv_usd": 10_000_000,     # 日均成交额 > $10M
    "min_history_years": 3,         # 成立 ≥ 3 年
    "hard_exclude_types": ["leveraged", "inverse", "money_market"],
    # 杠杆/反向 ETF 永久排除：路径依赖衰减使 TSMOM 信号失真
}
```

---

## 4. 量化信号引擎

### 4.1 时序动量（TSMOM）

**理论基础**：Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"，JFE。

**形成期窗口（12-1 月标准协议）**：
```
end_price   : as_of - skip_months(=1) 的最后收盘价
start_price : as_of - lookback_months(=12) 的最后收盘价
raw_return  : (end_price - start_price) / start_price
signal      : sign(raw_return) ∈ {-1, 0, +1}
```

跳过最近 1 个月原因（Jegadeesh & Titman 1993）：避免微观结构噪音（买卖价差反弹、做市商库存效应）导致的虚假短期反转信号。

**连续化改进（P2-2）**：原始二值改为跨截面 min-max 归一化的成形期 Sharpe 比率：
```python
sharpe_cs  = raw_return / ann_vol_21d.clip(lower=1e-6)
tsmom_norm = (sharpe_cs - min) / (max - min) * 100   # [0, 100]
# 中性资产（tsmom==0）锚定到 50，不被截面极值拉偏
```

### 4.2 截面动量（CSMOM）

**理论基础**：Moskowitz & Grinblatt (1999) "Do Industries Explain Momentum?"

**Within-Class 排序（P2-12 Batch B 激活）**：
不同资产类别收益驱动因子根本不同，按 `asset_class` 分组独立排序：
```python
for asset_class, class_map in get_universe_by_class().items():
    sub = df.loc[class_sectors]
    # top-tercile → +1, bottom-tercile → -1, middle → 0
```
`universe_manager` 不可用时回退全局排序。

**CSMOM 对 TSMOM 的截断修正（P1-6）**：CSMOM 不独立占权重，改为方向-排名冲突修正：
```
tsmom == +1 且 within-class 排名 < 33分位 → tsmom_norm 上限 70
tsmom == -1 且 within-class 排名 > 67分位 → tsmom_norm 下限 30
```

### 4.3 波动率估计（三层优先链）

| 优先级 | 方法 | 窗口 | 说明 |
|--------|------|------|------|
| 1 | GARCH(1,1) 条件波动率（P2-4） | 252日拟合，1步前向预测 | arch 库 |
| 2 | 实现波动率 21日（P1-7） | 独立 21 日窗口 | 独立于形成期 |
| 3 | 形成期年化波动率 | 与 TSMOM 同 12M 窗口 | 最终 fallback |

数据效率：一次下载 280 个交易日日收益率，同时供 21d 和 GARCH 使用。

**GARCH(1,1) 实现**：
```python
gm = arch_model(ret_garch * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
gres = gm.fit(disp="off")
fcast_var = float(gres.forecast(horizon=1).variance.iloc[-1, 0])
ann_vol_garch = sqrt(fcast_var / 10_000 * 252)   # (% daily)² → 年化小数波动率
```

### 4.4 复合评分（Composite Score，0-100）

**权重设计（P1-6 修订版）**：

| 因子 | 权重 | 计算方法 |
|------|------|---------|
| TSMOM（含 CSMOM 截断修正） | 50% | 连续化 Sharpe，截面 min-max [0,100] |
| Sharpe Ratio | 30% | Sigmoid：`100 / (1 + exp(-2 × sharpe_raw))` |
| 宏观制度 | 20% | `p_risk_on × 100`（MSM filtered probability） |
| FactorMAD（替换制度） | 20% | 当 ≥3 个 active 因子时替换 regime 槽位 |

**门控规则（Quant Gate，Goodhart-safe）**：
```
R1: tsmom=-1 AND csmom=-1  → 超配禁止
R2: tsmom=+1 AND csmom=+1  → 低配禁止
R3: composite < 20          → 仅低配/拦截允许
R4: composite > 80          → 低配禁止
R5: regime=risk-off         → 超配禁止（组合层面）
S1: 20 ≤ composite ≤ 35    → 软警告（不硬拦截）
```

---

## 5. 宏观制度识别

### 5.1 Hamilton Markov Switching Model（MSM）

**理论基础**：Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series"。

**主变量**：10Y-2Y 美债利差（月均值，FRED `DGS10 - DGS2`）。  
选择理由：收益率曲线是美国经济衰退最具预测力的单变量指标（Estrella & Hardouvelis 1991；Ang et al. 2006）。

**模型规格**：
```python
MarkovRegression(endog, k_regimes=k, trend="c", switching_variance=True)
# 切换均值 + 切换方差（Hamilton 1989 完整规格）
# k 由 BIC 在 {2, 3} 上选择；60-120 月数据 BIC 几乎总选 k=2
# 制度识别：risk_on_idx = argmax(regime_means_of_yield_spread)
```

**关键设计：filtered probability only**：
```python
# 仅用 filtered_marginal_probabilities，不用 smoothed
# Kalman 后向平滑使用未来数据，等价于引入前视偏差
filtered  = result.filtered_marginal_probabilities
p_risk_on = filtered.iloc[:, risk_on_idx].loc[as_of]
```

**制度分类（概率阈值 0.65）**：
```
p_risk_on ≥ 0.65 → "risk-on"（正常扩张）
p_risk_on ≤ 0.35 → "risk-off"（收缩/危机）
otherwise         → "transition"（模糊区间）
```

### 5.2 双变量 MSM（P2-14）

第二变量：FRED `BAMLC0A4CBBB`（ICE BofA BBB OAS 信用利差）；FRED 不可用时回退 LQD-IEF 月度收益差。

引入理由：10Y-2Y 捕获货币政策维度，信用利差捕获风险偏好/流动性维度。2022 年利率倒挂但商品暴涨的案例说明单变量制度识别存在系统性误判（错分率约 15-25%）。

```python
_combined = pd.DataFrame({"yield_spread": spread, "credit_spread": credit})
# BIC k 选择始终基于单变量 yield_spread（跨规格 BIC 不可比）
# 双变量拟合失败 → 回退单变量（try/except，版本兼容）
```

### 5.3 降级处理

| 条件 | 响应 | method 字段 |
|------|------|------------|
| n_obs < 60 | 规则基础分类 | "rule-based" |
| MSM 拟合不收敛 | 规则基础分类 | "msm-fallback" |
| 双变量对齐后 n < 60 | 回退单变量 | — |
| as_of 早于数据范围 | 规则基础分类 + warning | — |

**规则基础分类**（利差 + VIX 阈值 → sigmoid 概率）：
```python
score = 0.0
if yield_spread < -0.3: score -= 2.0   # 严重倒挂
elif yield_spread < 0:  score -= 1.0
if vix > 30:            score -= 2.0   # 危机水平
elif vix > 22:          score -= 0.8
p_risk_on = sigmoid(score)
```

### 5.4 Walk-Forward 协议

回测中每个再平衡日期 t 独立重新拟合（`train_end=t`），防止参数估计前视偏差。计算量为 O(n_dates) 次 MSM 拟合，是正确的学术协议。

---

## 6. 组合构建层

### 6.1 构建流程（6步，`engine/portfolio.py`）

**Step 1 — Inverse-Vol 原始权重**
```
raw_w_i = signal_i / σ_i   （σ 优先链：GARCH > 21d > 12M；signal=0 权重为 0）
```

**Step 2 — 单位总敞口归一化**
```
w_norm_i = raw_w_i / Σ|raw_w_j|
```

**Step 3 — 组合波动率估计**

优先路径（P2-3，Ledoit-Wolf 收缩）：
```python
lw = LedoitWolf().fit(returns_matrix[common_sectors])   # sklearn
cov_ann = lw.covariance_ * 252
port_var = w_vec @ cov_ann @ w_vec
# 条件：len(common) ≥ 3 且 len(ret) ≥ 60
```

降级路径（对角协方差，零相关假设）：
```
σ_port = sqrt(Σ (w_i × σ_i)²)
# 同时输出 ρ=0.5 上界警告：σ_upper ≈ σ_diag × sqrt(1 + 0.5×(n-1))
```

**Step 4 — 波动率目标化**
```python
scalar = min(TARGET_VOL / est_port_vol, MAX_LEVERAGE)   # 10% / σ，上限 2×
w_scaled = w_norm × scalar
```

**Step 5 — 制度覆盖**
```python
if regime == "risk-off":
    multiplier = REGIME_SCALE          # 0.30，多头压缩 70%
elif regime == "transition":
    multiplier = 0.30 + 0.70 × p_risk_on   # 线性插值
# 空头权重不变（飞向安全效应）
```

**Step 6 — 仓位上限**
```python
w_final = clip(w_scaled, -MAX_WEIGHT, +MAX_WEIGHT)   # 25% 上限
# 裁剪后比例重新调整，恢复总敞口意图
```

### 6.2 参数设定

| 参数 | 值 | 理论来源 |
|------|-----|---------|
| TARGET_VOL | 10% | Moreira & Muir (2017) |
| MAX_WEIGHT | 25% | 集中度约束 |
| MAX_LEVERAGE | 2× | 杠杆上界 |
| REGIME_SCALE | 0.30 | Ang & Bekaert (2004) |

---

## 7. 回测引擎与性能评估

### 7.1 Walk-Forward 协议（`engine/backtest.py`）

```python
for t in rebalancing_dates:
    signal_t  = get_signal_dataframe(as_of=t, use_cache=False)   # 严格无前视
    regime_t  = get_regime_on(as_of=t, train_end=t)              # 训练截至 t
    weights_t = construct_portfolio(signal_t, regime_t, returns_matrix_t)
    actual_ret_t = prices[t+1] / prices[t] - 1                   # 持有期实现收益
```

动态交易成本（P2-15）：
```python
cost_i = |Δw_i| × max(3bps, vol_14d_i × 0.15)
# 3bps 下限（大型 ETF 典型买卖价差）
# vol_14d × 0.15（Kyle's lambda 简化版，高波动期价差扩大代理）
# 基准组合使用相同动态 TC，保证比较公平
```

### 7.2 性能指标体系

| 指标 | 计算方法 | 来源 |
|------|---------|------|
| 年化收益 / 波动率 / Sharpe | 252日年化，标准公式 | — |
| **Deflated Sharpe Ratio** | Bailey & López de Prado (2014)，`EFFECTIVE_N_TRIALS=6` | López de Prado 2018 |
| 最大回撤 / Calmar Ratio | 滚动净值峰谷差 | — |
| 制度条件 Sharpe | risk-on 月 vs risk-off 月分层计算（P1-1） | — |
| 制度内回撤 / 胜率 | `drawdown_risk_on/off`，`hit_rate_risk_on/off` | P1-1 |
| Information Ratio | (策略月收益 - 基准) / tracking error | — |
| **BHY 多重检验校正** | FDR q-value，Benjamini-Hochberg-Yekutieli（P1-5） | — |
| Brier Score | `mean((P_direction - I_correct)²)`（P2-7） | — |
| avg_holding_months | 平均同方向信号连续持有月数（P1-1） | — |

**DSR**：
```python
EFFECTIVE_N_TRIALS = 6
# 保守估计（PRE-5：未系统网格扫描）
# ≈ 2 lookback × 2 vol_target × 1.5 skip ≈ 6
# 参考：Harvey, Liu & Zhu (2016)
```

---

## 8. FactorMAD Alpha 因子挖掘

### 8.1 设计定位

Track A 纯量化子系统（`engine/factor_mad.py`），与 LLM 流程零交互。目标：系统化地提出、验证、过滤新量化因子候选，消除人工挑选偏差。

### 8.2 四层防御架构

```
月度/季度触发
  │
  ▼ 【Layer 1】MI 污染扫描（纯统计，前置拦截）
  │  候选因子 MI > 基准因子均值 × 2.0 → 直接 reject，写 rejection_reason
  │  通过 → 节省后续 LLM 调用
  │
  ▼ 【Layer 2】Proposer-Critic 辩论（debate.py，LLM 驱动）
  │  验证集 IC/ICIR（最多 5 轮，连续 2 轮无改善提前终止）
  │  测试集最终验证（辩论全程隔离，不可见）
  │  通过条件：ICIR ≥ 0.3 + 与现有因子相关性 < 0.7
  │  → DiscoveredFactor（status=pending）
  │
  ▼ 【Layer 3】符号回归结构审计（gplearn，侦探报告）
  │  SymbolicRegressor 拟合候选因子 from 基准因子
  │  三态旁证：positive / neutral / danger（非二值门控，仅供 Supervisor 参考）
  │
  ▼ 【Layer 4】Supervisor 人工裁决（Admin UI）
     approve → active（composite_score 中 20% 槽位，权重上限 10%）
     reject  → rejected + rejection_reason
     defer   → pending_further_review
```

### 8.3 Layer 1：互信息污染扫描

```python
from sklearn.feature_selection import mutual_info_regression

# 滚动 24 个月度截面，每截面：
# X = 候选因子标准化截面值
# y = 22个交易日前向收益（严格无前视）
# MI = mutual_info_regression(X, y)   # k-NN 估计

candidate_mi  = compute_factor_mi(candidate_fn, prices, train_end)
baseline_mi   = mean([compute_factor_mi(f, ...) for f in _BASELINE_FACTOR_IDS])
ratio         = candidate_mi / (baseline_mi + 1e-9)
flagged       = ratio > 2.0   # _MI_CONTAMINATION_MULTIPLIER
```

4 个内置基准因子（无前视，白名单校准基准）：

| factor_id | 定义 |
|-----------|------|
| `mom_3m` | `prices[-65] / prices[-22] - 1`（3月动量，跳最近1月） |
| `rev_1m` | `-(prices[-1] / prices[-22] - 1)`（1月反转） |
| `vol_adj_mom_6m` | `ret_6m / vol_21d`（波动率调整6月动量） |
| `trend_strength` | `price / sma_120 - 1`（价格 vs 120日均线） |

### 8.4 Layer 3：符号回归审计

```python
from gplearn.genetic import SymbolicRegressor
sr = SymbolicRegressor(
    population_size=1000, generations=20,
    function_set=["add","sub","mul","div","sqrt","log","abs","neg"],
    metric="spearman",            # 与 IC 计算口径一致
    parsimony_coefficient=0.01,   # 惩罚复杂公式
    random_state=42,
)
sr.fit(X_baseline, y_candidate)   # X=基准因子矩阵，y=候选因子值

# 三态分类：
if r2 > 0.3 and keyword_overlap(formula, description):
    signal_type = "positive"    # 结构与声称逻辑一致
elif r2 > 0.3:
    signal_type = "danger"      # 高拟合但结构不匹配（可能是隐性拷贝）
else:
    signal_type = "neutral"     # 低拟合，符号回归无法复现
```

### 8.5 生产因子 IC/ICIR 月度监控

```python
def update_icir(calc_date, asset_class="equity_sector"):
    # 计算所有 active 因子的月度 Spearman IC（截面）
    # 写入 factor_icir 表（UniqueConstraint: factor_id × calc_date × asset_class）
    # 滚动 12 月 ICIR < 0.15 连续 2 月 → 自动标记 inactive
```

---

## 9. LLM Agent 层

### 9.1 LangGraph 图结构（`engine/agent.py`）

```
researcher → red_team → auditor → [is_robust?]
                                   ├─ True  → translator → END
                                   └─ False → reflection → END
```

### 9.2 各节点分工

| 节点 | 角色 | 关键输入 | 关键输出 |
|------|------|---------|---------|
| `researcher` | 蓝队：宏观+板块分析 | macro_context, news_context, quant_context_raw, position_context | technical_report |
| `red_team` | 红队：压力测试和反驳 | technical_report, quant_context_raw | red_team_critique |
| `auditor` | 仲裁：综合判断 | technical_report, red_team_critique | is_robust（最终） |
| `translator` | 翻译：结构化决策备忘录 | technical_report, red_team_critique | audit_memo |
| `reflection` | 反思：拦截情形替代建议 | full_state | alternative_suggestion, reflection_chain |

### 9.3 AgentState 关键字段

```python
class AgentState(TypedDict):
    target_assets      : str    # 分析标的
    vix_level          : float  # 实时 VIX
    macro_context      : str    # researcher 节点生成的宏观叙述
    news_context       : str    # P2-17 三层新闻时效衰减摘要
    position_context   : str    # P1-2 当前模拟持仓状态
    quant_context_raw  : str    # P0-4 原始量化数值（无方向性结论）
    is_robust          : bool   # auditor 最终判断
    audit_memo         : str    # translator 产出的决策备忘录
    reflection_chain   : str    # reflection 完整推理链
```

### 9.4 Gemini Key 池（`engine/key_pool.py`）

```
轮转：quota/429 错误 → report_quota_error()
      达 QUOTA_FAILS_BEFORE_SWITCH（默认3次）→ 自动换 Key
      AllKeysExhausted → UI 提示添加 Key

熔断：EmptyOutputCircuitBreaker：连续 2 次空输出 → 切换 Key
      BillingProtectionError：累计调用 > 阈值 → 停止

配置：st.secrets["GEMINI_POOL"]（多 Key）或 st.secrets["GEMINI_KEY"]（单 Key）
```

### 9.5 P0-4 量化注入规范

`QuantAssessment.to_prompt_context_raw()` 只输出：
```
tsmom_raw_return, ann_vol, atr_14, atr_63,
price_vs_sma_200, p_risk_on, csmom_rank
```
永久屏蔽：`tsmom_signal`（±1）、`gate_status`、`composite_score`、`regime_label`、任何方向性文字。

---

## 10. Alpha Memory 记忆与学习系统

### 10.1 DecisionLog 核心字段

**元数据**：`tab_type, sector_name, ticker, direction, horizon, macro_regime`

**LLM 自评**：`confidence_score(0-100), economic_logic, invalidation_conditions`

**系统完整性**：`model_version, prompt_version(SHA-256前8位), chain_hash`

**性能验证**（20日后 yfinance 回填）：`actual_return_5d/10d/20d, accuracy_score(0-1)`

**LCS 质量门控**：
```
lcs_score ≥ 0.70 → lcs_passed=True → 可写入学习表
lcs_mirror_passed : 信号全反转测试（逻辑对称性）
lcs_noise_passed  : 噪音注入测试（信号稳健性）
lcs_cross_passed  : 跨周期锚定测试（历史自洽性）
lcs_passed=False  → 不写 LearningLog/QuantPatternLog/NewsRoutingWeight/SkillLibrary
```

**三障碍法验证**：
```
TP：方向正确 1σ 移动（252日历史vol）
SL：方向相反 0.7σ 移动
时间障碍：2 × horizon 半衰期
barrier_hit: "tp"（strong_correct）| "sl"（clear_wrong）| "time"（time_decayed）
```

**双轨归因（P0-3）**：
```python
llm_weight_alpha = actual_return_20d × (main_weight - quant_weight)
```

### 10.2 时间衰减检索（P2-1）

```python
weight = exp(-ln(2)/90 × days_old) × accuracy_score
# 候选池 n×4 条 → 排序后取 top n → 注入 prompt
```

### 10.3 学习阶段状态机（P2-9）

```
S1 (n_verified < 10)  : 数据积累期，所有统计学习 Defer 项锁定
S2 (10 ≤ n < 30)      : 初步校验期，可做描述性统计
S3 (30 ≤ n < 50)      : 统计可信期，Defer-1/2/3/4/5/6/11 解锁
S4 (n ≥ 50)           : 完整研究期，PSR 激活，置信度校准有意义
```

### 10.4 实验日志哈希链（P2-10）

```python
payload = f"{id}|{created_at}|{direction}|{confidence_score}|{prev_hash}"
chain_hash = sha256(payload.encode()).hexdigest()[:32]
```

Admin UI 验证按钮：逐条重算，发现断点 → 报告截断位置。

### 10.5 知识库表

| 表名 | 用途 |
|------|------|
| `LearningLog` | 元 Agent 识别的系统性偏差（sector bias 等） |
| `SkillLibrary` | 高质量分析模式，自动更新，注入下次 prompt |
| `QuantPatternLog` | tsmom × csmom × regime 组合的历史准确率矩阵 |
| `NewsRoutingWeight` | 按 sector × regime 动态调整新闻来源权重 |
| `known_failures` | 已知失败模式，注入 prompt 防止重复 |

---

## 11. 风险控制基础设施

### 11.1 CircuitBreaker 三级状态机（P2-16，`engine/circuit_breaker.py`）

| 级别 | 触发条件 | 响应 | 持久化 |
|------|---------|------|-------|
| LIGHT | 单一数据源失效 | 行内警告，调用方降级 | 否（无状态） |
| MEDIUM | 当日 LLM 配额 > 80% | 暂停非核心 LLM 调用 | 否（自动重置） |
| SEVERE | VIX 单日涨幅 > 30% | 中止所有自动信号生成 | 是（JSON文件，跨重启保持） |

SEVERE 解除：`manual_reset(reason)` — 必须填写恢复理由。

### 11.2 Daily Batch 6步状态机（`engine/daily_batch.py`，幂等）

```
Step 1 新鲜度检查：SignalSnapshot.as_of_date 已存在 → 跳过
Step 2 信号 & 制度：QuantAgent 运行，写 SignalSnapshot / RegimeSnapshot
Step 3 Watchlist 巡检：评估 invalidation_conditions（watching 状态）
Step 4 入场检查：entry_condition 达成 → trigger_ready
Step 5 仓位巡检：硬止损 / 信号反转 / 制度压缩
Step 6 月末检查：最后交易日 → 生成 rebalance 订单送 PendingApproval
```

**ATR 止损（P0-A）**：`stop = trailing_high - 2 × ATR(21)`，用 trailing high（不用成本价）。

### 11.3 人工闸门（4个硬审批点）

```python
GATE_ANALYSIS_DRAFT      = "analysis_draft"
GATE_RISK_APPROVAL       = "risk_approval"
GATE_MONTHLY_REBALANCE   = "monthly_rebalance"
GATE_COVARIANCE_OVERRIDE = "covariance_override"
```

任何 trigger_ready 条目不会自动执行，须经 `approve_gate()` 人工确认。

### 11.4 动态交易成本（P2-15）

```python
cost_i = |Δw_i| × max(3bps, vol_14d_i × 0.15)
# Walk-forward 循环追踪 _w_sector_prev，19日窗口计算 ATR
```

---

## 12. 新闻数据源架构

### 12.1 三层降级设计（P2-17，`engine/news_fetcher.py`）

```
Layer 1: Finnhub /company-news      — 结构化情绪分数，60次/分钟
         Key: st.secrets["FINNHUB_KEY"]

Layer 2: GNews API（Layer 1 < 3条时触发）
         关键词："{sector} ETF {ticker} market"
         Key: st.secrets["GNEWS_KEY"]，100次/天

Layer 3: yfinance.news（Layer 1+2 均空时触发）
         零 API Key，终极备用，内容质量最低
```

### 12.2 时效性加权摘要

```python
time_w  = exp(-ln(2)/1.5 × days_old)      # 半衰期 1.5 天
tier_w  = {1: 1.0, 2: 0.7, 3: 0.4}[tier]
score   = time_w × tier_w × relevance_score
# 按 score 降序拼接，限 max_chars=1200
```

### 12.3 P0-4 新闻合规

情绪分数标注：`[情绪: +0.32]`（原始 VADER/TextBlob 数值）。禁止添加"建议买入/超配"等方向性文字。

---

## 13. 数据库设计

### 13.1 ORM 框架

SQLAlchemy `DeclarativeBase`，SQLite（开发，`macro_alpha_memory.db`）/ PostgreSQL（生产，`DATABASE_URL` 环境变量）。`expire_on_commit=False` 防止提交后访问触发额外查询。

### 13.2 迁移策略

`_migrate_db()` 使用 `ALTER TABLE ADD COLUMN`（try/except 忽略 OperationalError），每次 `init_db()` 运行，保证生产 DB 无缝升级不丢数据。

### 13.3 主要表清单

| 表名 | 主要用途 | 关键约束 |
|------|---------|---------|
| `decision_logs` | LLM 决策全量记录（40+ 字段） | — |
| `simulated_positions` | 模拟持仓（双轨） | UniqueConstraint(snapshot_date, sector, track) |
| `signal_snapshots` | 量化信号 24h 缓存 | 历史日期 only |
| `regime_snapshots` | MSM 制度状态缓存 | — |
| `watchlist_entries` | 交易监视列表 | status 4态 |
| `pending_approvals` | 人工闸门 | gate_type + status |
| `backtest_results` | 回测结果快照 | — |
| `structured_backtest_returns` | walk-forward 月度收益序列 | — |
| `universe_etfs` | ETF 动态注册表 | sector UNIQUE |
| `factor_definitions` | FactorMAD 因子注册 | factor_id UNIQUE |
| `factor_icir` | IC/ICIR 月度记录 | UniqueConstraint(factor_id, calc_date, asset_class) |
| `discovered_factors` | 候选因子审批流 | status 4态 |
| `learning_logs` | 元 Agent 偏差记录 | — |
| `skill_library` | 成功分析模式 | skill_name UNIQUE |
| `quant_pattern_logs` | 量化信号准确率矩阵 | — |
| `news_routing_weights` | 新闻来源权重 | (sector, regime, source_type) |
| `cycle_states` | Orchestrator 周期状态 | — |

---

## 14. UI 架构与各页面 Tab 详解

### 14.1 导航结构（`st.navigation`）

```python
{
    "Daily Overview": [
        "Market Snapshot"    → pages/live_dashboard.py
        "Signal Dashboard"   → pages/signal_dashboard.py
        "Regime Analysis"    → pages/regime_analysis.py
    ],
    "Research": [
        "Research Workbench" → app.py _page_research()  [default]
    ],
    "Execution": [
        "Trading Desk"       → pages/trading_desk.py
    ],
    "Review": [
        "Decision Review"    → pages/admin.py
        "Backtest"           → pages/backtest.py
        "Factor Dashboard"   → pages/factor_dashboard.py
    ],
    "System": [
        "Key Pool Manager"   → pages/key_manager.py
    ]
}
```

---

### 14.2 Research Workbench（6 Tab，`app.py _page_research()`）

#### Tab 1 — Macro Intelligence（宏观智能分析）

**定位**：全局宏观观点的 LLM 生成入口，每日一次分析，结果缓存到 session_state 和 DB。

**输入数据流**：
```
FRED 经济数据（P2-5）：CPI/PCE/非农/10Y收益率/利差等 9 个系列 MoM 趋势
↓
上期未解决监控清单（_build_watchlist_context()，最多 8 条）
↓
历史记忆上下文（时间衰减检索，top 5，半衰期 90 天）
↓
NewsPerceiver 全球宏观新闻（SPY 作标的，AV_KEY + GNEWS_KEY）
↓
augmented_ctx = [econ_ctx, watchlist_ctx, hist_ctx, news_ctx]
↓
LLM（researcher → red_team → auditor → translator）
```

**LLM 输出 6 节标准化格式**：
1. 全球宏观环境综述
2. 美国经济周期定位
3. 流动性与货币政策
4. 风险事件与尾部风险
5. 资产配置框架（方向性，需量化约束前提）
6. 监控清单（带截止日期的待观察项目）

**关键机制**：
- 分析前运行 `expire_overdue_watch_items()`，自动关闭过期监控项
- 输出解析为 `WatchItem` 列表（`parse_watch_items_from_memo()`）写入 DB
- `overwrite=True` 模式：`supersede_decision()` 软删除当日旧记录
- Levenshtein edit ratio 追踪用户对 LLM 输出的修改幅度
- PDF 报告生成（`generate_pdf_report()`）

**重新编排建议**：
- FRED 经济数据单独展示为数值表格，不仅注入 prompt
- 监控清单可视化为 Timeline 卡片，支持手动 resolve 和补充
- 历史相似制度的 top3 决策对比（同 macro_regime 检索）

---

#### Tab 2 — Live Dashboard（实时市场快照）

**定位**：量化信号的实时展示面板，纯数据驱动，无 LLM 调用。

**主要展示内容**：
- 全 Universe 信号表（sector / ticker / raw_return / ann_vol / tsmom / csmom / composite_score）
- 实时 VIX 指标 + 压力级别色带（COMPLACENCY / NORMAL / ELEVATED / CRISIS）
- 组合权重分布（inverse-vol 归一化后，含制度覆盖效果对比前后）
- 制度概率仪表盘：p_risk_on 仪表，risk-on / transition / risk-off 三分区
- 持仓变化：当日权重 vs 上期权重差异 delta

**数据来源**：`get_signal_dataframe(as_of=today)`，结果 24h 缓存（`st.cache_data`）。

**重新编排建议**：
- composite_score 柱状图改为热力图（sector × 因子贡献分解：TSMOM 50% / Sharpe 30% / Regime 20%）
- TSMOM vs CSMOM 方向一致性矩阵（一致绿/冲突红/中性灰 三色）
- 制度仪表盘加历史 p_risk_on 滚动时序图（12 个月）

---

#### Tab 3 — Regime Risk（制度风险分析）

**定位**：MSM 宏观制度检测结果的可视化与风险解读。

**主要展示内容**：
- Hamilton MSM 当前 `p_risk_on` 仪表 + 制度标签
- 10Y-2Y 利差时序图 + 零线 + 历史 NBER 衰退阴影
- 信用利差（P2-14）时序图：BAMLC0A4CBBB 或 LQD-IEF 代理，标注数据来源
- VIX 月均值时序图
- RegimeSnapshot 历史色条：按月展示 p_risk_on 色谱（蓝=risk-on，红=risk-off）
- MSM 诊断信息：method / n_obs / warning 字段展示
- 侧边栏 VIX 压力测试：覆盖实时 VIX → 制度信号实时重算

**重新编排建议**：
- 加"制度概率走廊"：rolling 12M p_risk_on 均值 ± 1σ
- 历史制度分类准确性（与 NBER 衰退期对比，展示制度一致率）
- Regime × Sector 平均收益矩阵（risk-on/off 各板块历史均值收益表）

---

#### Tab 4 — Quant Audit（量化审计）

**定位**：向用户暴露完整量化信号计算细节，用于 debug 和双轨研究验证。

**主要展示内容**：
- `QuantAssessment` 详情面板：每个板块的 `tsmom_raw_return / ann_vol / atr_14/63 / price_vs_sma_200 / p_risk_on / csmom_rank`（注入 LLM 的原始数值）
- 量化门控状态表：每个板块的 R1-R5 触发情况 + allowed/blocked 方向 + severity
- composite_score 分解：TSMOM 50% / Sharpe 30% / Regime(or FactorMAD) 20% 三层贡献数值
- CSMOM within-class 分组排序展示（equity_sector / equity_factor / fixed_income / commodity / volatility）
- VXX 信号极性翻转标注
- 波动率估计三层对比：GARCH / 21d / 12M 三种方法的实际值及当前选择

**重新编排建议**：
- 加入"注入 LLM 的原始字符串预览"（`to_prompt_context_raw()` 实际输出）
- 波动率三种估计方法的截面散点对比图
- Gate 规则历史触发频率统计（R1-R5 各被触发多少次）

---

#### Tab 5 — Sector Research（板块研究）

**定位**：单板块深度分析入口，整合 LLM 辩论 + 量化一致性检验 + XAI + 三层新闻。

**主要展示内容**：

**板块选择与上下文**：
- 全 Universe 32 ETF 选择器
- 当前持仓状态（`build_position_context()`）
- 三层新闻摘要（P2-17）：Finnhub 情绪分数 + GNews 关键词 + yfinance 备用

**LLM 辩论流程**（`engine/debate.py`）：
- Blue Team：结合 `quant_context_raw + news_context + position_context + gate_constraint` 生成初步分析
- Red Team：压力测试，逐条反驳 Blue Team 论点
- 迭代修订：最多 5 轮，连续 2 轮无实质改善提前终止
- Arbitration：仲裁节点综合双方，生成最终 `audit_memo`
- 辩论记录：写入 `debate_transcript`（JSON 完整历史，`debate_history / arb_notes / blue_output`）

**量化一致性检验**（`run_quant_coherence_check()`）：
- 检验 LLM 分析结论与 TSMOM/CSMOM 信号是否存在严重冲突
- 冲突时生成 coherence warning，供用户判断

**XAI 置信度分解面板**：
- `macro_confidence / news_confidence / technical_confidence` 三维评分
- 整体 `confidence_score`（0-100）
- `sensitivity_flag`（LOW/MEDIUM/HIGH）：输入噪音对结论的影响程度

**决策保存**：
- `save_decision()` → `WatchlistEntry` 写入（watching 状态）
- 成功后展示引导横幅，跳转到 Trading Desk（P1-C）

**重新编排建议**：
- 辩论记录展开/折叠交互式 timeline（Blue → Red → Arbitration 可视化）
- XAI 三维雷达图（macro/news/technical）
- "量化信号 vs LLM 方向"一致性矩阵高亮（conflict 时红色警告）
- 当前板块历史 accuracy_score 分布（过去 n 次的胜率）

---

#### Tab 6 — Alpha Memory（知识记忆）

**定位**：系统历史决策记录、学习状态和知识积累的可视化面板。

**主要展示内容**：

**学习阶段进度**（P2-9）：
- S1/S2/S3/S4 阶段色带进度条
- n_verified / n_clean 实时计数
- 当前阶段解锁的功能列表 vs 仍在 Defer 的功能

**历史决策表**：
- 分页展示 DecisionLog，支持 sector/tab_type/direction/date 筛选
- 每条展开：direction / accuracy_score / lcs_passed / barrier_hit / llm_weight_alpha
- LCS 详情：mirror/noise/cross 三个分项评分 + lcs_notes 诊断说明
- 三障碍法结果：barrier_hit 标签 + barrier_days + barrier_return
- 人工标注入口：black_swan / analysis_error / correct_call

**知识积累展示**：
- SkillLibrary 最近更新的成功模式（top 5 by usage_count）
- known_failures 摘要（防止重蹈覆辙的已知问题列表）
- 哈希链完整性状态（ok条数 / broken位置）

**统计面板**：
- 按 direction 分组胜率（超配 vs 标配 vs 低配）
- 按 horizon 分组胜率（短期/中期/长期）
- 按 regime 分组胜率（risk-on / transition / risk-off）

**重新编排建议**：
- 滚动 30 日准确率曲线（胜率时序图）
- LLM Alpha 归因散点图（main_weight vs quant_weight，按 accuracy_score 着色）
- regime × sector 准确率热力图（哪些组合 LLM 表现最好/最差）
- Clean Zone 进度计数器（距离 n=30 还差多少条，进度条展示）

---

### 14.3 Signal Dashboard（`pages/signal_dashboard.py`）

独立的实时信号快照页，适合每日快速浏览。

**主要内容**：
- 全 Universe 信号热力图（sector × 信号指标）
- TSMOM 三分排名列表（做多/做空/中性分组）
- Within-class CSMOM 分组展示（5 个 asset_class 独立排序结果）
- 波动率矩阵：GARCH / 21d / 12M 三层并列对比

---

### 14.4 Regime Analysis（`pages/regime_analysis.py`）

比 Research Workbench Tab 3 更深入的制度专页。

**主要内容**：
- MSM 模型诊断：BIC 曲线（k=2 vs k=3）、收敛状态、n_obs 样本量警告
- 双变量 MSM（P2-14）vs 单变量结果对比面板
- filtered probability 完整时序图（带制度标签分区）
- 制度转换矩阵：risk-on → transition → risk-off 历史转换频率
- 规则基础 vs MSM 结果对比（方法一致时高亮，分歧时提示用户）

---

### 14.5 Trading Desk（`pages/trading_desk.py`）

执行层页面，整合 Watchlist 管理、入场检查、模拟交易审批。

**主要内容**：
- 4-metric 状态条（P1-B）：信号状态 / 制度状态 / 止损触发 / 入场触发
- Watchlist 条目列表（watching / trigger_ready / active / closed 分组展示）
- 入场条件实时评估：entry_condition 达成 → 高亮 trigger_ready
- PendingApproval 审批操作：approve/reject + 原因填写
- 模拟持仓摘要：main 轨（LLM 调整）vs quant 轨（纯 TSMOM）权重对比
- RiskCondition 展示（P1-3）：vol_spike / drawdown / regime_cap 三种约束
- audit_target_sync：从 Admin Decision Review 跳转时自动预选板块（P1-F）

---

### 14.6 Decision Review（`pages/admin.py`，4 Tab）

#### Admin Tab 1 — Performance（绩效分析）

**主要内容**：
- Clean Zone 统计：n_verified / n_clean / accuracy_rate / avg_brier_score
- Horizon 分层表（P2-6）：按 horizon 分组展示 n / 胜率 / 均分 / 平均持仓天数 / Brier Score
- BHY 多重检验校正（P1-5）：TSMOM 和 Regime TSMOM 的 Sharpe p-value + q-value + 显著性判断
- 量化信号-准确率矩阵：tsmom × csmom × regime 组合的历史胜率表
- 时序胜率图：滚动 10 个决策的移动平均准确率

#### Admin Tab 2 — Decisions（决策记录）

**主要内容**：
- 决策列表（分页，支持 sector/tab_type/direction/date 筛选）
- 每条详情展开：完整 ai_conclusion + LCS + 三障碍 + llm_alpha + model_version
- 人工标注：`set_human_label()`（black_swan / analysis_error / correct_call）
- needs_review 标记的决策高亮展示
- 决策修订链（`get_revision_chains()`）：展示多次分析的迭代演进历史

#### Admin Tab 3 — System（系统管理）

**主要内容**：
- **学习阶段进度条（P2-9）**：S1→S4 色带 + 当前解锁状态
- **CircuitBreaker 状态色带（P2-16）**：当前级别 + 原因 + 手动恢复表单（SEVERE 时需填恢复理由）
- **Universe 管理面板（P2-11）**：ETF 注册表（sector / ticker / asset_class / batch / inception / active），"运行月度健康检查"按钮（`universe_health_check()`）
- **哈希链完整性验证（P2-10）**：验证按钮 + ok 条数 / broken 位置展示
- **TradingCycleOrchestrator 历史（P2-18）**：cycle_type / step / started_at 运行记录
- **Key Pool 状态**：当前活跃 Key + 各 Key 今日调用次数 + 配额错误次数

#### Admin Tab 4 — Watchlist（监视列表）

**主要内容**：
- Watchlist 条目全量（state 过滤器：watching / trigger_ready / active / closed）
- 未解决 watch items（宏观分析 §6 监控清单中的待观察项）
- 快速 resolve 操作（`resolve_watch_item()`）+ 过期项自动关闭记录

---

### 14.7 Backtest（`pages/backtest.py`）

Walk-forward 回测的运行入口和结果展示。

**主要内容**：
- 参数配置：start_date / lookback_months / skip_months / vol_target / max_weight
- Survivorship Bias 警告 expander（`audit_universe` 结果，每个超龄 ETF 附 gap_days）
- 三策略对比表：TSMOM / Regime TSMOM / Benchmark 的完整 BacktestMetrics
- BHY 校正 expander：两个策略的 Sharpe p-value / q-value / 显著性判断
- 制度条件绩效分拆：risk-on 期 vs risk-off 期 Sharpe 并排对比
- 月度收益时序图（策略净值曲线 vs 基准）
- 动态交易成本分析（成本 vs 固定 10bps 对比）

---

### 14.8 Factor Dashboard（`pages/factor_dashboard.py`，6 Tab）

| Tab | 内容 |
|-----|------|
| **Factor Overview** | 生产因子 IC 时序图 / ICIR 趋势线 / 状态表（factor_id / active / 最新 ICIR） |
| **Correlations** | active 因子间 Spearman 相关性热力图（滚动 12M），>0.7 标注⚠️ |
| **Parameters** | 因子 lookback 窗口敏感性分析（lookback vs IC 散点） |
| **Alpha Mining** | FactorMAD 生产因子 ICIR 月度汇总 + "更新 ICIR"按钮 + 自动 inactive 记录 |
| **Regime Sensitivity** | risk-on 期 IC vs risk-off 期 IC 分层统计，识别制度依赖性因子 |
| **FactorMAD 审批** | 候选因子三态裁决面板（Layer 1 MI 比率 + Layer 2 ICIR + Layer 3 符号回归报告）+ Approve/Reject/Defer 操作 + 历史记录表 |

---

### 14.9 Key Pool Manager（`pages/key_manager.py`）

- Gemini Key 列表（masked 显示）+ 各 Key 状态（active / exhausted）
- 今日调用量 / 配额错误次数 / 空输出熔断次数
- 添加新 Key 表单
- 手动重置单个 Key 配额计数

---

## 15. 合规与完整性机制

### 15.1 注入规则层级

| 数据类型 | 可注入 LLM | 原因 |
|---------|-----------|------|
| `tsmom_raw_return`（原始收益率） | ✅ | 原始数值，无方向性 |
| `ann_vol`（年化波动率） | ✅ | 原始数值 |
| `p_risk_on`（制度概率 0-1） | ✅ | 原始数值 |
| `csmom_rank`（排名百分位） | ✅ | 原始数值 |
| `atr_14/63`（ATR 绝对值） | ✅ | 原始数值 |
| `price_vs_sma_200`（价格/均线比） | ✅ | 原始数值 |
| `tsmom_signal`（±1 信号） | ❌ | 方向性结论，引入锚定偏差 |
| `gate_status`（通过/拦截） | ❌ | 方向性结论 |
| `composite_score`（合成分） | ❌ | 方向性结论 |
| `regime_label`（risk-on/off） | ❌ | 方向性结论 |
| "建议买入/超配" 等文字 | ❌ | 破坏双轨独立性 |

### 15.2 FactorMAD 特殊约束

- 候选因子代码**必须人工代码审查**：Critic Agent 无法检测 `t+1` 数据使用等代码层面前视偏差
- 测试集在 Layer 2 辩论全程隔离，辩论节点只能访问验证集
- ICIR 阈值配合 DSR 校正（`EFFECTIVE_N_TRIALS=6`），不用裸阈值
- 单一挖掘因子权重上限 10%（`weight_cap=0.10`）

---

## 16. 技术栈总览

| 层次 | 技术 | 用途 |
|------|------|------|
| UI 框架 | Streamlit（`st.navigation`，`st.cache_data`） | 多页面 + 状态管理 |
| LLM 模型 | Google Gemini 2.5 Flash | Agent 所有节点 |
| Agent 框架 | LangGraph（StateGraph） | 5节点 DAG 图 |
| 量化数据 | yfinance（`auto_adjust=True`，Adj Close） | ETF 日收益 + 新闻 Layer 3 |
| 宏观数据 | FRED API（直接 CSV 端点） | 利差、CPI、PCE 等 |
| 新闻 Layer 1 | Finnhub API | 情绪分数新闻（60次/分钟） |
| 新闻 Layer 2 | GNews API | 关键词精准检索（100次/天） |
| 时序分析 | statsmodels `MarkovRegression` | Hamilton MSM |
| 条件波动率 | arch `arch_model` | GARCH(1,1) |
| 协方差收缩 | sklearn `LedoitWolf` | 组合波动率估计 |
| 因子挖掘 | gplearn `SymbolicRegressor` | Layer 3 符号回归 |
| 互信息估计 | sklearn `mutual_info_regression` | Layer 1 MI 污染扫描（k-NN 估计） |
| 数据库 ORM | SQLAlchemy `DeclarativeBase` | 17+ 张表 |
| 数据库引擎 | SQLite（开发）/ PostgreSQL（生产） | `DATABASE_URL` 环境变量切换 |
| 交易日历 | pandas_market_calendars（NYSE） | 月末判断，有 weekday 回退 |
| 数值计算 | NumPy / Pandas / SciPy | 全栈 |
| 哈希完整性 | Python hashlib SHA-256 | 实验日志防篡改链 |

---

## 17. 当前状态与里程碑

### 17.1 已完成模块

| 批次 | 项目 | 状态 |
|------|------|------|
| P0 全部（4项） | DSR校准、Survivorship Audit、双轨归因、QuantAssessment注入 | ✅ |
| P1 全部（13项） | WEIGHT_LIMITS统一、Daily反馈、决策流、制度巡检、LLM Alpha归因、UX跳转、制度指标、持仓注入、RiskCondition、连续权重、BHY、权重重设计、独立Vol窗口 | ✅ |
| P2-1 ~ P2-10 | 时间衰减检索、TSMOM连续化、LW协方差、GARCH、宏观预期差、Horizon报告、Brier Score、Prompt版本管理、学习状态机、哈希链 | ✅ |
| P2-11 | 动态 Universe 管理框架（universe_manager.py） | ✅ |
| P2-12 Batch A | 7个因子/国际权益ETF，Universe 18→25 | ✅ |
| P2-12 Batch B | 7个跨资产ETF + VXX极性翻转 + Within-class CSMOM，Universe 25→32 | ✅ |
| P2-13 | FactorMAD 四层防御（factor_mad.py + Admin FactorMAD Tab） | ✅ |
| P2-14 | 双变量 MSM（信用利差 BAMLC0A4CBBB） | ✅ |
| P2-15 | 动态交易成本（ATR-based，walk-forward 换手率追踪） | ✅ |
| P2-16 | CircuitBreaker 三级状态机（circuit_breaker.py） | ✅ |
| P2-17 | 新闻三层数据源 + 时效衰减摘要（news_fetcher.py） | ✅ |
| P2-18 | TradingCycleOrchestrator 正式化（orchestrator.py） | ✅ |

### 17.2 里程碑状态

| 里程碑 | 解锁条件 | 状态 |
|--------|---------|------|
| M0 | 基础设施就绪 | ✅ |
| M1 | P0-P1 全部完成 | ✅ |
| M2 | P2-11 + P2-12 Batch A + P2-13 | ✅ Universe=25 ETF |
| **M3** | P2-12 Batch B + P2-14 + Universe ≥ 30 | ✅ **当前位置** Universe=32 ETF |
| M4 | Clean Zone n≥50，Brier Score < 0.25 | ⏳ 等待决策数据积累 |
| M5 | Track B 四层 + RAEiD 重设计 + FactorMAD 稳定运行 | ⏳ |

### 17.3 待完成项

| 项目 | 前置条件 | 规模 |
|------|---------|------|
| Defer-1 置信度校准（calibration curve） | n≥30 Clean Zone | 中 |
| Defer-2 EpisodicMemory | n≥30 | 中 |
| Defer-3 cvxpy BL 组合优化 | n≥30 | 大 |
| Defer-4 Kelly 仓位建议 | n≥30 | 小 |
| Defer-5 反事实情景测试（adversarial.py） | n≥20 | 中 |
| Defer-6 γ/λ 量化框架 | n≥15 | 中 |
| Defer-7 PSR（Probabilistic Sharpe Ratio） | n_months≥50 | 小 |
| Defer-8 多模型真实辩论（Claude/GPT-4o 红队） | n≥30 + Phase 2 | 大 |
| Defer-9 Track B 四层分层架构 | Defer-1 + n≥30 | 极大 |
| Defer-11 FactorMAD 四分位验证 | n≥30（Universe 已满足≥30） | 小 |
| Defer-12 4变量 MSM | 双变量 MSM 验证结果 | 中 |
| **RAEiD 在线学习重设计** | 独立（不依赖 n） | 极大 |
| PRE-9 Finnhub Key | 申请后配置到 secrets.toml | 行政 |
| PRE-10 GNews Key | 申请后配置到 secrets.toml | 行政 |

---

## 18. 已知局限性与学术诚信披露

### 18.1 数据局限

1. **Clean Zone n 不足**：当前 verified 决策数量不足，所有统计学习结论（Kelly、置信度校准、PSR）均被系统性门控到 Defer 状态，不具统计可信度。

2. **ETF 存续期偏差**：回测起点早于年轻 ETF 上市日期（XLC 2018-06-18、MTUM 2013-04-16），survivorship bias audit 发出警告，回测结论需附带此声明。

3. **yfinance Adj Close 债券票息处理**：AGG/IEF/TIP 节点前向填充误差约 -0.17% 到 -0.38%，已验证在可接受范围内（PRE-8）。

### 18.2 模型局限

4. **GARCH 在极端跳跃事件下失效**：假设 i.i.d. 残差，flash crash 和尾部事件时波动率估计显著滞后。

5. **Ledoit-Wolf 平稳性假设**：制度转换期（如 2020-03）协方差结构剧变，LW 估计偏向历史均值，低估转换期风险。

6. **MSM k=2 简化**：BIC 在 60-120 月数据上几乎总选 k=2，忽略"过渡期"作为独立制度的可能性；单变量模型在 2022-2024 加息周期制度错分率约 15-25%（双变量 P2-14 改善但未量化验证）。

7. **gplearn 样本量不足**：32 ETF × 月度截面训练样本量偏少，符号回归结果稳定性有限，三态报告仅作旁证。

8. **零相关假设**：对角协方差在危机期（相关性趋近 1.0）显著低估组合风险，系统输出 ρ=0.5 上界警告但不能替代真实协方差。

### 18.3 LLM 局限

9. **同源模型偏差**：所有节点使用 Gemini 2.5 Flash，缺乏认知多样性；Defer-8 引入 Claude/GPT-4o 红队但依赖 n≥30 对照组。

10. **confidence_score 未校准**：n<50 之前，LLM 自报的 confidence_score 与实际准确率的统计负相关尚未验证（Defer-1 的前置条件），目前仅作排序标签。

11. **LCS 不是完备性验证**：逻辑一致性检验内部自洽，不验证对外部市场现实的预测准确性，通过 LCS 只代表论证结构合理。

### 18.4 研究伦理声明

本系统为 **NUS MSBA 学术研究项目**，仅用于模拟盘。所有分析结果附带"仅供教育目的，不构成投资建议"声明。实盘交易接口被永久排除在项目边界之外。

---

*本报告根据 2026-04-24 代码状态生成。所有技术细节以实际源代码为准（`engine/` 模块优先）。*
