# Macro Alpha Pro — 改进计划技术规格

> 来源：系统1-4.pdf（5+7篇论文）分析 + 代码审查，2026-04-20
> 视角：资深量化AI工程师 + 学术严谨性双重标准
> 状态：P0×4 → P1×4 → P2×5 → defer×5 → 需重设计×1
> 更新：2026-04-20 补充 S1-S6 共7项（S2-2 Defer+风险标注）
> 代码审查：2026-04-20 全库扫描，已更新P0-1和P0-4描述，确认3项已知bug已修复

---

## 实施前准备项（动手前必须完成）

> 以下7项是阻塞或影响实现质量的前置确认，未完成前不应开始对应的改动项。

---

### PRE-1 数据库迁移机制确认（阻塞 P0-3、P1-3、P1-4、S6）

**风险**：多个改动项需向已存在的表加列。SQLAlchemy 的 `Base.metadata.create_all()` 只创建
不存在的表，**不会向已有表添加列**。若 `memory.py` 只用 `create_all()`，加了 ORM 字段后
本地 `macro_alpha_memory.db` 不会自动更新，查询新列静默返回 None 或直接报错。

**需要确认**：
- `engine/memory.py` 中是否有 `_migrate_db()` 或 `ALTER TABLE` 机制？
- 本地数据库是否有需要保护的真实数据？

**预期处理方案**（根据确认结果选一）：

| 情况 | 方案 |
|------|------|
| 有迁移函数 | 在函数中追加 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...` |
| 无迁移函数，无真实数据 | 删除 `.db` 文件，`create_all()` 重建 |
| 无迁移函数，有真实数据 | 手写迁移脚本，备份后执行 |

**操作**：读 `engine/memory.py` 末尾部分，搜索 `migrate` / `alter` / `create_all`。

---

### PRE-2 P0-4双轨接通的数据来源决策（阻塞 P0-4）

**问题**：`run_quant_assessment()` 在 `daily_batch.py` Step 2 已跑过一次（16个 ticker 批量下载）。
LangGraph agent 在 `trading_desk.py` 被单独触发。接通时若重新调用，是重复的16次网络请求。

**三个方案，需在实现前选定**：

| 方案 | 做法 | 推荐度 |
|------|------|--------|
| A | 在 agent 调用前重新调用 `run_quant_assessment()` | 简单但性能差 |
| **B** | **从 `SignalSnapshot` 表读已存的当日数据** | 推荐——复用 daily batch 结果，as_of 一致性最强 |
| C | 调用方传入缓存的 assessment 对象 | 灵活但需改所有调用点 |

**操作**：确认 `SignalSnapshot` 表（`memory.py:525`）的字段是否包含
`tsmom_raw_return`、`ann_vol`、`atr_14`、`atr_63`、`price_vs_sma_200`、`p_risk_on`、`regime_label`
——即 `to_prompt_context_raw()` 所需的全部字段。若字段完整，方案 B 可直接实现。

---

### PRE-3 build_agent_graph() 的所有调用点（阻塞 P0-4、P1-2）

P0-4 和 P1-2 都需要向 `AgentState` 注入新字段。若遗漏任何一个调用点，
那个入口的新字段静默为空，改动对该路径无效。

**操作**：全库搜索 `build_agent_graph`，列出所有调用位置及其构造的初始 state dict。

---

### PRE-4 ETF完整列表与回测起点（阻塞 P0-2）

P0-2 的 survivorship bias 审计需要：
1. `engine/history.py` 的 `SECTOR_ETF` 完整 ticker 列表
2. `engine/backtest.py` 的 `run_backtest()` 默认起始日期

**操作**：读 `engine/history.py` 确认完整列表，读 `engine/backtest.py` 找 `start_date` 默认值。

---

### PRE-5 n_trials 实际调参历史（阻塞 P0-1，需用户确认）

`EFFECTIVE_N_TRIALS` 的值必须反映真实的开发历史，填错则 DSR 校准无意义。

**需用户回答**：
- 是否做过 TSMOM lookback 期的系统性扫描（如 6/9/12 个月）？
- 是否测试过不同的 vol target（如 8%/10%/12%）？
- 是否测试过不同的 skip 月数？
- 若都没有系统扫描，只是手动改过几次参数，保守取 n=6。

---

### PRE-6 Clean Zone 当前 n 值（影响 S5 初始状态、P2-1 实际效果）

**需用户确认**：当前数据库中有多少条 `verified=True` 且 `lcs_passed=True` 的 DecisionLog 记录？

此值决定：
- `get_learning_stage()` 的初始状态（S5）
- P2-1 时间衰减检索的实际意义（n=0 时检索结果为空，衰减无意义）
- 所有 defer 项的实际距离

**操作**：可用以下查询直接获取：
```python
from engine.memory import SessionFactory, DecisionLog
with SessionFactory() as s:
    n = s.query(DecisionLog).filter(
        DecisionLog.verified == True,
        DecisionLog.lcs_passed == True,
        DecisionLog.superseded == False,
    ).count()
    print(f"Clean Zone n = {n}")
```

---

### PRE-7 现有测试套件（影响所有改动项的回归风险）

改动 `memory.py`（查询逻辑）、`trading_schema.py`（新字段）、`daily_batch.py`（新 patrol 逻辑）
均有回归风险。改之前需要知道有无测试可跑。

**操作**：检查根目录和 `tests/` 是否存在测试文件。若无，改动后需手动验证关键路径：
- daily_batch 的幂等性
- WatchlistEntry 状态机转换
- TradeRecommendation 序列化/反序列化

---

## 代码审查结论（2026-04-20 全库扫描）

| 检查项 | 结论 |
|--------|------|
| TSMOM时序安全性 | **安全** — `skip_months=1` 使窗口结束在 as_of 前1个月，无前视偏差 |
| P0-A trailing_high止损 | **已修复** — `daily_batch.py` 正确使用 `trailing_high - 2×ATR`，非 cost_basis |
| P0-B triggered状态命名 | **已修复** — 状态机为 `watching→triggered→active`，命名正确 |
| P0-C SimulatedTrade写入 | **已修复** — `_approve()` 正确写 SimulatedTrade 并 upsert SimulatedPosition |
| QuantAssessment与LLM接通 | **断开** — `to_prompt_context()` 从未被注入任何LLM prompt，两套系统完全独立运行 |
| get_historical_context排序 | **纯时序** — `ORDER BY decision_date DESC LIMIT 5`，无衰减权重 |
| SimulatedPosition.track字段 | **不存在** — 干净新增 |
| DecisionLog.chain_hash字段 | **不存在** — 干净新增 |
| WatchlistEntry.risk_conditions_json | **不存在** — 干净新增 |
| ORM表总数 | 17张表，结构清晰，无冲突 |
| 计划新建文件 | 4个均不存在（macro_fetcher、adversarial、universe_audit、portfolio_optimizer） |

---

## 已移除的3项（附理由）

| 原编号 | 名称 | 移除原因 |
|--------|------|---------|
| 原#7 | FOMC文本注入 | 边际价值低，macro_context已部分覆盖 |
| 原#8 | FinArena风险偏好注入 | Cosmetic操作，不改变LLM分析质量 |
| 原#10 | ATLAS分层Prompt | 没有OPRO反馈闭环就不是ATLAS，只是代码重构，无学术支撑 |

---

## P0 — 立即执行

---

### P0-1 DSR 试验次数校准（Walk-Forward 时序已确认安全）

**代码审查结论（2026-04-20）**

`engine/signal.py` 的 TSMOM 窗口以 `end_cutoff = _month_offset(as_of, skip_months=1)` 截断，
即窗口结束在 `as_of` 前约1个月。`as_of` 当日价格不参与信号计算，**无前视偏差**。
`signal.py:160-163` 的 `<= end_cutoff` 包含边界，但该边界本身已在 `as_of` 之前，安全。

> 注意：若 `skip_months` 被改为 0，则会引入前视偏差。当前默认值 1 是安全的，不需要修改信号代码。

**仍需修改：DSR 的 n_trials 校准**

`backtest.py:205` 和 `backtest.py:448-450` 均 hardcode `n_trials=2`。
DSR 公式（López de Prado 2018）对 `n` 极度敏感：`n=2` vs `n=18` 的 DSR 差距可达 0.2 以上。
若开发中进行过任何参数扫描，`n=2` 会严重低估多重检验惩罚。

**修改清单**

```python
# engine/backtest.py 顶部新增：
# 按实际开发历史填写——若做过完整网格搜索（3×2×3=18），取18；
# 若仅手动调参几次，取6；此数字应在注释中说明依据。
EFFECTIVE_N_TRIALS: int = 18   # ← 需要根据实际调参历史确认

# BacktestResult dataclass 新增字段：
@dataclass
class BacktestResult:
    # 现有字段...
    effective_n_trials: int = EFFECTIVE_N_TRIALS
```

```python
# backtest.py:448-450 调用处修改：
m_tsmom  = _compute_metrics(..., n_trials=EFFECTIVE_N_TRIALS)
m_regime = _compute_metrics(..., n_trials=EFFECTIVE_N_TRIALS)
```

UI 在 DSR 指标旁附说明：`f"基于 {n} 次有效假设试验（Harvey, Liu & Zhu 2016）"`

**学术意义**：未校正多重检验的 Sharpe 在因子研究中毫无意义。`EFFECTIVE_N_TRIALS` 是一个
必须显式声明的研究设计参数，不应隐藏在代码中。

---

### P0-2 Survivorship Bias 审计（新增）

**问题**

若18个sector ETF中有任何一个的成立日期晚于回测起点，就存在survivorship bias——
那个时间段的信号用了"未来才存在"的资产。

**修改清单**

```python
# 新建 engine/universe_audit.py 或在 backtest.py 顶部添加：
INCEPTION_DATES = {
    "XLK": "1998-12-22",
    "XLF": "1998-12-22",
    "XLE": "1998-12-22",
    # ... 其余15个
}

def check_survivorship_bias(backtest_start: str) -> list[str]:
    """返回成立日期晚于backtest_start的ticker列表"""
    late_tickers = []
    for ticker, inception in INCEPTION_DATES.items():
        if inception > backtest_start:
            late_tickers.append(f"{ticker} (inception {inception})")
    return late_tickers
```

在 `run_backtest()` 入口调用，将结果写入 `BacktestResult.warnings`。

---

### P0-3 归因实验设计（新增）

**这是整个双轨系统最核心的研究问题：LLM叠加层是否产生可归因的增量价值。**

**当前问题**

`TradeRecommendation` 记录了 `quant_baseline_weight` 和 `llm_adjustment_pct`，
但 `SimulatedPosition` 只记录了最终的 `actual_weight`。
经过人工审批修改后，无法还原"纯量化组合"的假设路径，LLM贡献度量失效。

**修改方案**

在 `SimulatedPosition` 表中新增 `track` 字段，并行维护两条轨道：

```python
# engine/memory.py SimulatedPosition新增字段：
track = Column(String(20), default="main")
# "main"   = quant_baseline + llm_adjustment（实际执行路径）
# "quant"  = 纯quant_baseline_weight，不含LLM调整，不经人工修改

# 每次创建SimulatedPosition时，同时写两条记录：
# 1. track="main"  → actual_weight（含LLM调整 + 人工审批修改）
# 2. track="quant" → quant_baseline_weight（纯量化，不修改）
```

**归因计算**（在 `verify_pending_decisions()` 中触发）：

```python
# engine/memory.py verify_pending_decisions() 新增：
def compute_llm_alpha(decision_log_id: int) -> float:
    """
    LLM边际贡献 = main组合收益率 - quant组合收益率
    基于同一时间段的actual_return对比
    """
    main_pos  = session.query(SimulatedPosition).filter_by(
        decision_log_id=decision_log_id, track="main").first()
    quant_pos = session.query(SimulatedPosition).filter_by(
        decision_log_id=decision_log_id, track="quant").first()
    if main_pos and quant_pos:
        return main_pos.period_return - quant_pos.period_return
    return None
# → 写入 DecisionLog.llm_weight_alpha（该字段已在memory已有记录中规划）
```

**学术意义**：没有对照组，Clean Zone的任何胜率数字都无法排除"纯量化本来就能做到"的零假设。这是双轨系统存在意义的实证基础。

---

### P0-4 双轨接通：QuantAssessment → LLM Prompt

**代码审查发现（2026-04-20）：两套系统完全断开**

`engine/agent.py` 的 LLM 分析使用**旧系统** `engine/quant.py` 的 `compute_quant_metrics()`
（Lasso-based，输出 `mom_1m/3m/6m`、`a_ret`、`a_vol`、`d_var`、`p_noise`、`val_r2`）。

`QuantAssessment.to_prompt_context()`（`trading_schema.py:222`，包含 TSMOM 信号、CSMOM rank、
composite_score、regime label、ATR、SMA200 等新系统信号）**从未被注入任何 LLM prompt**。

Trading Desk 用新系统做决策，Research Agent 用旧系统做分析，两者运行在完全独立的调用链上。

**此项任务的真实目标：接通，而非"防污染"**

旧系统注入的字段（`a_ret`、`a_vol`、`d_var`、`mom_1m/3m/6m`）均为原始数值，本身不构成污染。
真正的问题是新系统的 TSMOM/CSMOM/regime 信号从未让 LLM 看到，
导致 LLM 的宏观判断与量化轨完全脱节。

**字段注入规则（接通时遵守）**

| 字段类型 | 示例 | 是否注入 | 原因 |
|---------|------|---------|------|
| 旧系统原始数值 | `a_ret`、`a_vol`、`VaR`、`mom_3m` | ✅ 保留 | 是原始信息，不是结论 |
| 新系统原始信号 | `tsmom_raw_return`、`ann_vol`、`atr_14`、`price_vs_sma_200` | ✅ 新增注入 | 原始数值，无方向性 |
| 新系统方向性结论 | `tsmom_signal=+1`、`gate_status=open` | ❌ 屏蔽 | 量化系统已给出结论，LLM看到会被引导 |
| 新系统综合评分 | `composite_score=75` | ❌ 屏蔽 | 已聚合的判断，不是原始信号 |

**修改清单**

1. 在 `trading_schema.py` 的 `QuantAssessment` 新增 `to_prompt_context_raw()` 方法，
   只输出原始数值，不含 `tsmom_signal`、`gate_status`、`composite_score`：

```python
# trading_schema.py QuantAssessment 新增方法：
def to_prompt_context_raw(self) -> str:
    """仅输出原始数值，不含方向性结论，用于LLM prompt注入。"""
    return (
        f"[新量化信号] {self.sector} ({self.ticker}) | {self.as_of_date}\n"
        f"  TSMOM原始收益: {self.tsmom_raw_return:.1%} | "
        f"截面排名: {self.csmom_rank}/{18} | 年化波动率: {self.ann_vol:.1%}\n"
        f"  ATR(21): {self.atr_14:.2f} | ATR(63): {self.atr_63:.2f} | "
        f"vs SMA200: {self.price_vs_sma_200:+.1%}\n"
        f"  制度概率 p(risk-on)={self.p_risk_on:.2f} | 制度标签: {self.regime_label}\n"
        f"  注：以上为量化系统的原始计算结果，非方向性结论，请独立判断。"
    )
```

2. 在 `engine/agent.py` 的 `AgentState` 新增 `quant_assessment_context: str` 字段

3. 在 `red_team_node` 的 `audit_prompt` 中，在现有旧系统指标块之后追加：

```python
quant_new = state.get("quant_assessment_context", "")
quant_new_block = f"\n{quant_new}\n" if quant_new else ""
# 加入 audit_prompt
```

4. 调用方（`trading_desk.py` 或 `orchestrator.py`）在构建 `AgentState` 时，
   从 `run_quant_assessment()` 的结果中调用 `assessment.to_prompt_context_raw()` 填充该字段

---

## P1 — 本周完成

---

### P1-1 Regime-Conditional 指标完整性

**当前状态**

`BacktestMetrics` 已有 `sharpe_risk_on/off` 字段，需验证 `_compute_metrics()` 中是否真正填充。

**补充字段**

```python
@dataclass
class BacktestMetrics:
    # 现有字段（不变）...
    sharpe_risk_on:  float | None
    sharpe_risk_off: float | None
    # 新增：
    drawdown_risk_on:   float | None   # risk-on期间最大回撤
    drawdown_risk_off:  float | None   # risk-off期间最大回撤
    hit_rate_risk_on:   float | None   # risk-on月胜率
    hit_rate_risk_off:  float | None   # risk-off月胜率
    avg_holding_months: float | None   # 平均持仓月数（换手率代理指标）
```

**学术意义**：只报整体Sharpe会掩盖策略在risk-off的崩溃。双轨系统核心研究问题是制度条件化是否增加价值，没有regime拆分就无法回答。

---

### P1-2 持仓状态注入（FinPos）

**当前问题**

`red_team_node` 和 `technical_audit_node` 的 prompt 无任何持仓信息，
LLM隐含假设"全现金起点"，在已有大量持仓时产生系统性偏差（特别是重复推荐已持有标的）。

**实现方案**

```python
# engine/agent.py AgentState新增字段：
class AgentState(TypedDict):
    # 现有字段...
    position_context: str   # 自然语言持仓状态

# 新增helper函数（在trading_desk.py或orchestrator.py调用）：
def build_position_context(sector: str, as_of: datetime.date) -> str:
    with SessionFactory() as s:
        latest = s.query(SimulatedPosition.snapshot_date)\
                   .order_by(SimulatedPosition.snapshot_date.desc()).scalar()
        pos = s.query(SimulatedPosition).filter_by(
            sector=sector, snapshot_date=latest).first()
    if pos is None:
        return f"当前 {sector} 无持仓。"
    days_held = (as_of - pos.snapshot_date).days
    # unrealized PnL需要当日close，可从QuantAssessment获取
    return (
        f"当前持仓：{sector}({pos.ticker})，方向={pos.direction or '多头'}，"
        f"持仓权重={pos.actual_weight:.1%}，"
        f"持有约{days_held}天，"
        f"建仓价≈{pos.entry_price:.2f}，"
        f"制度标签={pos.regime_label or '未知'}"
    )
```

**在 `red_team_node` prompt 注入**：

```python
position_line = state.get("position_context", "无持仓数据")
# 加在audit_prompt的宏观背景段之后：
f"当前持仓状态：{position_line}"
```

---

### P1-3 risk_conditions Schema（TiMi）

**当前状态**

`daily_batch.py` 的 `_patrol_positions` 有三种硬编码风控（trailing stop、tsmom flip、regime compression），
无法per-position定制。`TradeRecommendation` 无 `risk_conditions` 字段。

**新增数据结构**

```python
# engine/trading_schema.py 新增：

from typing import Literal

RiskTrigger = Literal[
    "price_below_stop",     # 价格跌破指定止损价
    "drawdown_exceeds",     # 从建仓价亏损超过X%（如-0.05）
    "vol_spike",            # 年化波动率突破阈值
    "regime_shift",         # 制度切换（risk-on→risk-off）
    "tsmom_flipped",        # TSMOM信号翻转（与InvalidationCondition对齐）
]

RiskAction = Literal[
    "reduce_half",          # 减仓50%
    "exit_full",            # 清仓
    "flag_review",          # 标记人工审核，不自动操作
    "reduce_to_cap",        # 减仓至当前制度的WEIGHT_LIMITS上限
]

@dataclass
class RiskCondition:
    trigger:     RiskTrigger
    action:      RiskAction
    threshold:   Optional[float] = None  # 触发阈值（drawdown: -0.05；vol_spike: 0.40）
    description: Optional[str]   = None  # UI展示用

# TradeRecommendation 新增字段（在 invalidation_conditions 之后）：
risk_conditions: list[RiskCondition] = field(default_factory=list)
```

**to_watchlist_dict() 同步更新**：

```python
"risk_conditions_json": json.dumps([dataclasses.asdict(c) for c in self.risk_conditions]),
```

**WatchlistEntry ORM 新增列**：

```python
# engine/memory.py WatchlistEntry：
risk_conditions_json = Column(Text, nullable=True)
```

**`_patrol_positions` 中添加评估逻辑**：

```python
# daily_batch.py _patrol_positions，在现有三步之后：
risk_json = json.loads(entry.risk_conditions_json or "[]")
for rc_raw in risk_json:
    rc = RiskCondition(**rc_raw)
    triggered = False
    if rc.trigger == "drawdown_exceeds" and rc.threshold and pos.entry_price and close:
        drawdown = (close - pos.entry_price) / pos.entry_price
        triggered = drawdown < rc.threshold
    elif rc.trigger == "vol_spike" and rc.threshold:
        triggered = pos.ann_vol and pos.ann_vol > rc.threshold
    # ... 其他trigger类型
    if triggered:
        target_w = 0.0 if rc.action == "exit_full" \
            else pos.actual_weight * 0.5 if rc.action == "reduce_half" \
            else WEIGHT_LIMITS[pos.position_rank][regime_label] if rc.action == "reduce_to_cap" \
            else pos.actual_weight  # flag_review: 不改权重
        _add_risk_approval(session, pos, t_day, close,
            reason=f"risk_condition: {rc.trigger}",
            priority="normal",
            suggested_weight=target_w,
            action_type=rc.action)
```

---

### P1-4 连续权重输出

**当前状态**

`daily_batch.py:527` 的 `_add_risk_approval` hardcode `suggested_weight=0.0`，
所有风控触发都变成"建议清仓"，无法表达部分减仓。

**修改方案**

```python
# daily_batch.py _add_risk_approval 签名修改：
def _add_risk_approval(
    session,
    pos: SimulatedPosition,
    t_day: datetime.date,
    close: float,
    reason: str,
    priority: str,
    suggested_weight: float = 0.0,       # 新增：默认清仓保持向后兼容
    action_type: str = "exit_full",      # 新增
) -> None:
    ...
    session.add(PendingApproval(
        ...
        suggested_weight=suggested_weight,
        action_type=action_type,         # PendingApproval表需同步添加此字段
    ))
```

**调用处修改**（regime compression触发时）：

```python
# _patrol_positions Step 5.3：
cap = WEIGHT_LIMITS.get(rank, {}).get(regime_label, 0.15)
_add_risk_approval(
    session, pos, t_day, close,
    reason=f"Regime {regime_label}: weight {pos.actual_weight:.1%} > cap {cap:.1%}",
    priority="normal",
    suggested_weight=cap,                # 减至上限，不是清仓
    action_type="reduce_to_cap",
)
```

**依赖**：P1-3（risk_conditions）先完成，action_type字段语义对齐。

---

## P2 — 下周完成

---

### P2-1 时间衰减记忆检索

**当前状态**

`get_historical_context()` 的检索策略高概率是简单的最近N条记录，
没有时间衰减。旧的regime pattern（如2018年的risk-off记录）与2026年的权重相同。

**实现方案**

λ = ln(2)/90（3个月半衰期）。3个月前权重=50%，6个月前=25%，12个月前=6%。

```python
# engine/memory.py get_historical_context() 重写检索排序：
from sqlalchemy import func as sqlfunc

DECAY_LAMBDA = math.log(2) / 90   # 半衰期90天

# SQLite不直接支持exp()，用Python侧计算：
records = (
    session.query(DecisionLog)
    .filter(DecisionLog.sector_name == sector_name)
    .filter(DecisionLog.macro_regime == macro_regime)
    .filter(DecisionLog.verified == True)
    .filter(DecisionLog.lcs_passed == True)
    .order_by(DecisionLog.created_at.desc())
    .limit(20)   # 取最近20条，Python侧加权后取top 5
    .all()
)

def _decay_weight(record) -> float:
    days_old = (datetime.datetime.utcnow() - record.created_at).days
    quality  = record.accuracy_score or 0.5
    return math.exp(-DECAY_LAMBDA * days_old) * quality

top5 = sorted(records, key=_decay_weight, reverse=True)[:5]
```

**与TradingGPT情节记忆的关系**：P2-1是对现有 `DecisionLog` 检索的改进；
`#defer-2` 的 `EpisodicMemory` 是新增结构化表，两者不重复——P2-1先做，
等n≥30后再评估是否需要独立表。

---

## Defer — 等 Clean Zone n≥30

---

### Defer-1 FinDebate 置信度校准

**等待条件**：n≥30条 `verified=True` + `barrier_hit NOT NULL` 的 DecisionLog。

**为何推迟**

当前 `research_node` 的 `confidence_score = 100 - penalty_p - penalty_s`（rule-based）。
要求LLM输出数字置信度然后正则解析是brittle的——LLM不产生校准过的概率。
在没有calibration曲线（Platt scaling）验证前，Dispersion Index只是一个数字，没有信息内容。

**n≥30后的实现方案**

```python
# 每个LLM节点（red_team、auditor）的prompt末尾添加：
"最后一行必须输出：置信度评分: [0到100的整数，仅数字]"

# translator_node中汇总：
import re

def _parse_confidence(text: str) -> int:
    m = re.search(r"置信度评分:\s*(\d+)", text)
    return int(m.group(1)) if m else 50

confidences = [
    state["quant_results"]["confidence_score"],       # rule-based
    _parse_confidence(state["red_team_critique"]),    # LLM
    _parse_confidence(state["technical_report"]),     # LLM
]
dispersion = float(np.std(confidences)) / 50.0
C_final = int(np.mean(confidences) * (1 - dispersion))
# → 写入DecisionLog.confidence_score
```

**验证要求**：实施前，用已有30+条验证数据画calibration curve，
确认Dispersion Index与 `accuracy_score` 有统计负相关（p<0.05）再部署。

---

### Defer-2 TradingGPT 情节记忆

**等待条件**：n≥30 + P2-1时间衰减检索已验证效果。

**新增表结构**

```python
# engine/memory.py 新增：
class EpisodicMemory(Base):
    __tablename__ = "episodic_memories"

    id               = Column(Integer, primary_key=True)
    decision_log_id  = Column(Integer, nullable=False)   # FK → DecisionLog.id
    sector           = Column(String(100))
    regime_label     = Column(String(50))               # risk-on/risk-off/transition
    decision_summary = Column(Text)                     # 压缩摘要（约100字）
    outcome_label    = Column(String(20))               # "correct"/"wrong"/"neutral"
    confidence_at_decision = Column(Integer)
    accuracy_score   = Column(Float)
    barrier_hit      = Column(String(8))                # tp/sl/time
    created_at       = Column(DateTime)
    decay_weight     = Column(Float, default=1.0)       # 每日batch更新
```

**每日衰减更新**（DailyBatchJob Step 0前置）：

```python
LAMBDA = math.log(2) / 90
for mem in session.query(EpisodicMemory).all():
    days_old = (datetime.date.today() - mem.created_at.date()).days
    mem.decay_weight = math.exp(-LAMBDA * days_old)
```

---

### Defer-3 cvxpy 组合优化器

**等待条件**：n≥30 Clean Zone记录，验证LLM opinion质量后再引入优化。

**规划接口**

```python
# engine/portfolio_optimizer.py（待建）
import cvxpy as cp

def optimize_weights(
    opinion_vector: dict[str, float],   # sector → LLM强度 (-1到+1)
    quant_baseline:  dict[str, float],  # sector → vol-parity权重
    regime_label:    str,
    position_rank:   str = "satellite",
) -> dict[str, float]:
    """
    minimize: ||w - quant_baseline||_2^2 - λ × opinion_vector^T w
    subject to:
        Σw_i = 1
        w_i ≤ WEIGHT_LIMITS[position_rank][regime_label]
        w_i ≥ 0
        ||w - quant_baseline||_∞ ≤ 0.10   # LLM最大单品调整±10%
    """
    n = len(opinion_vector)
    w = cp.Variable(n)
    ...
```

---

## 需重设计 — RAEiD 在线学习

**当前问题**

LangGraph图结构：`researcher → red_team → auditor → translator`

只有 `red_team` 节点有明确的方向性判断（intercept/pass = -1/+1）。
`researcher` 只输出quant指标，`auditor` 输出技术分析，均无独立方向vote。
因此"三个agent的Bayesian权重"在当前架构下实际只有一个agent有效投票。

**先决条件**：重构LangGraph图，让每个节点都输出 `{direction: int, confidence: int}`

```python
# 目标架构（需重构）：
class AgentState(TypedDict):
    # 每个节点的独立投票（重构后新增）：
    researcher_vote:  dict  # {"direction": +1/-1/0, "confidence": 0-100}
    red_team_vote:    dict
    auditor_vote:     dict

# AgentVoteLog表（待建）：
class AgentVoteLog(Base):
    __tablename__ = "agent_vote_logs"
    id              = Column(Integer, primary_key=True)
    decision_log_id = Column(Integer)
    agent_name      = Column(String(50))    # "researcher"/"red_team"/"auditor"
    vote_direction  = Column(Integer)       # +1/-1/0
    vote_confidence = Column(Integer)
    regime_label    = Column(String(50))
    accuracy_score  = Column(Float, nullable=True)  # 验证后回填
    created_at      = Column(DateTime)
```

**Bayesian权重更新**（验证后触发）：

```python
LEARNING_RATE = 0.1

def update_agent_weights(sector: str, regime: str, decision_log_id: int):
    votes = session.query(AgentVoteLog).filter_by(
        decision_log_id=decision_log_id).all()
    for vote in votes:
        if vote.accuracy_score is not None:
            w_key = f"{vote.agent_name}_{sector}_{regime}"
            current_w = get_agent_weight(w_key)  # 从AgentWeight表读
            if vote.accuracy_score > 0.5 and vote.vote_direction != 0:
                new_w = current_w * (1 + LEARNING_RATE)
            elif vote.accuracy_score < 0.3:
                new_w = current_w * (1 - LEARNING_RATE)
            update_agent_weight(w_key, new_w / sum_all_weights)  # 归一化
```

**实施顺序**：先重构图结构 → 再建AgentVoteLog表 → 再实现Bayesian更新。

---

## 补充项 S1 — 回测统计推断严谨性

---

### S1-1 DSR 的 effective_n_trials 校准（补充 P0-1）

**当前问题**

`backtest.py:205` hardcode `n_trials=2`。DSR 公式中试验次数 $n$ 的取值直接影响显著性判断。
若开发过程中做过任何参数扫描，$n=2$ 会严重低估多重检验惩罚。

**修改方案**

```python
# engine/backtest.py 顶部新增：
HYPOTHESIS_TRIALS = {
    "tsmom_lookback":  [6, 9, 12],         # 曾测试过的回看期（月）
    "tsmom_skip":      [1, 2],             # 曾测试过的跳过月数
    "vol_target":      [0.08, 0.10, 0.12], # 曾测试过的目标波动率
}
# 若全部网格搜索：n = 3×2×3 = 18
# 若仅手动调参：保守取 n=6～10，记录于 EFFECTIVE_N_TRIALS 常量

EFFECTIVE_N_TRIALS: int = 18  # 按实际修改此值，注释说明依据

# BacktestResult 新增字段：
@dataclass
class BacktestResult:
    # 现有字段...
    effective_n_trials: int = EFFECTIVE_N_TRIALS  # UI 展示时说明来源
```

**UI 展示**：在 `pages/backtest.py` 的 DSR 指标旁追加说明：
`"基于 {n} 次有效假设试验的 Deflated Sharpe Ratio（Harvey, Liu & Zhu 2016）"`

**学术意义**：未校正多重检验的 Sharpe Ratio 在因子动物园中毫无意义。
显式声明 $n$ 是对 Harvey et al. (2016) 学术共识的工程化落实，也是学术展示的基本要求。

---

### S1-2 Probabilistic Sharpe Ratio（Defer-S1）

PSR 与 DSR 互补：DSR 回答"多次试验后该 Sharpe 是否显著"，
PSR 回答"该 Sharpe 比给定阈值（如 SR*=0.5）更优的概率"。

```python
# engine/backtest.py _compute_metrics() 中可选计算：
def _psr(sr_hat: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    """
    López de Prado (2018) §14.4 公式。
    P(SR > sr_star) = Φ( (sr_hat - sr_star) × sqrt(n-1) /
                         sqrt(1 - skew×sr_hat + ((kurt-1)/4)×sr_hat²) )
    """
    import scipy.stats as stats
    denom = math.sqrt(1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2)
    z = (sr_hat - sr_star) * math.sqrt(n - 1) / (denom + 1e-9)
    return float(stats.norm.cdf(z))

# BacktestMetrics 可选字段：
psr_vs_zero:      float | None = None   # P(SR > 0)
psr_vs_half:      float | None = None   # P(SR > 0.5)
```

**激活条件**：n_months ≥ 50（约4年月频数据）后激活，否则 PSR 置信区间过宽无意义。

---

## 补充项 S2 — 宏观预期差数据源

---

### S2-1 经济惊喜指数注入（P2）

**核心原理**

宏观交易的核心是"预期差"而非"绝对水平"。
当前系统注入 LLM 的是 `CPI=3.2%`，正确的应该是 `CPI=3.2%（预期3.0%，超预期+0.2%）`。
没有预期锚的宏观分析在方法论上是残缺的。

**数据方案**（按可获取性排序）

| 数据源 | 内容 | 可用性 |
|--------|------|--------|
| FRED `CESIUSD` 镜像 | Citi Economic Surprise Index（部分） | 免费，有延迟 |
| `econdb.com` API | 多国宏观一致预期 vs 实际值 | 免费层有限额 |
| 手工维护 CSV | FOMC/CPI/NFP 预期 vs 实际 | 低维护成本，高可控性 |

**注入方式**

```python
# engine/macro_fetcher.py 新增：
def get_economic_surprise(indicator: str, release_date: datetime.date) -> dict:
    """
    返回 {actual: float, consensus: float, surprise: float, surprise_direction: str}
    数据源优先级：FRED → econdb → 本地CSV → None
    """
    ...

# red_team_node prompt 中替换：
# 旧：f"最新 CPI: {cpi_value:.1%}"
# 新：f"最新 CPI: {actual:.1%}（市场预期 {consensus:.1%}，{'超预期' if surprise>0 else '低于预期'} {abs(surprise):.1%}）"
```

**风险标注**：免费数据源的一致预期质量参差不齐，需要在 UI 中标注数据来源和可信度。

---

### S2-2 资金流代理变量（Defer-S2，附风险标注）

**可行性风险**：`yf.Ticker(ticker).fund_flow` 在 yfinance 中对大多数 ETF 返回 `None`，
可靠性未经验证。OBV 本质上是动量的另一种形式，与 TSMOM 高度相关，增量 Alpha 可疑。

**条件**：实施前先验证目标 universe 中有多少 ETF 可以稳定获取资金流数据，
若覆盖率 < 50% 则不值得引入，改用成交量相对强弱（Volume Ratio）作代理。

```python
# 验证脚本（先跑，再决定是否实现）：
import yfinance as yf
from engine.history import SECTOR_ETF

coverage = {}
for sector, ticker in SECTOR_ETF.items():
    try:
        t = yf.Ticker(ticker)
        flow = getattr(t, 'fund_flow', None)
        coverage[ticker] = flow is not None and not (hasattr(flow, 'empty') and flow.empty)
    except Exception:
        coverage[ticker] = False
print(f"Fund flow coverage: {sum(coverage.values())}/{len(coverage)}")
```

---

## 补充项 S3 — Horizon 分层绩效报告（P2）

**当前问题**

`DecisionLog.horizon` 字段已有（如"季度约3个月"/"半年约6个月"），
但 Clean Zone 绩效看板按全部决策聚合，不区分 horizon。
如果系统在短期决策上显著优于长期，这是 LLM 预测能力边界的重要元信息。

**实现方案**

在 `pages/clean_zone.py`（或 `pages/admin.py`）的绩效 Tab 中增加分层表格：

```python
# 查询（memory.py 新增辅助函数）：
def get_horizon_stratified_metrics() -> list[dict]:
    rows = (
        session.query(
            DecisionLog.horizon,
            func.count(DecisionLog.id).label("n"),
            func.avg(DecisionLog.accuracy_score).label("avg_accuracy"),
            func.avg(DecisionLog.barrier_days).label("avg_holding_days"),
            func.avg(DecisionLog.confidence_score).label("avg_confidence"),
        )
        .filter(DecisionLog.verified == True)
        .filter(DecisionLog.superseded == False)
        .group_by(DecisionLog.horizon)
        .all()
    )
    return [r._asdict() for r in rows]
```

**UI 表格（Streamlit）**：

```
Horizon       | n  | Win Rate | Avg Accuracy | Avg Holding | Avg Confidence
季度 (~3M)    | 8  | 62%      | 0.68         | 78d         | 72
半年 (~6M)    | 7  | 43%      | 0.51         | 142d        | 68
中期 (~1M)    | 12 | 58%      | 0.61         | 24d         | 65
```

**低实现成本**：`horizon` 字段已有，这是纯 GROUP BY 查询 + 表格渲染，约半天工作量。

---

## 补充项 S4 — 反事实情景测试（Defer-S1）

**与现有 LCS 的关系**

LCS 镜像测试（`lcs_mirror_passed`）= 自动化、用于决策质量门控的反事实测试。
S4-1 = 人工触发、用于深度归因的反事实测试。两者互补，不重复。

**实现方案**

```python
# engine/adversarial.py（新文件）
def run_counterfactual_test(
    decision_log_id: int,
    alternative_scenario: str,  # 如"Fed 加息 50bps 而非 25bps"
) -> dict:
    """
    基于历史决策，修改关键宏观变量，重新触发 Debate Engine。
    评估推理是否真正依赖宏观逻辑，而非输入文本的模式匹配。
    """
    original = get_decision(decision_log_id)
    altered_context = f"{original.macro_context}\n[反事实修改] {alternative_scenario}"
    # 重新调用 agent.py 的完整 pipeline（仅 researcher+red_team，不写 DB）
    new_result = run_debate_engine_dry(altered_context, sector=original.sector_name)
    return {
        "original_direction":       original.direction,
        "counterfactual_direction": new_result["direction"],
        "decision_flipped":         original.direction != new_result["direction"],
        "confidence_delta":         new_result["confidence"] - original.confidence_score,
        "scenario":                 alternative_scenario,
    }
```

**激活条件**：n ≥ 20 条 verified 决策后激活，在 Admin 面板的 Post-Mortem Tab 提供入口。

---

## 补充项 S5 — 学习阶段状态机（P2）

**当前问题**

改进计划中有多个"n≥30"门控散落各处（Defer-1、Defer-2、需重设计），
缺乏统一的"系统当前处于哪个学习阶段"的显式声明，导致：
1. 无法在 UI 中向用户展示"还差多少步解锁下一阶段"
2. 多个组件各自判断 n，可能不一致

**简化实现**（计算函数而非 ORM 表，减少维护负担）

```python
# engine/memory.py 新增：
from dataclasses import dataclass

@dataclass
class LearningStageInfo:
    stage:         str    # "cold_start" / "memory_active" / "parameter_adaptive" / "structural_adaptive"
    n_verified:    int
    n_required_next: int | None
    unlocked_features: list[str]
    next_features:     list[str]

def get_learning_stage() -> LearningStageInfo:
    """计算当前学习阶段，基于现有表实时推导，不持久化状态。"""
    with SessionFactory() as s:
        n = s.query(func.count(DecisionLog.id))\
             .filter(DecisionLog.verified == True)\
             .filter(DecisionLog.lcs_passed == True)\
             .filter(DecisionLog.superseded == False)\
             .scalar() or 0

        regimes_covered = s.query(func.count(func.distinct(DecisionLog.macro_regime)))\
                           .filter(DecisionLog.verified == True)\
                           .scalar() or 0

    if n < 10:
        return LearningStageInfo(
            stage="cold_start", n_verified=n, n_required_next=10,
            unlocked_features=["规则决策", "Triple-Barrier验证"],
            next_features=["时间衰减记忆检索（P2-1）"],
        )
    elif n < 30 or regimes_covered < 2:
        return LearningStageInfo(
            stage="memory_active", n_verified=n,
            n_required_next=30 if n < 30 else None,
            unlocked_features=["时间衰减记忆检索", "Horizon分层报告"],
            next_features=["FinDebate置信度校准", "情节记忆（Defer-1,2）"],
        )
    elif n < 50:
        return LearningStageInfo(
            stage="parameter_adaptive", n_verified=n, n_required_next=50,
            unlocked_features=["FinDebate Dispersion Index", "情节记忆", "cvxpy优化"],
            next_features=["Agent权重Bayesian更新（需重设计）", "PSR统计量"],
        )
    else:
        return LearningStageInfo(
            stage="structural_adaptive", n_verified=n, n_required_next=None,
            unlocked_features=["全部功能解锁"],
            next_features=[],
        )
```

**UI 展示**：在 Admin 面板 System Tab 顶部显示学习阶段进度条和已解锁/待解锁功能列表。

---

## 补充项 S6 — 实验日志不可篡改性（P2）

**实现**

```python
# engine/memory.py 新增函数：
import hashlib, json

def compute_decision_chain_hash(prev_hash: str, decision_data: dict) -> str:
    """
    SHA-256 哈希链。每条 DecisionLog 记录时调用，prev_hash 来自上一条记录。
    首条记录的 prev_hash = "GENESIS"。
    """
    payload = prev_hash + json.dumps(decision_data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

def verify_chain_integrity() -> tuple[bool, int | None]:
    """
    遍历所有 DecisionLog（按 id 排序），重新计算哈希链。
    返回 (is_intact, first_broken_id)。
    """
    records = session.query(DecisionLog).order_by(DecisionLog.id).all()
    prev_hash = "GENESIS"
    for rec in records:
        expected = compute_decision_chain_hash(prev_hash, _decision_to_dict(rec))
        if rec.chain_hash and rec.chain_hash != expected:
            return False, rec.id
        prev_hash = rec.chain_hash or expected
    return True, None
```

**DecisionLog 新增字段**：

```python
chain_hash = Column(String(64), nullable=True)  # SHA-256，64字符
```

**诚实声明**（要在 UI 中展示）：
> "哈希链可检测事后单条记录的篡改。若整条链被重新计算替换，则无法检测。
>  对个人 paper trading 系统，这是可信度信号而非安全机制。"

**Admin 面板 System Tab** 增加"验证链完整性"按钮，调用 `verify_chain_integrity()`，
绿色 ✅ 或红色 ❌ 显示第一条损坏记录的 id。

---

1. **两轨独立验证**：`quant_baseline_weight` 和 `llm_adjustment_pct` 必须始终分开记录，
   任何混合指标都需要同时保留分解项，否则归因不可能。

2. **前视偏差零容忍**：每个信号计算点都要显式确认数据截断日期。P0-1是实施前提。

3. **置信度≠准确率**：LLM的confidence_score只有n≥50验证后才能被calibration曲线检验。
   在此之前，它只是一个标签，不是概率估计。

4. **λ选择依据**：λ=ln(2)/90（3月半衰期）适合宏观制度信息。
   技术信号（如TSMOM的12月窗口）可以考虑更短半衰期（λ=ln(2)/60，2月）。
   未来可以per-information-type差异化。

5. **自动学习的统计门槛**：月频12个样本/年。任何自动参数更新在n<30时等价于过拟合。
   （与spec_simulated_execution.md §11.6的"明确不做"保持一致）
