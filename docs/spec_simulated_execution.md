## 十一、模拟执行层技术规格（待实现，2026-04-17）

### 11.1 定位与边界

**定位**：有状态的模拟执行层（Simulated Execution Layer），是统计信号流水线与实盘对接之间的必要中间层。

**与回测的本质区别**：

| 维度 | `engine/backtest.py`（历史回测） | 模拟执行层（前向测试） |
|------|-------------------------------|---------------------|
| 时间方向 | 用历史数据重跑 | 实时按信号操作，结果未知 |
| 参数关系 | 参数可能接触过历史数据 | 完全样本外，零参数污染 |
| 仓位状态 | 无状态（每月独立计算） | 有状态，上月持仓影响本月操作 |
| 交易粒度 | 假设完美执行 | 生成具体交易指令，记录换手成本 |
| 路径依赖 | 无 | 有（持仓漂移、再平衡阈值） |
| 实盘对接 | 不可直接对接 | 是实盘接口的自然前体 |

**不尝试做的事（明确边界）**：
- 不做自动参数更新（月频 12 个样本/年，统计上不足以支撑可靠的参数学习）
- 不做日内执行模拟（月度信号不需要日内执行复杂度）
- 不做真实资金对接（当前阶段不建议）

---

### 11.2 数据模型（engine/memory.py 新增三张表）

#### SimulatedPosition — 持仓快照表

```python
class SimulatedPosition(Base):
    """
    每个月末再平衡后的持仓状态快照。
    记录目标权重已落实后的仓位，是下一次再平衡的起点。
    """
    __tablename__ = "simulated_positions"

    id             = Column(Integer, primary_key=True)
    snapshot_date  = Column(Date, nullable=False)    # 再平衡执行日期（月末）
    sector         = Column(String(50), nullable=False)
    ticker         = Column(String(20), nullable=False)
    target_weight  = Column(Float, nullable=False)   # 信号建议权重
    actual_weight  = Column(Float, nullable=True)    # 考虑再平衡阈值后的实际权重
    entry_price    = Column(Float, nullable=True)    # 建仓/调仓时收盘价
    regime_label   = Column(String(20), nullable=True)
    signal_tsmom   = Column(Integer, nullable=True)  # +1 / -1 / 0
    notes          = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("snapshot_date", "sector", name="uq_pos_date_sector"),
    )
```

#### SimulatedTrade — 交易记录表

```python
class SimulatedTrade(Base):
    """
    每次再平衡产生的具体交易指令记录。
    delta = target_weight - current_weight → BUY / SELL
    """
    __tablename__ = "simulated_trades"

    id             = Column(Integer, primary_key=True)
    trade_date     = Column(Date, nullable=False)
    sector         = Column(String(50), nullable=False)
    ticker         = Column(String(20), nullable=False)
    action         = Column(String(10), nullable=False)  # BUY / SELL / HOLD
    weight_before  = Column(Float, nullable=False)
    weight_after   = Column(Float, nullable=False)
    weight_delta   = Column(Float, nullable=False)       # 有符号
    cost_bps       = Column(Float, nullable=True)
    trigger_reason = Column(String(50), nullable=True)
    # signal_flip / rebalance / regime_change / threshold
```

#### SimulatedMonthlyReturn — 月度收益归因表

```python
class SimulatedMonthlyReturn(Base):
    """月度收益的持仓级别归因，用于前向测试 vs 历史回测对比。"""
    __tablename__ = "simulated_monthly_returns"

    id              = Column(Integer, primary_key=True)
    return_month    = Column(Date, nullable=False)
    sector          = Column(String(50), nullable=False)
    weight_held     = Column(Float, nullable=False)
    sector_return   = Column(Float, nullable=True)
    contribution    = Column(Float, nullable=True)   # weight × return
    regime_label    = Column(String(20), nullable=True)
    is_profitable   = Column(Boolean, nullable=True)

    __table_args__ = (
        UniqueConstraint("return_month", "sector", name="uq_ret_month_sector"),
    )
```

---

### 11.3 核心函数（engine/portfolio_tracker.py，新文件）

#### get_current_positions(as_of)
返回最近一次再平衡后的持仓快照 DataFrame。`as_of=None` 时返回最新持仓。

#### generate_rebalance_trades(current_positions, new_weights, rebalance_date, min_trade_size=0.005)
对比当前持仓与目标权重，生成交易指令列表。

核心逻辑：
```python
delta = target_weight - current_weight
if abs(delta) > min_trade_size:
    action = "BUY" if delta > 0 else "SELL"
    trigger_reason = 根据以下优先级判断：
        1. signal_flip   —— TSMOM 信号方向翻转
        2. regime_change —— MSM regime 标签改变
        3. rebalance     —— 权重漂移超过阈值
```

#### execute_rebalance(rebalance_date, dry_run=True)
完整月度再平衡流程：
1. `get_current_positions()` 获取当前持仓
2. `get_signal_dataframe()` 计算本月信号
3. `get_regime_on()` 获取本月制度
4. `construct_portfolio()` 构建目标权重
5. `generate_rebalance_trades()` 生成交易指令
6. 计算换手率和交易成本
7. `dry_run=False` 时写入 DB

Returns: `{trades, new_positions, total_cost_bps, turnover}`

#### record_monthly_return(return_month)
次月初调用，记录上月实际收益归因：
1. 读取对应月份的 `SimulatedPosition` 快照
2. `yfinance` 获取各 ETF 当月收益率
3. 计算逐板块 `contribution = weight × return`
4. 写入 `SimulatedMonthlyReturn`
5. 亏损仓位（`is_profitable=False`）自动标注待 `failure_type` 归因

---

### 11.4 UI 规格（pages/portfolio_monitor.py）

```
┌──────────────────────────────────────────────────────────┐
│  KPI Strip                                                │
│  [月度收益] [累计收益] [最大回撤] [vs基准] [换手率] [成本]  │
└──────────────────────────────────────────────────────────┘

┌─────────────────────┐  ┌──────────────────────────────┐
│  当前持仓权重         │  │  本月待执行交易              │
│  横向条形图           │  │  BUY  XLF  +5.2%            │
│  多头绿 / 空头红      │  │  SELL XLE  -3.1%            │
│  每行：板块|权重|信号  │  │  换手率: 18.3%              │
│                      │  │  预计成本: 12.4 bps          │
│  [执行再平衡] 按钮    │  │  [Dry Run 预览] [确认执行]   │
└─────────────────────┘  └──────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  前向测试 vs 历史回测对比折线图                             │
│  蓝：模拟实盘（首次执行再平衡起）                           │
│  灰虚：历史回测（同参数）                                   │
│  绿：等权基准                                              │
│  说明文字：实盘与回测偏差来源于执行成本和市场微结构差异       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  月度收益归因表                                             │
│  月份 | 总收益 | 最大贡献板块 | 最大拖累板块 | 制度 | 归因   │
│  亏损月份高亮红色，点击展开 failure_type 归因下拉           │
└──────────────────────────────────────────────────────────┘
```

**执行再平衡交互流程**：
1. 点击"Dry Run 预览" → 调用 `execute_rebalance(dry_run=True)` → 展示交易列表 + 成本
2. 用户确认 → 调用 `execute_rebalance(dry_run=False)` → 写入 DB → 刷新持仓图

**亏损月份归因触发**：
- `is_profitable=False` 的板块行自动展示 `failure_type` 下拉（与 待实现-A 打通）
- LLM 辅助预填候选原因（结合 regime 信息和信号方向）
- 确认后写入 `DecisionLog.failure_type + failure_note`

---

### 11.5 再平衡阈值设计

不是每月都需要完整再平衡，引入分层阈值避免高换手：

| 阈值类型 | 默认值 | 触发条件 |
|---------|--------|---------|
| 信号翻转 | 立即触发 | TSMOM 从 +1 变 -1（或反向） |
| 制度变化 | 立即触发 | risk-on ↔ risk-off 切换 |
| 漂移再平衡 | 2% | 实际权重偏离目标 > 2% |
| 微调忽略 | 0.5% | 权重变化 < 0.5%，跳过节省成本 |

---

### 11.6 正负反馈机制（经验归纳，非自动参数更新）

`record_monthly_return()` 执行后触发人工辅助归纳提示：

```
盈利仓位（contribution > 0）：
  → 提示："当前 regime={X}，signal={+1/-1}，sector={Y} 组合盈利
           是否写入 SkillLibrary.boundary_conditions？"

亏损仓位（contribution < 0）：
  → 触发 failure_type 归因下拉（待实现-A）
  → 提示："是否将此失效情形写入 SkillLibrary.known_failures？"（待实现-B）
```

**明确不做**：不根据 P&L 自动调整 `target_vol`、`regime_scale`、LLM prompt 权重等参数。月频 12 个样本/年，自动参数学习在统计上等同于过拟合。

---

### 11.7 模块集成关系

```
engine/signal.py          → get_signal_dataframe()    → 目标权重输入
engine/regime.py          → get_regime_on()           → 制度标签
engine/portfolio.py       → construct_portfolio()     → 目标权重
engine/backtest.py        → run_backtest()            → 历史基准线
engine/memory.py          → 三张新表 + _migrate_db()
engine/portfolio_tracker.py  → 新文件：核心执行逻辑

pages/portfolio_monitor.py   → 新文件：UI 渲染
pages/backtest.py            → 对比图历史数据来源
```

打通关系：
- `failure_type` 归因 ←→ 待实现-A（`DecisionLog.failure_type`）
- `known_failures` 写入 ←→ 待实现-B（`SkillLibrary.known_failures`）

---

### 11.8 实现拆分（约 4~4.5 天工作量）

| Phase | 内容 | 预计工作量 |
|-------|------|----------|
| Phase 1 | `engine/memory.py` 三张新表 + 迁移脚本 | 0.5 天 |
| Phase 2 | `engine/portfolio_tracker.py` 四个核心函数 | 1.5 天 |
| Phase 3 | `pages/portfolio_monitor.py` UI 渲染 | 1.5 天 |
| Phase 4 | 与 待实现-A/B 打通，集成测试 | 1 天 |
