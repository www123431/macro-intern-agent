# Daily Tactical Patrol — 设计规格

> 状态：待实现  
> 决策日期：2026-04-25  
> 优先级：P4 批次首项

---

## 设计背景

当前系统的日频巡逻（`_patrol_positions()`）只处理硬止损和回撤止损。  
本规格描述新增的战术层（`_patrol_daily_tactical()`），目标是在月度再平衡之间实现动态调仓，使系统更贴近真实交易台行为。

---

## 架构分层（最终形态）

```
Layer 2 自动执行（无需审批）
  ├─ ATR 硬止损                    ← _patrol_positions()  已实现
  ├─ 回撤止损（≤ stop_max_weight） ← _patrol_positions()  已实现
  ├─ 制度跃变全组合压缩             ← _patrol_daily_tactical()  待实现
  └─ 高置信度新入场（见阈值）        ← _patrol_daily_tactical()  待实现

Layer 3 人工审批
  ├─ 大仓位止损（> stop_max_weight）← _patrol_positions()  已实现
  ├─ TSMOM 信号翻转减仓             ← _patrol_daily_tactical()  待实现
  ├─ 普通新入场触发                  ← _patrol_daily_tactical()  待实现
  └─ 月度再平衡                     ← orchestrator  已实现
```

---

## 信号体系：双轨 TSMOM

| 信号 | 参数 | 用途 | 更新频率 |
|------|------|------|---------|
| TSMOM-Slow | 12-1 月 | 月度战略权重 | 月频 |
| TSMOM-Fast | 3-1 月  | 日频战术方向确认 | 日频 |

**关键设计原则**：Fast 信号只用于方向确认和翻转检测，不直接生成权重。权重仍由 Slow 信号决定。Fast 信号方向与 Slow 不一致时，视为信号分歧，触发审批而非自动执行。

实现位置：`engine/signal.py` — 扩展 `get_signal_dataframe()` 支持 `lookback_months=3, skip_months=1`，或新增 `get_fast_signal_dataframe()`。

---

## 五类战术事件

### 事件 1：制度跃变（Regime Jump）

**检测条件**：  
- P(risk-off) 单日变化 > 30 ppt（如从 0.20 → 0.52）  
- 或 VIX 单日涨幅 > 25%

**动作**：Layer 2 自动执行 — 将所有多头仓位权重压缩至 `regime_off_max`（默认 8% 上限）  
**触发记录**：写 `SimulatedTrade(trigger_reason="regime_jump_compress")`，`PendingApproval` 写一条 informational 记录（不需要审批，仅留档）

**撤销条件**：次日制度恢复 risk-on 且 P 变化 > 20 ppt → 生成 Layer 3 审批（恢复权重需人工确认）

---

### 事件 2：Fast 信号翻转（TSMOM-Fast Flip）

**检测条件**：  
- TSMOM-Fast 方向与当前持仓方向相反（持多头但 Fast 为 -1，或持空头但 Fast 为 +1）  
- 且 TSMOM-Slow 尚未翻转（否则月度再平衡会处理）

**动作**：Layer 3 审批 — 建议减仓至 50% 当前权重  
**触发记录**：`PendingApproval(approval_type="risk_control", priority="high", triggered_condition="tsmom_fast_flip")`

**不进 Layer 2 的理由**：Fast 信号噪声较大（3个月窗口），误触率高；减仓决策需人工判断是否为短期噪声。

---

### 事件 3：高置信度新入场（High-Confidence Entry）

**检测条件**（全部满足）：  
1. TSMOM-Fast = +1（方向确认）  
2. TSMOM-Slow = +1（快慢一致）  
3. 制度 = risk-on  
4. 当前该 ETF 无持仓（actual_weight ≈ 0）  
5. composite_score ≥ 60（因子综合评分，来自 factor_mad）  
6. 5日动量 z-score ≥ 1.5σ（相对宇宙）

**动作**：  
- 以上全部满足 → Layer 2 自动开仓，权重 = min(vol_parity_weight, 5%)  
- 缺少条件 5 或 6 → Layer 3 审批  

**关键保护**：
- 开仓权重上限 5%（首次入场保守）  
- 每日最多触发 2 个新入场（防止集中进入）  
- 当日已有 Layer 2 止损执行 → 新入场全部降级为 Layer 3

---

### 事件 4：浮亏扩大预警（Drawdown Alert）

**检测条件**：  
- 持仓浮亏（current_price / entry_price - 1）< -8%  
- 且未触发 ATR 硬止损（否则已由 `_patrol_positions()` 处理）

**动作**：Layer 3 审批 — 建议减仓至 50%，附带止损价位建议  
**与 ATR 止损的区别**：ATR 止损跟踪高点，浮亏预警跟踪入场成本。两者独立触发，互不替代。

---

### 事件 5：波动率尖峰压缩（Vol Spike Compress）

**检测条件**：  
- 持仓 ETF 的 21 日 ATR / 价格 > 3%（年化波动率 > 47%）  
- 且该仓位权重 > 目标权重 × 1.5（即权重已超配）

**动作**：Layer 3 审批 — 建议压缩至 vol_parity 目标权重  
**注**：这是对现有 `vol_spike` 逻辑的升级，当前系统已有 vol_spike 检测但动作不完整。

---

## 执行函数规格

```python
def _patrol_daily_tactical(
    t_day: datetime.date,
    result: BatchResult,
    nav: float,
) -> None:
    """
    日频战术巡逻。在 _patrol_positions() 之后运行。
    
    检测顺序（有优先级）：
    1. 制度跃变（最高优先，Layer 2，影响全组合）
    2. 高置信度新入场（Layer 2 / 3，取决于阈值）
    3. Fast 信号翻转（Layer 3）
    4. 浮亏预警（Layer 3）
    5. 波动率尖峰（Layer 3）
    
    制度跃变触发时，跳过 2-5（全组合压缩已覆盖）。
    """
```

**新增 BatchResult 字段**：
```python
@dataclass
class BatchResult:
    ...
    tactical_entries:    list[str]  # 新增：触发新入场的 sector 列表
    tactical_reduces:    list[str]  # 新增：触发减仓的 sector 列表
    regime_jump:         bool       # 新增：是否发生制度跃变
```

---

## 配置参数（新增到 secrets.toml [trading]）

```toml
[trading]
# 现有
auto_execute_stops     = true
auto_execute_entries   = false
monthly_rebalance_auto = false
stop_max_weight_auto   = 0.25

# 新增
auto_execute_regime_compress = true   # 制度跃变自动压缩
auto_execute_high_conf_entry = false  # 高置信度入场自动执行（默认关，调试期）
tactical_entry_max_weight    = 0.05   # 战术入场权重上限
tactical_entry_daily_limit   = 2      # 每日最多自动入场数量
regime_jump_threshold_ppt    = 30     # 制度跃变 P 变化阈值（百分点）
fast_signal_lookback         = 3      # TSMOM-Fast 形成期（月）
fast_signal_skip             = 1      # TSMOM-Fast 跳过期（月）
entry_composite_score_min    = 60     # 高置信度入场最低综合评分
entry_momentum_zscore_min    = 1.5    # 高置信度入场最低 5日动量 z-score
```

---

## 实施顺序

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1 | `engine/signal.py` | 新增 `get_fast_signal_dataframe(lookback=3, skip=1)` |
| 2 | `engine/config.py` | 扩展 `get_trading_config()` 读取新增配置项 |
| 3 | `.streamlit/secrets.toml` | 新增 `[trading]` 配置项 |
| 4 | `engine/daily_batch.py` | 新增 `_patrol_daily_tactical()` 函数 |
| 5 | `engine/daily_batch.py` | `run_daily_batch()` 在 `_patrol_positions()` 之后调用 |
| 6 | `engine/memory.py` | `BatchResult` 新增 `tactical_entries` / `tactical_reduces` / `regime_jump` 字段 |
| 7 | `pages/orchestrator.py` | Section B 展示战术事件；Section A Alerts 展示新入场审批 |
| 8 | `pages/live_dashboard.py` | 顶部横幅展示当日战术动作摘要 |

---

## 方法论红线（不可逾越）

1. **Fast 信号不生成权重** — 只做方向 flag（+1 / -1），权重计算永远走 Slow TSMOM
2. **每日最多 2 个 Layer 2 自动入场** — 防止信号共线时集中建仓
3. **制度跃变当日禁止新入场** — 压缩和扩张不能同时发生
4. **首次入场权重 ≤ 5%** — 战术入场是试探性的，不是全仓建仓
5. **回测验证先于上线** — `_patrol_daily_tactical()` 必须先在历史数据上跑通，验证换手率和 Sharpe 影响，再接入实盘流

---

*参考文件：`docs/master_backlog.md`（P4 批次），`engine/daily_batch.py`（现有 `_patrol_positions()`）*
