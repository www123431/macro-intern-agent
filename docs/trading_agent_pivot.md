# Trading Agent 转向评估与改进建议

**日期**：2026-04-16  
**立场**：从量化金融从业者角度客观评估，不回避局限性

---

## 一、前提声明

本文档基于以下前提撰写：

1. 目标是在 NUS MSBA 毕业项目框架内展示 trading system 经验
2. 时间窗口有限（数月），不是多年期研究项目
3. 诚实的评估优先于乐观的叙事

**核心结论先行**：现有系统与一个可信的 trading system 之间存在结构性差距，部分差距在 MSBA 时间窗口内可弥补，部分不可弥补。本文明确区分两者。

---

## 二、现状诊断：诚实的差距分析

### 2.1 现有系统实际具备的能力（基于代码核查）

| 模块 | 实际状态 | 对 trading system 的价值 |
|------|----------|--------------------------|
| `engine/history.py` — FRED 数据接入 | **真实实现**，含 publication lag 防护 | 直接复用，数据层基础扎实 |
| `engine/history.py` — Walk-forward 框架 | **真实实现**，有 embargo days、temporal isolation | 骨架存在，但信号来源有根本问题（见2.3） |
| `engine/scanner.py` — ETF 资产宇宙 | **真实实现**，18个板块 ETF | 直接复用为信号层资产宇宙 |
| `engine/quant.py` — Sharpe/VaR/bootstrap | **真实实现** | 复用于组合评估层 |
| 宏观 regime 分类 | 人工输入 + LLM 辅助 | 可作为低频 overlay，价值未验证 |
| 决策数据库 | 完整记录，含 edit_ratio | 可作为人工覆盖行为分析数据源 |
| QC 检验 | QC-1/2 有效，QC-3/4 已移除 | 可作为信号过滤层 |

### 2.2 关键缺口

| 缺失模块 | 性质 | MSBA 窗口内可弥补？ |
|----------|------|----------------------|
| 纯结构化信号生成层 | **核心缺失** | 可行，2-3周工作量 |
| 组合构建层（波动率目标化） | 核心缺失 | 部分可行（等权或简单优化） |
| 信号验证（统计显著性） | 方法论 | 部分可行，样本量是硬约束 |
| 执行层 | 基础设施 | 模拟可行，真实执行不建议 |

### 2.3 不可回避的根本性问题

**LLM 知识污染（Look-ahead Contamination）**

这是 LLM-based trading system 特有的、区别于传统回测前视偏差的问题：

- LLM 训练数据截止日期之前的所有历史事件，LLM 实际上"已经知道结果"
- 当你让 LLM 分析"2022年3月的宏观环境"时，它的判断不可避免地受到对后续事件的隐性认知影响
- 这个污染**无法通过工程手段完全消除**，只能在方法论上诚实披露

**关键发现（代码核查后的更新）**：

`engine/history.py` 的 `run_walk_forward_backtest()` 已实现了完整的 walk-forward 时序隔离逻辑，包括 embargo days 和 cutoff_date 防护。但这个框架的信号来源是**LLM 调用**——每个历史时间点的"判断"是让 LLM 分析该时点的数据后生成的。这意味着知识污染问题不是外围风险，而是嵌入在回测架构的核心层。无论 temporal isolation 做得多严格，LLM 对历史事件的隐性认知无法被隔离。

任何声称用 LLM 历史回测得到高收益的论文，如果没有明确讨论这一问题，结论均不可信。

**因此，现有回测框架的正确定位是**：作为前瞻性数据积累管道（从当前时点开始运行，LLM 不知道未来），而非历史回测工具。

---

## 三、最小可行的 Trading System 架构

在 MSBA 时间窗口内，目标不是构建完整的量化交易系统，而是构建一个**有方法论意识、可防守的 research prototype**。

```
[宏观 Regime 层]          ← 现有系统（改进）
  LLM 辅助 + 人工确认
  输出：regime 标签 + 置信度
         ↓
[信号生成层]              ← 新增（核心工作）
  结构化因子信号
  （动量、价值、质量等）
  与 regime 条件性结合
         ↓
[组合构建层]              ← 新增（简化版）
  等权或 1/N 组合
  Regime-conditional 仓位上限
         ↓
[模拟回测层]              ← 新增（核心工作）
  Walk-forward validation
  Regime-aware performance attribution
         ↓
[实证评估层]              ← 新增（答辩核心）
  有无 macro overlay 的 Sharpe 对比
  统计显著性检验
  局限性诚实披露
```

---

## 四、具体改进建议

### 4.1 优先级最高：信号结构化输出

**现状问题**：现有系统输出的是自然语言（"超配科技板块"），无法直接接入量化流程。

**改进方向**：在 `_run_sector_analysis()` 输出中增加结构化权重字段：

```python
# 在 draft dict 中新增
"sector_weights": {
    "Technology": 0.30,
    "Energy": 0.15,
    "Financials": 0.20,
    # ...
}
"regime_label": "risk-off"  # 标准化 regime 分类
```

这是连接现有系统与量化流程的最关键接口。

### 4.2 优先级高：双轨回测架构

现有 `history.py` 的 walk-forward 框架是真实可用的基础设施，但需要在其之上构建一条**不依赖 LLM 的独立信号轨道**用于历史回测。

```
轨道 A — 历史回测（方法论干净，可信）：
  history.py 数据层（FRED + yfinance，已实现）
  → 纯结构化 regime 信号（VIX阈值 / 收益率曲线斜率）
  → 动量/价值因子组合
  → walk-forward 验证（复用现有框架的时序隔离逻辑）
  → Sharpe 对比（有/无 regime overlay）

轨道 B — 前瞻性积累（诚实标注，不能用于历史回测）：
  现有系统从当前时点起运行
  → LLM regime 分类 + 人工确认（edit_ratio 追踪）
  → Clean Zone 胜率积累
  → N 个月后做 LLM 置信度校准分析
```

**复用策略**：`history.py` 中的 `get_fred_snapshot()`、`get_vix_on()`、`get_sector_momentum()` 直接复用；`run_walk_forward_backtest()` 的时序隔离逻辑保留，但把 LLM 调用替换为纯结构化信号计算。

### 4.3 优先级高：Regime-conditional 因子组合

参考 Daniel & Moskowitz (2016) 的框架，用**已有历史数据**验证：

- 定义两种 regime（risk-on / risk-off），用 VIX 阈值或收益率曲线斜率等纯结构化指标
- 构建 regime-conditional 板块组合（等权）
- 计算条件 Sharpe、最大回撤、Sortino
- 对比无条件基准（等权全板块）

这部分完全不依赖 LLM，方法论干净，可以作为系统的**实证基准层**。

### 4.4 优先级高：数据驱动的 Regime 检测（替换人工输入）

**现状问题**：regime 由人工填写，引入主观偏差且无法用于历史回测。

**改进方向**：用 Hamilton (1989) Markov Switching Model 做数据驱动的 regime 分类，输入已有的 FRED 数据（`history.py` 的 `get_fred_snapshot()` 已实现）。

```python
# engine/regime.py — 新建
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def fit_markov_regime(train_data: pd.Series, n_regimes: int = 2) -> pd.Series:
    """
    输入：单一宏观序列（如 yield_spread），仅用训练窗口数据估计参数。
    输出：每个日期的 regime 概率（滤波概率，非平滑概率）

    关键约束：
    1. 必须使用 filtered_marginal_probabilities，不能用 smoothed_marginal_probabilities。
       平滑概率使用了未来数据（Kalman 平滑器向后传播），在回测中等同于前视偏差。
    2. Walk-forward 回测中，每个测试时点必须只用该时点之前的训练数据重新估计模型，
       不能用全样本估计后再切片——那样引入的是参数估计层面的前视偏差。
    3. 月度 FRED 数据下，60-120 个观测值的 MSM 参数估计不稳定，
       置信区间较宽，结论须附带不确定性说明。
    """
    model = MarkovRegression(train_data, k_regimes=n_regimes, trend="c")
    result = model.fit()
    # 必须用滤波概率，不用平滑概率
    return result.filtered_marginal_probabilities.iloc[:, 1]  # P(regime=1 | data up to t)
```

**这个改进的研究价值**：把系统自动分类的 regime 与人工判断的 regime 做对比，回答一个有意义的问题：

> "人类对宏观 regime 的主观判断，相对于统计模型，是否包含增量信息？"

这直接连接 edit_ratio 的逻辑——如果人类覆盖了模型判断且方向正确，edit_ratio 和 regime 分歧量就有了研究价值。

**与现有代码的连接点**：
- 数据输入：直接复用 `history.py` 的 `get_fred_snapshot()` 和 `get_vix_on()`
- 结果存储：在 `memory.py` 的 `DecisionLog` 中新增 `regime_model` 字段，与 `macro_regime`（人工）并存
- UI：在 Tab1 展示模型 regime vs. 人工 regime 的对比，差异大时触发提示

### 4.5 优先级高：严格统计检验框架

**现状问题**：现有胜率统计是原始命中率，没有处理多重检验问题。如果测试了多个参数组合后选最好的展示，Sharpe 会系统性虚高。

**需要加入的三个层次**：

**层次一：Walk-forward（已有骨架，需改造信号层）**

`history.py` 的 `run_walk_forward_backtest()` 时序隔离逻辑已实现，参见 §4.2 的双轨架构改造。

**层次二：Deflated Sharpe Ratio（López de Prado 2018）**

校正因多次策略筛选导致的 Sharpe 虚高。公式依赖两个输入：策略的 Sharpe 分布统计特征 + 试验次数。

```python
# engine/quant.py — 新增函数
import numpy as np
import scipy.stats as stats

def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_trials: int,
    skew: float,
    kurt: float,
    T: int,
) -> float:
    """
    López de Prado (2018) Deflated Sharpe Ratio.
    公式来源：Advances in Financial Machine Learning, Ch.8

    sharpe_obs : 观测到的（最优）年化 Sharpe Ratio
    n_trials   : 实际尝试过的策略/参数组合数（必须诚实记录，低估会高估 DSR）
    skew       : 策略收益序列的偏度
    kurt       : 策略收益序列的峰度（excess kurtosis）
    T          : 样本内观测数量（交易日）

    注意事项：
    - n_trials 必须包含所有被尝试但未展示的策略，否则校正失效
    - 样本量不足时（T < 60），DSR 估计本身不稳定
    """
    # E[max SR | N iid trials] via Extreme Value Theory approximation
    euler_gamma = 0.5772156649
    sr_star = (
        (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
        + euler_gamma       * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    # DSR: probability that observed SR exceeds expected max under null
    numerator   = (sharpe_obs - sr_star) * np.sqrt(T - 1)
    denominator = np.sqrt(1 - skew * sharpe_obs + (kurt - 1) / 4.0 * sharpe_obs ** 2)
    if denominator <= 0:
        return float("nan")  # 数值不稳定，不报告
    return float(stats.norm.cdf(numerator / denominator))
```

**层次三：BHY 多重检验校正**

当同时测试多个板块 × 多个 regime 组合时，需要控制 False Discovery Rate。

```python
from statsmodels.stats.multitest import multipletests

def bhy_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg-Yekutieli 校正（适用于相关检验）
    返回：每个检验是否在校正后仍显著
    """
    reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_by")
    return reject.tolist()
```

**答辩价值**：能报告 DSR 和 BHY 校正后的结果，是区分"会用公式"和"懂统计推断"的实质分水岭。即使校正后结果不再显著，诚实报告本身就是方法论贡献。

### 4.6 优先级中：置信度校准模块

**现状问题**：`confidence_score` 是 LLM 自报值，校准性未知。

**改进方向**：积累足够样本后（建议 ≥ 50 条），对 `confidence_score` 做事后校准分析：

```python
# 按置信度分桶，计算各桶的实际胜率
# 理想情况：confidence=70% 的判断，实际胜率接近 70%
# 现实情况：LLM 通常系统性过度自信
```

即使结果证明 LLM 置信度完全不可信，这个分析本身也是有价值的学术发现。

### 4.7 优先级低：Experience Buffer（Reflexion 架构）

在现有 `verify_pending_decisions()` 基础上，为每条失败的判断生成结构化复盘：

```python
# experience_buffer 表结构
{
    "regime_label": "risk-off",
    "sector": "Technology",
    "failure_mode": "momentum_ignored",  # 归因
    "lesson": "在 risk-off + 高 VIX 环境下，做多科技板块需要更高的动量确认门槛"
}
```

**重要约束**：这个机制的价值依赖于 regime 标签的一致性。如果 regime 分类本身不稳定，Buffer 会积累噪声。优先级低于前三项。

---

## 五、实证验证框架：答辩可防守的最低标准

### 5.1 必须回答的问题

答辩委员会会问：

1. **"你的 macro overlay 改善了组合表现吗？"**  
   → 回答：展示 regime-conditional 组合 vs. 无条件基准的 Sharpe 对比，用纯结构化 regime 信号（非 LLM）做历史验证

2. **"LLM 信号的部分怎么验证？"**  
   → 回答：诚实说明知识污染问题，展示前瞻性信号积累计划，给出 preliminary calibration 分析

3. **"你的结果统计显著吗？"**  
   → 回答：报告 t-stat，承认样本量限制，讨论 Type II 错误风险

### 5.2 绝对不能做的事

- 用 LLM 对历史时期重新生成信号然后声称"回测验证"
- 不报告 t-stat 只报告收益率
- 不设对照组（无 overlay 的基准）
- 挑选最好的结果窗口展示

### 5.3 推荐的诚实叙事结构

> "本系统设计为 macro overlay 研究原型。历史验证部分使用纯结构化 regime 信号（Markov Switching）避免 LLM 知识污染问题；LLM-based 信号层已开始前瞻性数据积累，校准结果将在 [N] 个月后更新。当前实证结果为 preliminary，统计功效受限于样本量。"

这个叙事在学术上是诚实的，在业界面试中反而会加分——因为它证明你理解方法论局限，而不是被结果驱动。

---

## 六、时间与优先级规划

| 阶段 | 工作内容 | 预计工作量 | 产出 |
|------|----------|------------|------|
| Phase 1 | 信号结构化输出 + regime 标签标准化 | 1-2周 | 接口就绪 |
| Phase 2 | Markov Switching regime 检测模块 | 1-2周 | 数据驱动 regime，可与人工对比 |
| Phase 3 | 双轨回测架构（纯结构化信号轨道） | 2-3周 | 可信的历史实证基准 |
| Phase 4 | DSR + BHY 统计检验层 | 1周 | 统计严谨性，答辩核心防线 |
| Phase 5 | LLM 置信度校准分析 | 1周（需积累数据） | 方法论贡献点 |
| Phase 6 | Experience Buffer（可选） | 1-2周 | 加分项 |

**不建议在 MSBA 时间窗口内尝试的**：
- 真实资金执行
- 高频信号
- 复杂组合优化（Black-Litterman 等）
- 多资产类别扩展

---

## 七、必须诚实披露的局限性

无论在报告还是答辩中，以下局限性必须主动说明：

1. **LLM 知识污染**：所有涉及 LLM 信号的历史分析结论均受此影响，无法完全消除
2. **样本量不足**：当前 Clean Zone 样本量不支持任何统计推断，仅为描述性统计
3. **无实盘验证**：所有回测结果为模拟，未考虑实际交易成本、滑点、流动性约束
4. **regime 分类的主观性**：人工输入的 regime 标签引入主观偏差，无法与纯系统化基准直接比较
5. **单一资产类别**：当前仅覆盖股票板块，宏观 overlay 对跨资产配置的适用性未验证

---

## 八、对就业目标的实际评估

**如果目标是买方 quant researcher 岗位**：

本文档描述的改进方向（结构化信号 + regime-conditional 回测 + 置信度校准）足以展示 trading system 思维。重点在方法论严谨性，而非系统复杂度。

**如果目标是系统化 quant 基金（Two Sigma、Citadel 类）**：

这个方向不够用。这类机构看的是：高频信号研究、因子 alpha 衰减分析、统计套利实现、或竞赛/论文成果。建议在此项目之外单独构建一个纯系统化因子研究项目作为补充。

**如果目标是宏观对冲基金或 CTA**：

这个方向契合度最高。宏观 overlay + 制度自觉 + 人机协作的定位，与宏观基金的研究文化最接近。改进后的系统作为面试谈资有实际说服力。
