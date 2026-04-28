# P2 施工蓝图（复杂项）

> 覆盖范围：P2-11→12（Universe 管理 + ETF 扩展）/ P2-13（FactorMAD）/ P2-14（双变量 MSM）/ P2-17（新闻升级）  
> 简单项（P2-3/6/7/8/10）和中等项（P2-1/2/4/5/9/15/16）按 master_backlog.md 描述直接实施，无需蓝图。  
> 日期：2026-04-21

---

## BP-A｜P2-11 Universe 管理框架 → P2-12 ETF 扩展

### 背景与风险

**为什么需要蓝图**：`SECTOR_ETF`（`engine/history.py`）当前是硬编码静态字典，所有下游模块直接 `from engine.history import SECTOR_ETF` 导入。P2-12 要新增 15 个 ETF，若直接扩展静态字典，`within-class CSMOM`（类别内截面排序）将被破坏——债券 ETF 和权益 ETF 混排会导致信号失真。P2-11 必须先建立 `asset_class` 分层机制，P2-12 才能安全纳入。

**改动波及范围**：`engine/history.py` → `engine/signal.py` → `engine/portfolio.py` → `engine/backtest.py` → `pages/signal_dashboard.py` → `ui/tabs.py`

---

### A-1 新建 `engine/universe_manager.py`

```python
# engine/universe_manager.py

from __future__ import annotations
import datetime
import logging
from dataclasses import dataclass, field
from engine.memory import SessionFactory, engine as db_engine
from sqlalchemy import Column, Integer, String, Boolean, Date, Float, DateTime, text
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)
_Base = declarative_base()


class UniverseETF(_Base):
    """
    单一事实来源：所有可纳入 ETF 的完整注册表。
    每次 get_active_universe() 调用时读取，替代 history.py 的 SECTOR_ETF 静态字典。
    """
    __tablename__ = "universe_etfs"

    id            = Column(Integer,     primary_key=True, autoincrement=True)
    sector        = Column(String(100), nullable=False, unique=True)   # 中文名，与历史一致
    ticker        = Column(String(20),  nullable=False)
    asset_class   = Column(String(20),  nullable=False)  # "equity_sector" / "equity_factor" / "fixed_income" / "commodity" / "volatility"
    batch         = Column(Integer,     nullable=False, default=0)     # 0=初始18 / 1=批次A / 2=批次B
    inception_date= Column(Date,        nullable=True)   # 用于 survivorship bias audit
    active        = Column(Boolean,     default=True)    # False = 流动性不足，自动标记
    added_at      = Column(Date,        nullable=True)
    removed_at    = Column(Date,        nullable=True)
    notes         = Column(String(200), nullable=True)


def init_universe_db() -> None:
    _Base.metadata.create_all(db_engine)
    _seed_initial_universe()


def _seed_initial_universe() -> None:
    """一次性写入初始 18 个 ETF（幂等，已存在则跳过）。"""
    INITIAL = [
        # (sector, ticker, inception_date)
        ("AI算力\\半导体",   "SMH",  "2000-05-05"),
        ("科技成长(纳指)",   "QQQ",  "1999-03-10"),
        ("生物科技",         "XBI",  "2006-01-31"),
        ("金融",             "XLF",  "1998-12-22"),
        ("能源",             "XLE",  "1998-12-22"),
        ("工业",             "XLI",  "1998-12-22"),
        ("非必需消费",       "XLY",  "1998-12-22"),
        ("必需消费",         "XLP",  "1998-12-22"),
        ("医疗",             "XLV",  "1998-12-22"),
        ("公用事业",         "XLU",  "1998-12-22"),
        ("材料",             "XLB",  "1998-12-22"),
        ("房地产",           "XLRE", "2015-10-07"),
        ("通信服务",         "XLC",  "2018-06-18"),
        ("黄金",             "GLD",  "2004-11-18"),
        ("原油(期货ETF)",    "USO",  "2006-04-10"),
        ("新兴市场",         "EEM",  "2003-04-07"),
        ("欧洲发达市场",     "VGK",  "2005-03-04"),
        ("中概股",           "KWEB", "2013-01-31"),
    ]
    with SessionFactory() as session:
        for sector, ticker, inc in INITIAL:
            exists = session.query(UniverseETF).filter_by(sector=sector).first()
            if not exists:
                session.add(UniverseETF(
                    sector=sector, ticker=ticker,
                    asset_class="equity_sector",
                    batch=0,
                    inception_date=datetime.date.fromisoformat(inc),
                    active=True,
                    added_at=datetime.date(2024, 1, 1),
                ))
        session.commit()


def get_active_universe(asset_classes: list[str] | None = None) -> dict[str, str]:
    """
    返回 {sector: ticker}，仅含 active=True 的行。
    asset_classes 过滤：None = 全部，["equity_sector"] = 只取权益。
    替代 SECTOR_ETF 静态字典。
    """
    with SessionFactory() as session:
        q = session.query(UniverseETF).filter(UniverseETF.active == True)
        if asset_classes:
            q = q.filter(UniverseETF.asset_class.in_(asset_classes))
        rows = q.all()
    return {r.sector: r.ticker for r in rows}


def get_universe_by_class() -> dict[str, dict[str, str]]:
    """
    返回 {asset_class: {sector: ticker}}。
    供 within-class CSMOM 使用。
    """
    with SessionFactory() as session:
        rows = session.query(UniverseETF).filter(UniverseETF.active == True).all()
    result: dict[str, dict[str, str]] = {}
    for r in rows:
        result.setdefault(r.asset_class, {})[r.sector] = r.ticker
    return result


@dataclass
class UniverseHealthReport:
    checked_at:   datetime.date
    inactive_flagged: list[str]  # 自动标记为 inactive 的 sector
    warnings: list[str]


def universe_health_check(as_of: datetime.date | None = None) -> UniverseHealthReport:
    """
    每月第一个交易日调用。
    检测 ADV（20日均量）< 阈值 → 标记 inactive。
    阈值：equity=1M shares/day，fixed_income=500K，commodity=300K。
    """
    import yfinance as yf
    if as_of is None:
        as_of = datetime.date.today()

    ADV_THRESHOLDS = {
        "equity_sector":  1_000_000,
        "equity_factor":  1_000_000,
        "fixed_income":     500_000,
        "commodity":        300_000,
        "volatility":       200_000,
    }
    inactive_flagged: list[str] = []
    warnings: list[str] = []

    with SessionFactory() as session:
        rows = session.query(UniverseETF).filter(UniverseETF.active == True).all()
        start = as_of - datetime.timedelta(days=35)
        for row in rows:
            try:
                dl = yf.download(row.ticker, start=str(start), end=str(as_of),
                                 progress=False, auto_adjust=True)
                if dl.empty or "Volume" not in dl.columns:
                    warnings.append(f"{row.sector}: 无法获取成交量数据")
                    continue
                adv = float(dl["Volume"].tail(20).mean())
                threshold = ADV_THRESHOLDS.get(row.asset_class, 500_000)
                if adv < threshold:
                    row.active = False
                    row.removed_at = as_of
                    inactive_flagged.append(row.sector)
                    logger.warning("Universe: %s (%s) ADV=%.0f < %.0f → inactive",
                                   row.sector, row.ticker, adv, threshold)
            except Exception as e:
                warnings.append(f"{row.sector}: {e}")
        session.commit()

    return UniverseHealthReport(
        checked_at=as_of,
        inactive_flagged=inactive_flagged,
        warnings=warnings,
    )
```

**关键约束**：`UniverseETF.sector` 字符串必须与现有 `SECTOR_ETF` 键完全一致（包含反斜杠、括号），否则 DecisionLog 中的历史 `sector_name` 将无法与新宇宙匹配。

---

### A-2 修改 `engine/history.py`

**改动目标**：`get_active_sector_etf()` 改为调用 `universe_manager.get_active_universe()`，使动态宇宙成为全局事实来源。

```python
# engine/history.py — 修改 get_active_sector_etf()

def get_active_sector_etf() -> dict[str, str]:
    """
    返回当前活跃的 sector→ETF 映射。
    优先读 UniverseETF 表（P2-11后），回退到静态 SECTOR_ETF。
    """
    try:
        from engine.universe_manager import get_active_universe
        active = get_active_universe()
        if active:
            return active
    except Exception:
        pass
    # 回退：旧版静态字典（P2-11 未初始化时）
    return dict(SECTOR_ETF)
```

**注意**：`SECTOR_ETF` 静态字典**保留不删**，作为回退和文档参考。所有新代码通过 `get_active_sector_etf()` 读取，不直接导入 `SECTOR_ETF`。

---

### A-3 修改 `engine/signal.py` — within-class CSMOM

这是 P2-12 批次 B 的核心改动。当前 `get_signal_dataframe()` 对所有 18 个行业 ETF 做统一截面排序，批次 B 纳入债券/商品后此逻辑失效。

**改动位置**：`compute_composite_scores()` 内的 CSMOM 排序段。

```python
# engine/signal.py — compute_composite_scores() 内

# 旧逻辑（仅 equity_sector，全局排序）：
# base["csmom_rank"] = base["raw_return"].rank(pct=True) * 100

# 新逻辑（within-class 排序）：
from engine.universe_manager import get_universe_by_class

universe_by_class = get_universe_by_class()

# 为每个资产类别单独排序，再合并
rank_series = pd.Series(dtype=float, name="csmom_rank")
for asset_class, class_map in universe_by_class.items():
    class_sectors = list(class_map.keys())
    mask = base.index.isin(class_sectors)
    if mask.sum() < 2:
        # 单资产类别无截面意义，赋 50（中性）
        rank_series = pd.concat([rank_series,
                                 pd.Series(50.0, index=base.index[mask])])
        continue
    class_ranks = base.loc[mask, "raw_return"].rank(pct=True) * 100
    rank_series = pd.concat([rank_series, class_ranks])

base["csmom_rank"] = rank_series.reindex(base.index).fillna(50.0)
```

**触发条件**：批次 A 纳入（全部 equity_factor）时此改动不必要，因同属权益可合并排序。批次 B 纳入固定收益后**必须激活**。建议 P2-12 批次 B 实施时同步切换，不提前。

---

### A-4 P2-12 批次 A：新增 8 个权益 ETF

**在 `_seed_initial_universe()` 同类函数或独立 migration 中写入**：

| sector（中文名） | ticker | asset_class | inception_date |
|----------------|--------|-------------|----------------|
| 小盘价值 | IWN | equity_factor | 2000-07-24 |
| 小盘成长 | IWO | equity_factor | 2000-07-24 |
| 动量因子 | MTUM | equity_factor | 2013-04-16 |
| 低波动因子 | USMV | equity_factor | 2011-10-20 |
| 质量因子 | QUAL | equity_factor | 2013-07-18 |
| 日本 | EWJ | equity_sector | 1996-03-12 |
| 中国A股 | ASHR | equity_sector | 2013-11-06 |
| 印度 | INDA | equity_sector | 2012-02-02 |

**实施步骤**：
1. 写入 `universe_etfs` 表（batch=1）
2. `universe_health_check()` 首次运行，确认 ADV 达标
3. 重跑 backtest，确认新 ETF 的 CSMOM 排序未破坏原有 18 个行业结果
4. `engine/universe_audit.py` 的 `audit_universe()` 自动覆盖新 ETF（已读 UniverseETF 表）

---

### A-5 P2-12 批次 B：新增 7 个跨资产 ETF（顺序约束）

**前置检查（PRE-8）**：确认 yfinance `Adj Close` 对 AGG/IEF/TIP 含票息再投资后再启动。

| sector | ticker | asset_class | 特殊处理 |
|--------|--------|-------------|---------|
| 美国综合债 | AGG | fixed_income | within-class CSMOM 隔离 |
| 美国中期国债 | IEF | fixed_income | 同上 |
| 通胀保值债 | TIP | fixed_income | 同上 |
| 黄金矿业 | GDX | commodity | 与 GLD 同 class |
| 农产品 | DBA | commodity | — |
| 波动率 | VXX | volatility | **信号极性相反**，tsmom=-1 买入 |
| 房地产信托 | REM | equity_sector | 并入 equity_sector |

**VXX 极性翻转**：VXX 反转动量（动量为负时往往是买入时机）。在 `get_signal_dataframe()` 内加特殊标记：
```python
REVERSE_MOMENTUM_TICKERS = {"VXX"}  # 信号极性翻转
# 在 tsmom 计算后：
if ticker in REVERSE_MOMENTUM_TICKERS:
    df.loc[sector, "tsmom"] = -df.loc[sector, "tsmom"]
```

**A-3 within-class CSMOM** 在此步激活。

---

### A-6 调用点清单（迁移检查表）

实施 P2-11 后，需确认以下位置已改为 `get_active_sector_etf()` 而非直接使用 `SECTOR_ETF`：

| 文件 | 当前用法 | 目标用法 |
|------|---------|---------|
| `engine/signal.py` line ~56 | `from engine.history import SECTOR_ETF` | 保留 import，运行时改为 `get_active_sector_etf()` |
| `engine/history.py` `get_prices()` | `list(SECTOR_ETF.values())` | `list(get_active_sector_etf().values())` |
| `ui/tabs.py` `AUDIT_TICKERS` | 静态字典 | 确认是否需要动态化（暂时可保持静态，P2-11 后再议） |
| `engine/daily_batch.py` | 通过 signal.py 间接使用 | 无需直接改 |
| `pages/signal_dashboard.py` | 通过 signal.py 间接使用 | 无需直接改 |

---

## BP-B｜P2-13 FactorMAD — Alpha 因子自动挖掘引擎

### 背景与风险

**为什么需要蓝图**：FactorMAD 是纯量化模块（Track A），新建 2 张表 + 1 个新引擎文件，与现有 LLM 流程零交互。主要跑偏风险：① IC/ICIR 计算窗口定义不清楚导致前视偏差；② 因子生命周期（active/inactive）状态机如果跨 session 不一致会污染复合信号；③ FactorMAD 信号如何与 TSMOM 信号组合（替换？叠加？）需提前明确。

**设计决策**：FactorMAD 输出**叠加**到现有 `composite_score`，不替换 TSMOM。TSMOM 保持 50% 权重主导，FactorMAD 贡献最多 20% 替代当前 `regime_score` 位置（当 FactorMAD 有足够因子时）。在 FactorMAD 因子数量 < 3 时，退化为当前权重方案。

**候选因子发现流程（四层防御）**：

```
Proposer 生成候选因子
  ↓ 【Layer 1：MI 污染扫描】  ← 纯统计，无 LLM，拦截隐蔽前视
  MI 异常？ → 是 → 直接终止，写 rejection_reason，不进入辩论
  ↓ 否
  【Layer 2：Critic 辩论 + 验证集回测】  ← 现有流程（B-2/B-3）
  通过辩论？ → 否 → 迭代或终止
  ↓ 是
  【Layer 2：测试集最终验证（ICIR ≥ 0.3 + 相关性 < 0.7）】
  ↓ 通过
  【Layer 3：符号回归结构审计】  ← 新增，生成侦探报告（非门控）
  ↓ 生成审计报告
  【Layer 4：Supervisor 人工裁决】  ← approve / reject / pending_further_review
```

---

### B-1 数据库 Schema

新增两张表（写入 `engine/memory.py`，追加 ORM class + `_migrate_db()` 条目）：

```python
class FactorDefinition(Base):
    """
    已注册的 alpha 因子定义。每个因子对应一个可计算函数。
    """
    __tablename__ = "factor_definitions"

    id            = Column(Integer,     primary_key=True, autoincrement=True)
    factor_id     = Column(String(50),  nullable=False, unique=True)  # "mom_3m" / "rev_1m" / "vol_adj_mom"
    description   = Column(String(200), nullable=True)
    asset_class   = Column(String(20),  nullable=False, default="equity_sector")
    active        = Column(Boolean,     default=True)   # ICIR < 0.15 连续2月 → False
    created_at    = Column(DateTime,    default=datetime.datetime.utcnow)


class FactorICIR(Base):
    """
    每月计算一次 IC / ICIR。
    IC = Spearman 相关（因子值, 下月截面收益）。
    ICIR = rolling 12月 mean(IC) / std(IC)。
    """
    __tablename__ = "factor_icir"

    id            = Column(Integer,     primary_key=True, autoincrement=True)
    factor_id     = Column(String(50),  nullable=False)   # FK → factor_definitions.factor_id
    calc_date     = Column(Date,        nullable=False)   # 计算当月月末
    ic_value      = Column(Float,       nullable=True)    # 当月 IC
    icir_12m      = Column(Float,       nullable=True)    # 滚动12月 ICIR
    n_assets      = Column(Integer,     nullable=True)    # 参与计算的资产数
    asset_class   = Column(String(20),  nullable=False, default="equity_sector")

    __table_args__ = (
        UniqueConstraint("factor_id", "calc_date", "asset_class",
                         name="uq_factor_icir_date_class"),
    )
```

**`_migrate_db()` 追加**：
```python
# 在最后一个 with engine.connect() 块之后追加
Base.metadata.create_all(engine)  # 已存在，创建 factor_definitions / factor_icir
```

---

### B-1.5 Layer 1：MI 污染扫描（`engine/factor_mad.py` 新增函数）

**定位**：在 Proposer 生成候选因子代码后，进入 Critic 辩论前，执行一次纯统计的互信息预扫描。实现成本低（sklearn，无 LLM 调用），却能拦截 Critic Agent 逻辑审查难以发现的隐蔽统计前视。

**实现**（追加到 `factor_mad.py`）：

```python
from sklearn.feature_selection import mutual_info_regression

# 基准因子 MI 白名单：这些因子已知无前视，用于校准 MI 基线
_BASELINE_FACTOR_IDS = ["mom_3m", "rev_1m", "vol_adj_mom_6m", "trend_strength"]
_MI_CONTAMINATION_MULTIPLIER = 2.0  # 候选 MI > 基准均值 × 此倍数 → 可疑


def compute_factor_mi(
    factor_fn: callable,
    prices: pd.DataFrame,
    train_end: datetime.date,
    forward_return_days: int = 22,
    n_cross_sections: int = 24,
) -> float | None:
    """
    在训练集上估算因子值与未来收益之间的互信息。
    
    通过滚动截面（每月一次）聚合 (factor_val, fwd_ret) 样本对，
    再用 mutual_info_regression 估算 MI。
    
    ⚠️ 重要：因子值必须用 t-forward_days 时的数据计算，
    未来收益用 t-forward_days → t，两者不重叠，无前视。
    如果一个因子存在前视偏差（用了 t+1 数据），其 MI 会异常偏高。
    """
    prices.index = pd.to_datetime(prices.index).normalize()
    end_idx = prices.index[prices.index <= pd.Timestamp(train_end)]
    if len(end_idx) < forward_return_days + n_cross_sections:
        return None

    samples_factor, samples_fwd = [], []
    for i in range(n_cross_sections):
        t_end_i  = end_idx[-(i * forward_return_days + 1)]
        t_ref_i  = end_idx[-(i * forward_return_days + 1 + forward_return_days)]
        pre_prices = prices[prices.index <= t_ref_i]
        if len(pre_prices) < 50:
            continue
        fvals = factor_fn(pre_prices)
        if fvals.empty or fvals.isna().all():
            continue
        fwd_ret = prices.loc[t_end_i] / prices.loc[t_ref_i] - 1
        combined = pd.DataFrame({"f": fvals, "r": fwd_ret}).dropna()
        if len(combined) < 4:
            continue
        samples_factor.extend(combined["f"].tolist())
        samples_fwd.extend(combined["r"].tolist())

    if len(samples_factor) < 20:
        return None

    X = np.array(samples_factor).reshape(-1, 1)
    y = np.array(samples_fwd)
    mi = mutual_info_regression(X, y, n_neighbors=5, random_state=42)[0]
    return float(mi)


def scan_mi_contamination(
    candidate_fn: callable,
    prices: pd.DataFrame,
    train_end: datetime.date,
) -> dict:
    """
    Layer 1 入口。返回扫描结果 dict：
    {
        "candidate_mi": float,
        "baseline_mi_mean": float,
        "ratio": float,
        "flagged": bool,
        "reason": str,
    }
    
    限制：MI 阈值 (2×) 是经验值，未在本宇宙上系统验证。
    强因子的真实 MI 本身不低，若基准因子较强则误报率上升。
    建议积累 ≥10 个已知无前视因子后重新校准倍数。
    """
    baseline_mis = []
    for fid in _BASELINE_FACTOR_IDS:
        fn = FACTOR_REGISTRY.get(fid)
        if fn is None:
            continue
        mi = compute_factor_mi(fn, prices, train_end)
        if mi is not None:
            baseline_mis.append(mi)

    if not baseline_mis:
        return {"candidate_mi": None, "baseline_mi_mean": None,
                "ratio": None, "flagged": False, "reason": "基准 MI 无法计算，跳过扫描"}

    baseline_mean = float(np.mean(baseline_mis))
    candidate_mi  = compute_factor_mi(candidate_fn, prices, train_end)

    if candidate_mi is None:
        return {"candidate_mi": None, "baseline_mi_mean": baseline_mean,
                "ratio": None, "flagged": False, "reason": "候选因子 MI 无法计算，跳过扫描"}

    ratio   = candidate_mi / baseline_mean if baseline_mean > 0 else 0.0
    flagged = ratio > _MI_CONTAMINATION_MULTIPLIER

    return {
        "candidate_mi":    round(candidate_mi, 6),
        "baseline_mi_mean": round(baseline_mean, 6),
        "ratio":            round(ratio, 3),
        "flagged":          flagged,
        "reason":           f"MI ratio={ratio:.2f}，阈值={_MI_CONTAMINATION_MULTIPLIER}×" if flagged
                            else f"MI ratio={ratio:.2f}，未超阈值",
    }
```

**调用点**（在候选因子代码 eval 后、发起 Critic LLM 调用前）：

```python
mi_result = scan_mi_contamination(candidate_fn, train_prices, train_end_date)
if mi_result["flagged"]:
    # 直接终止，写入 DiscoveredFactor(status="rejected", rejection_reason=...)
    logger.warning("FactorMAD Layer1 拦截: %s", mi_result["reason"])
    return FactorScanResult(passed=False, mi_report=mi_result)
```

---

### B-2 新建 `engine/factor_mad.py`

```python
"""
FactorMAD — Alpha 因子自动挖掘引擎
=====================================
Track A 纯量化模块。与 LLM 流程零交互。
"""
from __future__ import annotations
import datetime
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from engine.history import get_active_sector_etf  # P2-11 后使用动态宇宙
from engine.memory import SessionFactory, FactorDefinition, FactorICIR

logger = logging.getLogger(__name__)

# ── 已注册因子库 ────────────────────────────────────────────────────────────────

FACTOR_REGISTRY: dict[str, callable] = {}

def register_factor(factor_id: str, description: str = ""):
    """装饰器：注册因子计算函数。签名：(prices: pd.DataFrame) -> pd.Series (index=sector)"""
    def decorator(fn):
        FACTOR_REGISTRY[factor_id] = fn
        return fn
    return decorator


@register_factor("mom_3m", "3个月动量（跳过最近1月）")
def factor_mom_3m(prices: pd.DataFrame) -> pd.Series:
    if len(prices) < 65:
        return pd.Series(dtype=float)
    ret = prices.iloc[-65] / prices.iloc[-22] - 1   # T-65 到 T-22，跳过1月
    return ret


@register_factor("rev_1m", "1个月反转")
def factor_rev_1m(prices: pd.DataFrame) -> pd.Series:
    if len(prices) < 22:
        return pd.Series(dtype=float)
    return -(prices.iloc[-1] / prices.iloc[-22] - 1)  # 反转：负号


@register_factor("vol_adj_mom_6m", "波动率调整6月动量")
def factor_vol_adj_mom_6m(prices: pd.DataFrame) -> pd.Series:
    if len(prices) < 130:
        return pd.Series(dtype=float)
    ret_6m  = prices.iloc[-130] / prices.iloc[-22] - 1
    vol_21d = prices.pct_change().iloc[-22:].std() * np.sqrt(252)
    return ret_6m / vol_21d.replace(0, np.nan)


@register_factor("trend_strength", "SMA200 偏离度")
def factor_trend_strength(prices: pd.DataFrame) -> pd.Series:
    if len(prices) < 200:
        return pd.Series(dtype=float)
    sma200 = prices.iloc[-200:].mean()
    return (prices.iloc[-1] / sma200 - 1)


# ── IC 计算 ────────────────────────────────────────────────────────────────────

def compute_monthly_ic(
    factor_id: str,
    calc_date: datetime.date,
    lookback_prices_days: int = 280,
    forward_return_days:  int = 22,
    asset_class: str = "equity_sector",
) -> float | None:
    """
    计算单因子在 calc_date 的 IC（Spearman rank correlation）。
    
    防止前视：因子值用 T-forward_return_days 的价格计算；
              实际收益用 T-forward_return_days → T 的价格计算。
    即：在过去某个时点计算因子，预测从那个时点起 forward_return_days 的收益。
    """
    import yfinance as yf
    from engine.universe_manager import get_active_universe
    sector_etf = get_active_universe(asset_classes=[asset_class])
    if not sector_etf:
        return None

    tickers = list(sector_etf.values())
    sectors = list(sector_etf.keys())

    start = calc_date - datetime.timedelta(days=lookback_prices_days + forward_return_days + 30)
    end   = calc_date + datetime.timedelta(days=5)

    try:
        dl = yf.download(tickers, start=str(start), end=str(end),
                         progress=False, auto_adjust=True)
        if dl.empty:
            return None
        prices = dl["Close"] if "Close" in dl else dl
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [c[0] for c in prices.columns]
        prices = prices.reindex(columns=tickers).dropna(how="all")
    except Exception as e:
        logger.warning("FactorMAD IC download error: %s", e)
        return None

    # 对齐到 calc_date（用最近的交易日）
    prices.index = pd.to_datetime(prices.index).normalize()
    t_end   = prices.index[prices.index <= pd.Timestamp(calc_date)]
    if len(t_end) < forward_return_days + 20:
        return None
    t_ref   = t_end[-forward_return_days]   # 因子计算时点（无前视）
    t_start = t_end[-1]                     # 实际为 calc_date

    # 切片：calc_date 前 lookback 窗口用于因子计算
    pre_prices  = prices[prices.index <= t_ref]
    if len(pre_prices) < 50:
        return None

    # 因子值（ticker 索引）
    factor_fn = FACTOR_REGISTRY.get(factor_id)
    if factor_fn is None:
        return None
    ticker_to_sector = {v: k for k, v in sector_etf.items()}
    factor_vals = factor_fn(pre_prices).rename(index=ticker_to_sector)

    # 实际收益（t_ref → calc_date）
    fwd_rets = (prices.loc[t_start] / prices.loc[t_ref] - 1).rename(index=ticker_to_sector)

    # 合并并计算 Spearman IC
    combined = pd.DataFrame({"factor": factor_vals, "fwd_ret": fwd_rets}).dropna()
    if len(combined) < 4:
        return None
    ic, _ = spearmanr(combined["factor"], combined["fwd_ret"])
    return float(ic)


# ── ICIR 月度更新 ──────────────────────────────────────────────────────────────

def update_icir(calc_date: datetime.date, asset_class: str = "equity_sector") -> None:
    """
    每月调用一次。计算所有 active 因子在 calc_date 的 IC，
    更新 FactorICIR 表，并检查生命周期（连续2月 ICIR < 0.15 → inactive）。
    """
    with SessionFactory() as session:
        active_factors = (
            session.query(FactorDefinition)
            .filter(FactorDefinition.active == True,
                    FactorDefinition.asset_class == asset_class)
            .all()
        )
        for fdef in active_factors:
            ic = compute_monthly_ic(fdef.factor_id, calc_date, asset_class=asset_class)
            if ic is None:
                continue

            # 写入 IC
            row = FactorICIR(
                factor_id=fdef.factor_id,
                calc_date=calc_date,
                ic_value=ic,
                asset_class=asset_class,
            )
            session.merge(row)

            # 计算滚动12月 ICIR
            ic_history = (
                session.query(FactorICIR.ic_value)
                .filter(FactorICIR.factor_id == fdef.factor_id,
                        FactorICIR.asset_class == asset_class,
                        FactorICIR.calc_date <= calc_date)
                .order_by(FactorICIR.calc_date.desc())
                .limit(12)
                .all()
            )
            ic_vals = [r[0] for r in ic_history if r[0] is not None]
            icir = float(np.mean(ic_vals) / np.std(ic_vals)) if len(ic_vals) >= 3 else None
            row.icir_12m   = icir
            row.n_assets   = 18  # 动态化后替换

            # 生命周期检查：连续2月 ICIR < 0.15 → inactive
            if icir is not None and icir < 0.15:
                recent_bad = (
                    session.query(FactorICIR)
                    .filter(FactorICIR.factor_id == fdef.factor_id,
                            FactorICIR.icir_12m.isnot(None),
                            FactorICIR.icir_12m < 0.15)
                    .order_by(FactorICIR.calc_date.desc())
                    .limit(2)
                    .count()
                )
                if recent_bad >= 2:
                    fdef.active = False
                    logger.info("FactorMAD: %s 连续2月 ICIR < 0.15 → inactive", fdef.factor_id)

        session.commit()


# ── 复合 FactorMAD 信号 ────────────────────────────────────────────────────────

def get_factor_mad_scores(
    as_of: datetime.date,
    asset_class: str = "equity_sector",
    min_factors: int = 3,
) -> pd.Series | None:
    """
    返回 FactorMAD 复合截面得分（0-100），index=sector。
    若活跃因子数量 < min_factors，返回 None（退化到当前权重方案）。
    """
    with SessionFactory() as session:
        active_factors = (
            session.query(FactorDefinition)
            .filter(FactorDefinition.active == True,
                    FactorDefinition.asset_class == asset_class)
            .all()
        )
    if len(active_factors) < min_factors:
        return None

    # 按 ICIR 加权（ICIR 越高权重越大）
    from engine.universe_manager import get_active_universe
    sector_etf = get_active_universe(asset_classes=[asset_class])
    tickers = list(sector_etf.values())

    start = as_of - datetime.timedelta(days=280)
    import yfinance as yf
    try:
        dl = yf.download(tickers, start=str(start), end=str(as_of + datetime.timedelta(days=2)),
                         progress=False, auto_adjust=True)
        prices = dl["Close"] if "Close" in dl else dl
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [c[0] for c in prices.columns]
        prices.index = pd.to_datetime(prices.index).normalize()
    except Exception:
        return None

    ticker_to_sector = {v: k for k, v in sector_etf.items()}
    factor_scores: list[pd.Series] = []
    factor_weights: list[float] = []

    with SessionFactory() as session:
        for fdef in active_factors:
            fn = FACTOR_REGISTRY.get(fdef.factor_id)
            if fn is None:
                continue
            vals = fn(prices).rename(index=ticker_to_sector)
            if vals.empty or vals.isna().all():
                continue
            # 归一化到 0-100（截面排名）
            ranked = vals.rank(pct=True) * 100
            # 取最近 ICIR 作为权重
            icir_row = (
                session.query(FactorICIR.icir_12m)
                .filter(FactorICIR.factor_id == fdef.factor_id,
                        FactorICIR.calc_date <= as_of,
                        FactorICIR.icir_12m.isnot(None))
                .order_by(FactorICIR.calc_date.desc())
                .first()
            )
            w = max(float(icir_row[0]), 0.0) if icir_row else 0.1
            factor_scores.append(ranked)
            factor_weights.append(w)

    if not factor_scores:
        return None

    total_w = sum(factor_weights) or 1.0
    composite = sum(s * w for s, w in zip(factor_scores, factor_weights)) / total_w
    return composite.round(1)
```

---

### B-3 集成到 `engine/signal.py` 的 `compute_composite_scores()`

```python
# 在 compute_composite_scores() 末尾，替换 regime_score 权重部分：

from engine.factor_mad import get_factor_mad_scores

factor_mad = get_factor_mad_scores(as_of, asset_class="equity_sector")

if factor_mad is not None:
    # FactorMAD 活跃：TSMOM 50% + Sharpe 30% + FactorMAD 20%
    base["factor_mad_score"] = factor_mad.reindex(base.index).fillna(50.0)
    base["composite_score"] = (
        0.50 * base["tsmom_norm"]       +
        0.30 * base["sharpe_norm"]      +
        0.20 * base["factor_mad_score"]
    ).round(1)
else:
    # 退化：TSMOM 50% + Sharpe 30% + Regime 20%（当前方案）
    base["composite_score"] = (
        0.50 * base["tsmom_norm"]   +
        0.30 * base["sharpe_norm"]  +
        0.20 * base["regime_score"]
    ).round(1)
```

**返回列变化**：新增 `factor_mad_score`（可能为 NaN，当退化时）。消费方（`ui/tabs.py` 等）用 `.get("factor_mad_score")` 读取，不存在时不展示。

---

### B-4 Admin UI 展示

在 `pages/admin.py`（或现有 Factor Dashboard）新增：
- FactorMAD 状态表格：factor_id / active / 最新 ICIR / 最近 IC / 加入日期
- "运行 ICIR 更新" 按钮（手动触发 `update_icir(today)`）
- 活跃因子数量 metric（< 3 时显示"退化模式"警告）
- 候选因子审核面板：展示 MI 扫描结果 + 符号回归审计报告 + 三态裁决按钮

---

### B-4.5 Layer 3：符号回归结构审计（`engine/factor_mad.py` 新增函数）

**定位**：在候选因子通过测试集验证（ICIR ≥ 0.3）后、提交 Supervisor 人工审批前，执行一次独立的数学结构分析。**它不是通过/否决的二值门控**，是提供增量信息的侦探报告。

**依赖**：`gplearn`（`pip install gplearn`）。PySR 更强大但需要 Julia 环境，gplearn 对本项目短序列数据更实用。

**实现**（追加到 `factor_mad.py`）：

```python
def audit_factor_structure(
    candidate_fn: callable,
    candidate_description: str,
    prices: pd.DataFrame,
    train_end: datetime.date,
    n_cross_sections: int = 24,
    forward_return_days: int = 22,
) -> dict:
    """
    Layer 3 符号回归审计。
    
    尝试用基准因子（TSMOM类）的原始值通过符号回归拟合候选因子值序列。
    目标：检验候选因子的数学结构是否与其宣称的金融逻辑一致。
    
    返回：
    {
        "signal_type": "positive" | "neutral" | "danger",
        "best_formula": str,          # 符号回归发现的最优公式
        "r2_train": float,            # 拟合优度
        "consistency_note": str,      # 人工解读提示
        "raw_report": str,            # 完整报告文本（写入 DiscoveredFactor.audit_report）
    }
    
    ⚠️ 限制：
    - 当前 18 ETF × 24截面 = ~432 样本；Phase 2 扩展到 33 ETF 后约 ~792 样本，效果显著改善
    - 拟合 R² 低（如 < 0.1）不代表因子无效，Alpha 本身就是噪音中的弱信号
    - 公式与宣称逻辑的"一致性"判断依赖 candidate_description 关键词匹配，存在误判
    """
    try:
        from gplearn.genetic import SymbolicRegressor
    except ImportError:
        return {
            "signal_type": "neutral",
            "best_formula": "N/A",
            "r2_train": None,
            "consistency_note": "gplearn 未安装，跳过符号回归审计",
            "raw_report": "gplearn not available",
        }

    prices.index = pd.to_datetime(prices.index).normalize()
    end_idx = prices.index[prices.index <= pd.Timestamp(train_end)]

    # 收集训练样本
    X_rows, y_vals = [], []
    baseline_fns = [FACTOR_REGISTRY[fid] for fid in _BASELINE_FACTOR_IDS
                    if fid in FACTOR_REGISTRY]

    for i in range(n_cross_sections):
        t_ref_i = end_idx[-(i * forward_return_days + 1 + forward_return_days)]
        pre_prices = prices[prices.index <= t_ref_i]
        if len(pre_prices) < 50:
            continue
        cand_vals = candidate_fn(pre_prices)
        if cand_vals.empty:
            continue
        row_features = []
        for bfn in baseline_fns:
            bvals = bfn(pre_prices).reindex(cand_vals.index).fillna(0.0)
            row_features.append(bvals.values)
        if not row_features:
            continue
        X_block = np.stack(row_features, axis=1)
        X_rows.append(X_block)
        y_vals.append(cand_vals.values)

    if not X_rows:
        return {"signal_type": "neutral", "best_formula": "无法收集训练数据",
                "r2_train": None, "consistency_note": "", "raw_report": ""}

    X = np.vstack(X_rows)
    y = np.concatenate(y_vals)
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid], y[valid]
    if len(y) < 30:
        return {"signal_type": "neutral", "best_formula": "样本不足（<30）",
                "r2_train": None, "consistency_note": "无法进行符号回归", "raw_report": ""}

    sr = SymbolicRegressor(
        population_size=500, generations=20, stopping_criteria=0.01,
        p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
        p_point_mutation=0.1, max_samples=0.9, verbose=0,
        parsimony_coefficient=0.01, random_state=42, n_jobs=1,
    )
    try:
        sr.fit(X, y)
    except Exception as e:
        return {"signal_type": "neutral", "best_formula": f"拟合失败: {e}",
                "r2_train": None, "consistency_note": "", "raw_report": str(e)}

    formula_str = str(sr._program)
    r2 = float(sr.score(X, y))
    feature_names = [fid for fid in _BASELINE_FACTOR_IDS if fid in FACTOR_REGISTRY]

    # 将 X0/X1/... 替换为实际因子名
    readable_formula = formula_str
    for i, fname in enumerate(feature_names):
        readable_formula = readable_formula.replace(f"X{i}", fname)

    # 信号分类
    desc_lower = candidate_description.lower()
    if r2 > 0.3:
        # 找到可解释公式：检查与宣称逻辑是否一致
        # 简单关键词一致性检查（需人工最终判断）
        claimed_keywords = _extract_logic_keywords(desc_lower)
        formula_keywords = _extract_logic_keywords(readable_formula.lower())
        overlap = claimed_keywords & formula_keywords
        if overlap:
            signal_type = "positive"
            note = f"符号回归 R²={r2:.3f}，公式与宣称逻辑关键词重叠: {overlap}。可信度独立支持。"
        else:
            signal_type = "danger"
            note = (f"符号回归 R²={r2:.3f}，但公式关键词与宣称逻辑无重叠。"
                    f"宣称: {claimed_keywords}，公式含: {formula_keywords}。"
                    f"因子有效性来源可能与逻辑不符，建议 Supervisor 重点审查。")
    else:
        signal_type = "neutral"
        note = (f"符号回归 R²={r2:.3f}，无法用简单公式拟合候选因子。"
                f"这可能说明该因子捕捉到了超出简单结构的规律（Alpha 典型特征），"
                f"不构成否决理由。")

    raw_report = (
        f"=== 符号回归审计报告 ===\n"
        f"候选因子描述: {candidate_description}\n"
        f"最优公式: {readable_formula}\n"
        f"训练集 R²: {r2:.4f}\n"
        f"信号类型: {signal_type}\n"
        f"解读: {note}\n"
    )

    return {
        "signal_type": signal_type,
        "best_formula": readable_formula,
        "r2_train": round(r2, 4),
        "consistency_note": note,
        "raw_report": raw_report,
    }


def _extract_logic_keywords(text: str) -> set[str]:
    """从描述文本中提取金融逻辑关键词（用于一致性粗检）。"""
    keywords = {
        "momentum", "reversal", "volatility", "vol", "trend", "sma", "moving average",
        "动量", "反转", "波动率", "趋势", "均线", "动能", "carry", "value", "quality",
    }
    return {kw for kw in keywords if kw in text}
```

**写入 `DiscoveredFactor`**（在 Layer 3 完成后追加字段）：

在 `master_backlog.md` P2-13 `DiscoveredFactor` 表新增：
```python
audit_report      = Column(Text, nullable=True)    # Layer 3 符号回归报告全文
audit_signal_type = Column(String(10), nullable=True)  # "positive" / "neutral" / "danger"
```

**状态更新**：`DiscoveredFactor.status` 从三态扩展为四态：
- `pending` → 刚通过测试集验证，等待人工审批
- `active` → Supervisor 批准，纳入 composite_score，权重上限 10%
- `rejected` → Supervisor 驳回，保留完整记录
- `pending_further_review` → Supervisor 要求补充验证（延长测试周期 / 要求 Proposer 补充逻辑解释）

---

## BP-C｜P2-14 双变量 MSM（加入信用利差）

### 背景与风险

**为什么需要蓝图**：`engine/regime.py` 的 `_fit_and_filter()` 当前接受单变量 `spread_series`（10Y-2Y），`MarkovRegression` 以此为唯一 `endog`。双变量切换为 `endog = DataFrame(2列)`，statsmodels `MarkovRegression` 的多变量支持有版本差异，需要精确的 API 调用方式。同时，`get_regime_on()` 的返回接口（`RegimeResult`）保持不变，确保所有下游消费者零改动。

**改动边界**：仅修改 `engine/regime.py`。`RegimeResult` 接口、`get_regime_on()` 签名、`filtered_marginal_probabilities` 原则**全部保持不变**。

---

### C-1 信用利差数据获取

```python
# engine/regime.py — 新增函数

def _get_monthly_credit_spread(
    train_end: datetime.date,
    n_months: int = 120,
) -> pd.Series:
    """
    获取 IG 信用利差月度序列（ICE BofA BBB Corporate OAS，FRED: BAMLC0A4CBBB）。
    
    数据说明：
    - FRED series BAMLC0A4CBBB：BBB 级公司债 OAS（基点）
    - 发布频率：日度；月度化用月末值
    - 可用起始：1997-01（覆盖 10Y-2Y 利差历史）
    
    如果 FRED 不可用，回退到 HYG-LQD 价差作为代理（yfinance）。
    """
    start_str = (train_end - datetime.timedelta(days=n_months * 31 + 60)).strftime("%Y-%m-%d")
    end_str   = train_end.strftime("%Y-%m-%d")

    try:
        series = _fetch_fred("BAMLC0A4CBBB", start_str, end_str)
        if series.empty:
            raise ValueError("FRED credit spread empty")
        # 月度化：取月末值
        monthly = series.resample("ME").last().dropna()
        return monthly
    except Exception:
        logger.warning("FRED 信用利差获取失败，使用 HYG-LQD 代理")
        return _credit_spread_proxy(start_str, end_str)


def _credit_spread_proxy(start: str, end: str) -> pd.Series:
    """
    备用：用 (LQD - IEF) 收益率差作为信用利差代理。
    注意：这是价格代理，非 OAS，不可与 FRED 序列混用。
    仅在 FRED 不可用时启用。
    """
    import yfinance as yf
    try:
        dl = yf.download(["LQD", "IEF"], start=start, end=end,
                         progress=False, auto_adjust=True)
        prices = dl["Close"] if "Close" in dl else dl
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [c[0] for c in prices.columns]
        lqd_ret = prices["LQD"].pct_change()
        ief_ret = prices["IEF"].pct_change()
        proxy = (lqd_ret - ief_ret).resample("ME").mean().dropna()
        return proxy * 100   # 单位对齐（bps 量级）
    except Exception:
        return pd.Series(dtype=float)
```

---

### C-2 修改 `_fit_and_filter()` 支持双变量

**statsmodels `MarkovRegression` 多变量方法**：`endog` 传入 `pd.DataFrame`（列=变量），`MarkovRegression` 对每列独立建模 switching mean/variance，转移矩阵共享。

```python
def _fit_and_filter(
    spread_series: pd.Series,
    credit_series: pd.Series | None = None,   # 新增参数，None = 单变量退化
) -> tuple[pd.Series, int] | None:
    """
    双变量 MSM（当 credit_series 不为 None 时）。
    单变量行为完全保持。
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    if credit_series is not None and not credit_series.empty:
        # 对齐索引（月度，取交集）
        combined = pd.DataFrame({
            "yield_spread":  spread_series,
            "credit_spread": credit_series,
        }).dropna()
        endog = combined
        # 注意：statsmodels >= 0.14 支持 DataFrame endog
        # 制度识别：yield_spread 均值最高 = risk-on（保持原有约定）
        regime_col = "yield_spread"
    else:
        endog = spread_series.dropna()
        regime_col = None   # 单变量，原有逻辑

    if len(endog) < _MIN_OBS_FOR_MSM:
        return None

    try:
        k = _select_k_by_bic(
            endog if regime_col is None else endog["yield_spread"],
            k_range=(2, 3),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovRegression(
                endog,
                k_regimes=k,
                trend="c",
                switching_variance=True,
            )
            result = model.fit(disp=False, maxiter=300)

        # 制度识别：yield_spread 分量均值最高 = risk-on
        if regime_col is not None:
            # 双变量时，params 命名为 "yield_spread.const[i]"（statsmodels约定）
            try:
                means = result.params[[f"yield_spread.const[{i}]" for i in range(k)]]
            except KeyError:
                # 回退到单变量命名（版本兼容）
                means = result.params[[f"const[{i}]" for i in range(k)]]
        else:
            means = result.params[[f"const[{i}]" for i in range(k)]]

        risk_on_idx = int(np.argmax(means))
        filtered    = result.filtered_marginal_probabilities
        p_risk_on_series = filtered.iloc[:, risk_on_idx]
        return p_risk_on_series, risk_on_idx

    except Exception as exc:
        logger.warning("MSM fitting failed (bivariate): %s", exc, exc_info=True)
        # 回退到单变量
        if credit_series is not None:
            logger.info("双变量 MSM 失败，回退到单变量")
            return _fit_and_filter(spread_series, credit_series=None)
        return None
```

---

### C-3 修改 `get_regime_on()` 调用链

```python
# engine/regime.py — get_regime_on() 内

# 原有：
spread_monthly = _get_monthly_yield_spread(train_end, n_train_months)
# 新增：
credit_monthly = _get_monthly_credit_spread(train_end, n_train_months)

# 原有：
msm_result = _fit_and_filter(spread_monthly)
# 新增：
msm_result = _fit_and_filter(spread_monthly, credit_series=credit_monthly)
```

`RegimeResult` 数据类、`get_regime_on()` 返回值、`_PROB_THRESHOLD`、`filtered_marginal_probabilities` 原则**一律不变**。

---

### C-4 statsmodels 版本兼容性注意事项

| 版本 | 多变量 endog 行为 |
|------|----------------|
| < 0.13 | 不支持 DataFrame endog，会报错 |
| 0.13–0.13.5 | 支持但 params 命名为 `const[i]`（与单变量相同） |
| ≥ 0.14 | 支持，params 命名为 `{col}.const[i]` |

**实施前验证**：`import statsmodels; print(statsmodels.__version__)` 确认版本，再决定参数名解析方式。上述代码已含 `try/except KeyError` 兼容两种命名。

---

### C-5 回归测试清单

1. 单变量模式（`credit_series=None`）：运行 `get_regime_on(date.today())`，对比改动前后 `p_risk_on` 差异 < 0.01
2. 双变量模式：FRED 可用时 `credit_series` 非空，确认 `p_risk_on` 在 [0, 1] 内
3. FRED 不可用：确认回退到 `_credit_spread_proxy`，再回退到单变量
4. `engine/portfolio.py` / `engine/signal.py` 的 `get_regime_on()` 调用：输出格式不变（`RegimeResult` 相同）

---

## BP-D｜P2-17 新闻来源升级

### 背景与风险

**为什么需要蓝图**：三层数据源（FMP/Finnhub → GNews/NewsAPI → yfinance.news 备用）涉及外部 API Key 管理、失败降级逻辑、情绪分数存储。最大跑偏风险：① 新闻内容直接注入 prompt 可能违反 P0-4 的注入规则（需确认情绪分数属于"原始数值"而非"方向性结论"）；② 来源权重与现有 `NewsRoutingWeight` 表的集成方式；③ API Key 通过 `engine/key_manager.py` 统一管理，不能硬编码。

**P0-4 合规确认**：情绪分数（sentiment_score = -1.0 到 +1.0）属于**原始数值**，允许注入 Red Team prompt（同 `p_risk_on`、`ann_vol`）。但新闻**摘要文本**注入现有 prompt 不受 P0-4 限制（原已存在）。新增内容仅是为现有新闻摘要加上来源可信度过滤和时效性权重。

---

### D-1 新建 `engine/news_fetcher.py`

```python
"""
engine/news_fetcher.py — 三层新闻数据源
==========================================
Layer 1: FMP（Financial Modeling Prep）+ Finnhub — 付费，结构化情绪
Layer 2: GNews / NewsAPI — 免费层，覆盖广
Layer 3: yfinance.news — 终极备用，零 API Key

注意：Finnhub 免费层情绪基于 VADER/TextBlob，非 LLM 级别，
      用于辅助过滤而非独立决策依据。
"""
from __future__ import annotations
import datetime
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    title:          str
    summary:        str
    published_at:   datetime.datetime
    source:         str           # "fmp" / "finnhub" / "gnews" / "newsapi" / "yfinance"
    source_tier:    int           # 1 / 2 / 3
    sentiment_score: float | None = None   # -1.0 到 +1.0，None = 未评分
    url:            str = ""
    relevance_score: float = 1.0  # 来源 API 提供的相关度（0-1）


def _get_api_key(service: str) -> str | None:
    """通过 engine/key_manager.py 获取 API Key，不返回则 None。"""
    try:
        from engine.key_manager import get_key
        return get_key(service)
    except Exception:
        return None


def fetch_finnhub_news(ticker: str, days: int = 3) -> list[NewsItem]:
    """Finnhub /news 端点，含情绪分数。免费层限速 60 次/分钟。"""
    api_key = _get_api_key("finnhub")
    if not api_key:
        return []
    import requests
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": str(start), "to": str(end), "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        items = []
        for art in resp.json()[:10]:   # 最多10条
            items.append(NewsItem(
                title=art.get("headline", ""),
                summary=art.get("summary", "")[:500],
                published_at=datetime.datetime.fromtimestamp(art.get("datetime", 0)),
                source="finnhub",
                source_tier=1,
                sentiment_score=art.get("sentiment", {}).get("companyNewsScore"),
                url=art.get("url", ""),
            ))
        return items
    except Exception as e:
        logger.warning("Finnhub news error for %s: %s", ticker, e)
        return []


def fetch_gnews(query: str, days: int = 3, max_items: int = 5) -> list[NewsItem]:
    """GNews API（免费层：100 次/天）。"""
    api_key = _get_api_key("gnews")
    if not api_key:
        return []
    import requests
    try:
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params={"q": query, "lang": "en", "max": max_items,
                    "from": (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z"),
                    "apikey": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        items = []
        for art in resp.json().get("articles", []):
            items.append(NewsItem(
                title=art.get("title", ""),
                summary=art.get("description", "")[:500],
                published_at=datetime.datetime.fromisoformat(
                    art.get("publishedAt", "2000-01-01T00:00:00Z").replace("Z", "+00:00")
                ).replace(tzinfo=None),
                source="gnews",
                source_tier=2,
                url=art.get("url", ""),
            ))
        return items
    except Exception as e:
        logger.warning("GNews error for %s: %s", query, e)
        return []


def fetch_yfinance_news(ticker: str, max_items: int = 5) -> list[NewsItem]:
    """yfinance.Ticker.news — 零 API Key，终极备用。"""
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).news or []
        items = []
        for art in raw[:max_items]:
            content = art.get("content", {})
            pub_str = content.get("pubDate") or art.get("providerPublishTime", "")
            try:
                pub_dt = (datetime.datetime.fromisoformat(pub_str.replace("Z", ""))
                          if isinstance(pub_str, str) else
                          datetime.datetime.fromtimestamp(pub_str))
            except Exception:
                pub_dt = datetime.datetime.utcnow()
            items.append(NewsItem(
                title=content.get("title", art.get("title", "")),
                summary=content.get("summary", "")[:500],
                published_at=pub_dt,
                source="yfinance",
                source_tier=3,
            ))
        return items
    except Exception as e:
        logger.warning("yfinance news error for %s: %s", ticker, e)
        return []


def fetch_sector_news(
    sector: str,
    ticker: str,
    days: int = 3,
    max_total: int = 8,
) -> list[NewsItem]:
    """
    三层降级：Layer 1 → Layer 2 → Layer 3。
    返回去重、时效性排序的新闻列表。
    """
    items: list[NewsItem] = []

    # Layer 1
    items.extend(fetch_finnhub_news(ticker, days=days))

    # Layer 2（如果 Layer 1 不足）
    if len(items) < 3:
        query = f"{sector} ETF {ticker}"
        items.extend(fetch_gnews(query, days=days, max_items=5))

    # Layer 3（如果前两层都失败）
    if not items:
        items.extend(fetch_yfinance_news(ticker, max_items=max_items))

    # 去重（title 前30字符）
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for it in items:
        key = it.title[:30]
        if key not in seen:
            seen.add(key)
            unique.append(it)

    # 时效性衰减排序（新闻越新排越前）
    unique.sort(key=lambda x: x.published_at, reverse=True)
    return unique[:max_total]
```

---

### D-2 时效性加权摘要函数

```python
# engine/news_fetcher.py — 追加

def build_weighted_news_summary(
    items: list[NewsItem],
    max_chars: int = 1200,
    decay_halflife_days: float = 1.5,
) -> str:
    """
    将 NewsItem 列表合并为带时效权重的摘要字符串，供 prompt 注入。
    
    时效衰减：weight = exp(-ln(2) / halflife × days_old)
    来源权重：tier 1 = 1.0, tier 2 = 0.7, tier 3 = 0.4
    情绪标注：当 sentiment_score 不为 None 时，在标题后附加 [情绪: ±x.xx]
    """
    import math
    now = datetime.datetime.utcnow()
    TIER_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.4}
    DECAY_LAMBDA = math.log(2) / decay_halflife_days

    scored: list[tuple[float, NewsItem]] = []
    for item in items:
        days_old = max(0.0, (now - item.published_at).total_seconds() / 86400)
        time_w   = math.exp(-DECAY_LAMBDA * days_old)
        tier_w   = TIER_WEIGHTS.get(item.source_tier, 0.4)
        score    = time_w * tier_w * (item.relevance_score or 1.0)
        scored.append((score, item))

    scored.sort(reverse=True)

    lines: list[str] = []
    total = 0
    for score, item in scored:
        sent_str = (f" [情绪: {item.sentiment_score:+.2f}]"
                    if item.sentiment_score is not None else "")
        age_h = (now - item.published_at).total_seconds() / 3600
        line = f"[{item.source.upper()} {age_h:.0f}h前] {item.title}{sent_str}\n{item.summary}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)

    return "\n\n".join(lines) if lines else "（无可用新闻）"
```

---

### D-3 API Key 注册

在 `pages/key_manager.py` 或 `engine/key_manager.py` 的 Key 列表中新增：

| service 名称 | 备注 |
|-------------|------|
| `"finnhub"` | Finnhub API Token（PRE-9，免费层可用） |
| `"gnews"` | GNews API Key（PRE-10，免费层 100次/天） |
| `"fmp"` | Financial Modeling Prep（可选，付费） |
| `"newsapi"` | NewsAPI.org（可选，免费层 100次/天） |

---

### D-4 集成到现有 prompt 构建

**改动位置**：`engine/agent.py` 或 `ui/tabs.py` 中构建 `news_summary` 的段落。

```python
# 替换现有 news_summary 构建逻辑

from engine.news_fetcher import fetch_sector_news, build_weighted_news_summary
from engine.history import get_active_sector_etf

sector_etf = get_active_sector_etf()
ticker     = sector_etf.get(sector_name, "")

if ticker:
    news_items   = fetch_sector_news(sector_name, ticker, days=3, max_total=8)
    news_summary = build_weighted_news_summary(news_items, max_chars=1200)
else:
    news_summary = "（无 ETF ticker，跳过新闻获取）"
```

**P0-4 合规注意**：`build_weighted_news_summary()` 输出的情绪分数 `[情绪: +0.32]` 是原始数值，合规。禁止在此函数内加入 "建议超配" / "TSMOM 信号" 等方向性文字。

---

### D-5 `NewsRoutingWeight` 表集成

现有 `NewsRoutingWeight` 表（`engine/memory.py`，字段：`sector_name / macro_regime / news_category / weight / sample_count`）用于控制哪些新闻类别被优先引用。

P2-17 后，在 `fetch_sector_news()` 返回结果时，按 `NewsRoutingWeight` 中该 sector × regime 的类别权重对 `items` 重新排序（非必须，可作为 P2-17 增强版）：

```python
# 可选：按已学到的类别权重重新排序
def rerank_by_routing_weight(
    items: list[NewsItem],
    sector: str,
    regime: str,
) -> list[NewsItem]:
    """按 NewsRoutingWeight 表中学到的类别权重重排新闻。"""
    from engine.memory import SessionFactory, NewsRoutingWeight
    with SessionFactory() as session:
        weights = {
            r.news_category: r.weight
            for r in session.query(NewsRoutingWeight)
                .filter_by(sector_name=sector, macro_regime=regime)
                .all()
        }
    if not weights:
        return items
    # 简单：将高权重类别的新闻排在前面（基于关键词匹配）
    HIGH_WEIGHT_THRESHOLD = 0.7
    high_cats = {k for k, v in weights.items() if v >= HIGH_WEIGHT_THRESHOLD}
    priority  = [it for it in items if any(c in it.title for c in high_cats)]
    rest      = [it for it in items if it not in priority]
    return priority + rest
```

---

## 实施顺序与依赖

```
BP-A: P2-11 → P2-12批次A → P2-12批次B
  ↳ P2-11 必须先完成（UniverseETF 表建立）
  ↳ 批次 A 不触发 within-class CSMOM（全是 equity）
  ↳ 批次 B 触发 within-class CSMOM（A-3 同步激活）

BP-B: P2-13 FactorMAD
  ↳ 依赖 BP-A P2-11（get_active_universe）
  ↳ 独立于 BP-C / BP-D，可并行

BP-C: P2-14 双变量 MSM
  ↳ 独立，可最先实施（改动范围最小）
  ↳ 实施前验证 statsmodels 版本

BP-D: P2-17 新闻升级
  ↳ 依赖 PRE-9 / PRE-10（API Key 申请）
  ↳ 独立于 BP-A / BP-B / BP-C，可并行
```

---

## 每项开工前检查清单

### BP-A 开工前
- [ ] `python -c "from engine.universe_manager import init_universe_db; init_universe_db()"` 无报错
- [ ] `get_active_sector_etf()` 返回字典与旧 `SECTOR_ETF` 一致（18 项）
- [ ] `engine/signal.py` 的 `from engine.history import SECTOR_ETF` 直接用法已清查

### BP-B 开工前
- [ ] BP-A P2-11 已完成（`UniverseETF` 表存在）
- [ ] `FactorDefinition` / `FactorICIR` 表已通过 `init_db()` 创建
- [ ] `FACTOR_REGISTRY` 有 ≥ 1 个注册因子可测试

### BP-C 开工前
- [ ] `import statsmodels; print(statsmodels.__version__)` → 记录版本
- [ ] 在 notebook 中测试 `MarkovRegression(DataFrame_2col, k_regimes=2, ...)` 可运行
- [ ] 备份当前 `engine/regime.py`（或 git stash）

### BP-D 开工前
- [ ] PRE-9：Finnhub API Key 已存入 key_manager
- [ ] PRE-10：GNews API Key 已存入 key_manager
- [ ] `fetch_yfinance_news("XLK")` 测试（零 Key，验证 Layer 3 可用）
- [ ] 确认 `engine/key_manager.py` 的 `get_key()` 函数签名
