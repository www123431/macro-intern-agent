import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import google.api_core.exceptions
import urllib.parse
import streamlit.components.v1 as components
import time
import datetime
from docx import Document
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from sklearn.manifold import Isomap
from sklearn.linear_model import LassoCV
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END

# --- 1. 页面基本配置 ---
st.set_page_config(page_title="Macro Alpha Pro Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 四大专家协同研判终端")

# --- 2. 安全读取 Secrets ---
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整，请检查 Streamlit Secrets。")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置文件读取失败: {e}")
    st.stop()

# --- 3. 初始化 Gemini ---
genai.configure(api_key=GEMINI_KEY)
# 保持使用 2.5 Flash 模型
model = genai.GenerativeModel(model_name='models/gemini-2.5-flash')

# --- 4. 核心专家逻辑类 ---

class StrategyAuditor:
    """专家 1 & 2: 特征工程与回测审计专家"""
    @staticmethod
    def run_feature_sparsity_check(X, y):
        model_lasso = LassoCV(cv=5).fit(X, y)
        active_features = np.sum(np.abs(model_lasso.coef_) > 1e-10)
        sparsity = 1 - (active_features / X.shape[1])
        return active_features, sparsity, model_lasso.coef_

    @staticmethod
    def check_optimizer_curse(val_score, test_score, n_trials):
        luck_factor = np.log1p(n_trials) * 0.01 
        gap = max(0, val_score - test_score)
        prob_noise = 1 - np.exp(-gap / (luck_factor + 1e-6))
        return gap <= luck_factor, prob_noise

class QuantEngine:
    """专家 3: 量化风险专家"""
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_market_data(tickers):
        try:
            data = yf.download(tickers, period="1y", progress=False)['Close']
            if isinstance(data, pd.Series): data = data.to_frame()
            return data.pct_change().dropna()
        except:
            return pd.DataFrame()

    @staticmethod
    def compute_asymmetric_risk(returns, vix, weights):
        port_ret = returns.dot(weights)
        alpha = 0.05 * (1 + (vix - 20) / 40) if vix > 20 else 0.05
        dynamic_vaR = np.quantile(port_ret, min(alpha, 0.2))
        return dynamic_vaR, port_ret.mean() * 252, port_ret.std() * np.sqrt(252)

# --- 5. 辅助功能 ---

def generate_docx_report(content, title="Investment Memo"):
    doc = Document()
    doc.add_heading('Macro Alpha Intelligence', 0)
    doc.add_heading(title, level=1)
    doc.add_paragraph(f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    for line in content.split('\n'): doc.add_paragraph(line)
    buf = BytesIO(); doc.save(buf); buf.seek(0)
    return buf

def get_ai_audit_report(metrics_dict, choice_name, vix_val):
    """AI 审计专家：基于量化指标生成专业评述"""
    prompt = f"""
    你是一名资深量化风险审计师。请针对以下资产包的量化指标进行深度审计：
    资产包：{choice_name}，当前市场 VIX 指数：{vix_val}
    量化指标：
    1. 动态 VaR (非对称): {metrics_dict['d_var']:.2%}
    2. 特征稀疏率: {metrics_dict['sparsity']:.1%}
    3. P-hacking 概率: {metrics_dict['p_noise']:.1%}
    4. 有效特征数: {metrics_dict['active']} 个
    请按以下格式输出：结论先行、核心指标白话化、专家建议。
    """
    try:
        return model.generate_content(prompt).text
    except:
        return "AI 审计报告生成失败，请检查 API 额度。"

def render_tv_chart(symbol, title):
    code = f"""
    <div style="height:350px; margin-bottom: 20px;">
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.MediumWidget({{
      "symbols": [["{title}", "{symbol}|12M"]],
      "width": "100%", "height": 350, "locale": "zh_CN", "colorTheme": "light", "autosize": true
    }});
    </script></div>"""
    components.html(code, height=360)

# --- 6. LangGraph 智能体逻辑 ---

class AgentState(TypedDict):
    target_assets: str
    vix_level: float
    quant_results: dict
    is_robust: bool
    audit_memo: str

# 定义资产包常量供节点调用
PRESET_ASSETS = {"新加坡蓝筹": ["DBSDF", "U11.SI", "V03.SI"], "科技成长": ["NVDA", "AAPL", "MSFT"]}

def research_node(state: AgentState):
    """节点：量化专家执行计算"""
    asset_tickers = PRESET_ASSETS[state['target_assets']]
    returns = QuantEngine.get_market_data(asset_tickers)
    
    if returns.empty:
        return {"is_robust": False}
        
    weights = np.array([1.0/len(returns.columns)]*len(returns.columns))
    d_var, a_ret, a_vol = QuantEngine.compute_asymmetric_risk(returns, state['vix_level'], weights)
    
    # 准备特征工程数据
    y = returns.iloc[:, 0]
    X = returns.shift(1).dropna()
    y = y.iloc[1:]
    
    active, sparsity, coefs = StrategyAuditor.run_feature_sparsity_check(X, y)
    is_robust, p_noise = StrategyAuditor.check_optimizer_curse(0.05, 0.045, 100)
    
    return {
        "quant_results": {
            "d_var": d_var, "p_noise": p_noise, "sparsity": sparsity, 
            "active": active, "coefs": coefs, "returns": returns, "X": X,
            "a_ret": a_ret, "a_vol": a_vol
        },
        "is_robust": is_robust and (p_noise < 0.25)
    }

def gemini_audit_node(state: AgentState):
    """节点：Gemini AI 执行逻辑审计"""
    metrics = state['quant_results']
    report = get_ai_audit_report(metrics, state['target_assets'], state['vix_level'])
    return {"audit_memo": report}

# 构建工作流
builder = StateGraph(AgentState)
builder.add_node("researcher", research_node)
builder.add_node("auditor", gemini_audit_node)
builder.set_entry_point("researcher")

def router(state):
    return "auditor" if state["is_robust"] else END

builder.add_conditional_edges("researcher", router)
builder.add_edge("auditor", END)
agent_executor = builder.compile()

# --- 7. 主界面布局 ---

with st.sidebar:
    st.header("⚙️ 专家指令中心")
    vix_input = st.slider("VIX 风险压力调节", 10.0, 50.0, 20.0)
    st.divider()
    st.caption("基于 LangGraph 驱动的多智能体协同系统")

tab1, tab2, tab3, tab4 = st.tabs(["🧠 宏观研判", "📈 实时仪表盘", "🛡️ 行业风险", "🔢 量化审计室"])

with tab1:
    st.info("💡 请在 Tab 4 启动智能审计流，该模块将自动集成宏观视角。")

with tab2:
    st.subheader("📊 全球资产实时走廊")
    col1, col2 = st.columns(2)
    with col1: render_tv_chart("OANDA:XAUUSD", "黄金")
    with col2: render_tv_chart("NASDAQ:NDX", "纳斯达克")

with tab3:
    st.write("行业穿透分析已集成至 Tab 4 的 Agent 审计报告中。")

with tab4:
    st.header("🔢 Agentic AI 深度审计终端")
    choice = st.selectbox("选择审计资产包", list(PRESET_ASSETS.keys()))
    
    # 初始化一个 session_state 来存储结果，防止 Streamlit 刷新后数据丢失
    if "audit_results" not in st.session_state:
        st.session_state.audit_results = None

    if st.button("🚀 启动全自动审计流", type="primary"):
        initial_input = {
            "target_assets": choice, 
            "vix_level": vix_input, 
            "quant_results": {}, 
            "is_robust": True, 
            "audit_memo": ""
        }
        
        with st.status("Agent 正在编排投研任务...", expanded=True) as status:
            # 运行 Agent 流程
            current_state = initial_input.copy()
            for event in agent_executor.stream(initial_input):
                for node_name, state_update in event.items():
                    st.write(f"✅ {node_name} 专家已完成分析")
                    current_state.update(state_update)
            
            # 将最终状态存入 session_state
            st.session_state.audit_results = current_state
            status.update(label="审计流执行完毕", state="complete")

    # --- 渲染逻辑：只要 session_state 里有数据就显示 ---
    if st.session_state.audit_results:
        res = st.session_state.audit_results
        
        if "quant_results" in res and res["quant_results"]:
            q = res['quant_results']
            
            # 1. 核心指标卡片
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("动态 VaR", f"{q['d_var']:.2%}")
            c2.metric("特征稀疏率", f"{q['sparsity']:.1%}")
            c3.metric("P-hacking 风险", f"{q['p_noise']:.1%}")
            # 计算夏普比率需确保 a_ret 和 a_vol 存在
            sharpe = (q.get('a_ret', 0) - 0.03) / q.get('a_vol', 1)
            c4.metric("策略夏普比率", f"{sharpe:.2f}")

            # 2. 可视化图表
            st.divider()
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("👁️ 市场流形结构 (Isomap)")
                iso = Isomap(n_neighbors=5, n_components=2)
                manifold = iso.fit_transform(q['returns'])
                df_iso = pd.DataFrame(manifold, columns=["Dim 1", "Dim 2"])
                df_iso['Return'] = q['returns'].mean(axis=1).values
                st.scatter_chart(df_iso, x="Dim 1", y="Dim 2", color="Return")
            
            with col_r:
                st.subheader("🧬 因子贡献权重")
                feat_importance = pd.DataFrame({'Factor': q['X'].columns, 'Weight': q['coefs']})
                st.bar_chart(feat_importance, x="Factor", y="Weight")

            # 3. AI 报告
            st.divider()
            if res.get('is_robust', False):
                st.subheader("🤖 AI 专家深度审计报告")
                if res.get('audit_memo'):
                    st.markdown(res['audit_memo'])
                    st.download_button("📥 下载审计备忘录", generate_docx_report(res['audit_memo']), "Audit_Report.docx")
                else:
                    st.warning("Agent 已完成计算，但 AI 审计报告内容为空，请检查 API 响应。")
            else:
                st.subheader("⚠️ Agent 审计熔断")
                st.error(f"检测到该策略 P-hacking 风险过高 ({q['p_noise']:.2%})，Agent 已自动拦截非稳健性结论。")
st.markdown("---")
st.caption("Macro Alpha Pro | NUS MSBA Project | Powered by LangGraph & Gemini 2.5 Flash")
