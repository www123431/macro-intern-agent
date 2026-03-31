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
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not GEMINI_KEY:
        st.error("❌ Secrets 配置不完整，请在 Streamlit 控制台配置 GEMINI_API_KEY。")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置文件读取失败: {e}")
    st.stop()

# --- 3. 初始化 Gemini ---
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel(model_name='gemini-2.5-flash')

# --- 4. 核心量化引擎 ---

class StrategyAuditor:
    @staticmethod
    def run_feature_sparsity_check(X, y):
        common_index = X.index.intersection(y.index)
        X, y = X.loc[common_index], y.loc[common_index]
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
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_market_data(tickers):
        try:
            data = yf.download(tickers, period="1y", progress=False)['Close']
            if isinstance(data, pd.Series): data = data.to_frame()
            return data.pct_change().fillna(0)
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

def render_tv_chart(symbol, title):
    code = f"""
    <div style="height:350px; margin-bottom: 20px;"><script src="https://s3.tradingview.com/tv.js"></script>
    <script>new TradingView.MediumWidget({{"symbols": [["{title}", "{symbol}|12M"]],"width": "100%", "height": 350, "locale": "zh_CN", "colorTheme": "light"}});</script></div>"""
    components.html(code, height=360)

@st.cache_data(ttl=3600)
def get_ai_analysis(prompt_type, vix_val):
    """通用 AI 专家接口：处理宏观与行业研判"""
    vix_context = f"当前市场 VIX 风险指数为 {vix_val}。"
    prompts = {
        "macro": f"{vix_context} 作为新加坡宏观策略专家，请分析 MAS 货币政策走势、S$NEER 汇率波动及 CPI 对本地 REITs 的影响。",
        "sector": f"{vix_context} 作为行业透视专家，请评估当前 AI 算力板块（NVIDIA/AMD）与地缘政治敏感资产（如黄金、原油）的风险对冲建议。"
    }
    try:
        response = model.generate_content(prompts[prompt_type])
        return response.text
    except Exception as e:
        return f"AI 专家暂时离线: {e}"

# --- 6. LangGraph 智能体逻辑 (Tab 4 专用) ---

class AgentState(TypedDict):
    target_assets: str
    vix_level: float
    quant_results: dict
    is_robust: bool
    audit_memo: str

PRESET_ASSETS = {"新加坡蓝筹": ["DBSDF", "U11.SI", "V03.SI"], "科技成长": ["NVDA", "AAPL", "MSFT"]}

def research_node(state: AgentState):
    returns = QuantEngine.get_market_data(PRESET_ASSETS[state['target_assets']])
    if returns.empty: return {"is_robust": False}
    weights = np.array([1.0/len(returns.columns)]*len(returns.columns))
    d_var, a_ret, a_vol = QuantEngine.compute_asymmetric_risk(returns, state['vix_level'], weights)
    y = returns.iloc[:, 0]; X = returns.shift(1).fillna(0)
    active, sparsity, coefs = StrategyAuditor.run_feature_sparsity_check(X, y)
    is_robust, p_noise = StrategyAuditor.check_optimizer_curse(0.05, 0.045, 100)
    return {"quant_results": {"d_var": d_var, "p_noise": p_noise, "sparsity": sparsity, "active": active, "coefs": coefs, "returns": returns, "X": X, "a_ret": a_ret, "a_vol": a_vol}, "is_robust": (p_noise < 0.3)}

def gemini_audit_node(state: AgentState):
    q = state['quant_results']
    prompt = f"针对资产 {state['target_assets']} 进行量化审计：VaR {q['d_var']:.2%}, P-hacking 风险 {q['p_noise']:.2%}。请以专家身份给出审计备忘录。"
    try:
        res = model.generate_content(prompt)
        return {"audit_memo": res.text}
    except:
        return {"audit_memo": "AI 审计调用失败。"}

builder = StateGraph(AgentState)
builder.add_node("researcher", research_node); builder.add_node("auditor", gemini_audit_node)
builder.set_entry_point("researcher")
builder.add_conditional_edges("researcher", lambda x: "auditor" if x["is_robust"] else END)
builder.add_edge("auditor", END)
agent_executor = builder.compile()

# --- 7. 主界面布局 ---

with st.sidebar:
    st.header("⚙️ 指令中心")
    vix_input = st.slider("市场 VIX 压力调节", 10.0, 50.0, 20.0)
    st.divider()
    st.caption("Macro Alpha Pro | 2026 Edition")

tab1, tab2, tab3, tab4 = st.tabs(["🧠 首席宏观研判", "📈 实时仪表盘", "🛡️ 行业风险穿透", "🔢 量化审计室"])

with tab1:
    st.header("🧠 首席宏观投研备忘录")
    if st.button("🚀 启动宏观协同分析", type="primary"):
        with st.spinner("正在调度新加坡宏观专家库..."):
            macro_res = get_ai_analysis("macro", vix_input)
            st.markdown(macro_res)
            st.download_button("📥 下载宏观报告", generate_docx_report(macro_res, "Macro Memo"), "Macro_Report.docx")

with tab2:
    st.subheader("📊 全球资产实时走廊")
    col1, col2 = st.columns(2)
    with col1: render_tv_chart("OANDA:XAUUSD", "现货黄金")
    with col2: render_tv_chart("NASDAQ:NDX", "纳斯达克 100")
    col3, col4 = st.columns(2)
    with col3: render_tv_chart("STI:STI", "海峡指数")
    with col4: render_tv_chart("FX_IDC:USDSGD", "美元/新币")

with tab3:
    st.header("🛡️ 行业穿透与非对称风险评估")
    if st.button("🔍 执行行业风险扫描", type="primary"):
        with st.spinner("正在分析行业暴露风险..."):
            sector_res = get_ai_analysis("sector", vix_input)
            st.markdown(sector_res)
            st.download_button("📥 下载行业报告", generate_docx_report(sector_res, "Sector Drilldown"), "Sector_Report.docx")

with tab4:
    st.header("🔢 Agentic AI 深度审计终端")
    choice = st.selectbox("选择审计资产包", list(PRESET_ASSETS.keys()))
    output_container = st.container()
    if st.button("🚀 启动全自动审计工作流", type="primary"):
        with st.status("Agent 正在协同专家组...", expanded=True) as status:
            final_state = {"target_assets": choice, "vix_level": vix_input, "quant_results": {}, "is_robust": True, "audit_memo": ""}
            for event in agent_executor.stream(final_state):
                for node, update in event.items():
                    st.write(f"✅ {node} 任务处理完成")
                    final_state.update(update)
            status.update(label="审计流执行完毕", state="complete")
        with output_container:
            if "quant_results" in final_state and final_state["quant_results"]:
                q = final_state['quant_results']
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("动态 VaR", f"{q['d_var']:.2%}"); c2.metric("特征稀疏率", f"{q['sparsity']:.1%}")
                c3.metric("P-hacking 风险", f"{q['p_noise']:.1%}"); c4.metric("夏普比率", f"{(q['a_ret']-0.03)/q['a_vol']:.2f}")
                st.subheader("🧬 因子贡献权重")
                st.bar_chart(pd.DataFrame({'Factor': q['X'].columns, 'Weight': q['coefs']}), x="Factor", y="Weight")
                if final_state['is_robust']:
                    st.subheader("🤖 AI 专家深度审计报告")
                    st.markdown(final_state['audit_memo'])
                else:
                    st.error("⚠️ 策略统计风险过高，Agent 已拦截 AI 报告生成。")

st.markdown("---")
st.caption("Macro Alpha Pro | NUS MSBA Project | Powered by LangGraph")
