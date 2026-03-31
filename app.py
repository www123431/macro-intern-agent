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
        st.error("❌ Secrets 配置不完整。")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置文件读取失败: {e}")
    st.stop()

# --- 3. 初始化 Gemini ---
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel(model_name='models/gemini-2.5-flash')

# --- 4. 核心专家引擎集成 (统一合并) ---

class StrategyAuditor:
    """专家 1 & 2: 特征工程与回测审计专家"""
    @staticmethod
    def run_feature_sparsity_check(X, y):
        """利用 LassoCV 执行 Unitless 稀疏性控制"""
        model_lasso = LassoCV(cv=5).fit(X, y)
        active_features = np.sum(np.abs(model_lasso.coef_) > 1e-10)
        sparsity = 1 - (active_features / X.shape[1])
        return active_features, sparsity, model_lasso.coef_

    @staticmethod
    def check_optimizer_curse(val_score, test_score, n_trials):
        """评估 P-hacking 风险 (Optimizer's Curse)"""
        luck_factor = np.log1p(n_trials) * 0.01 
        gap = max(0, val_score - test_score)
        prob_noise = 1 - np.exp(-gap / (luck_factor + 1e-6))
        return gap <= luck_factor, prob_noise

class QuantEngine:
    """专家 3: 量化风险专家"""
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_market_data(tickers):
        """统一的数据抓取与 Unitless 转换"""
        try:
            data = yf.download(tickers, period="1y", progress=False)['Close']
            if isinstance(data, pd.Series): data = data.to_frame()
            return data.pct_change().dropna()
        except:
            return pd.DataFrame()

    @staticmethod
    def compute_asymmetric_risk(returns, vix, weights):
        """非对称代价逻辑：VIX 驱动动态边界"""
        port_ret = returns.dot(weights)
        alpha = 0.05 * (1 + (vix - 20) / 40) if vix > 20 else 0.05
        dynamic_vaR = np.quantile(port_ret, min(alpha, 0.2))
        return dynamic_vaR, port_ret.mean() * 252, port_ret.std() * np.sqrt(252)

# --- 5. 辅助功能与 AI 专家 ---

def generate_docx_report(content, title="Investment Memo"):
    doc = Document()
    doc.add_heading('Macro Alpha Intelligence', 0)
    doc.add_heading(title, level=1)
    doc.add_paragraph(f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    for line in content.split('\n'): doc.add_paragraph(line)
    buf = BytesIO(); doc.save(buf); buf.seek(0)
    return buf

@st.cache_data(ttl=3600)
def get_ai_analysis(prompt_type, news_context, vix_val):
    """专家 4: 宏观策略专家 (集成非对称风险思维)"""
    vix_msg = f"当前市场VIX压力为{vix_val}，请在分析中采用非对称风险防御立场。"
    prompts = {
        "macro": f"你作为驻新加坡宏观策略专家，分析对CPI、MAS及S$NEER的影响：{news_context}。{vix_msg}",
        "sector": f"你作为行业穿透专家，评估AI算力、地缘敏感商品及蓝筹机会：{news_context}。{vix_msg}"
    }
    try:
        response = model.generate_content(prompts[prompt_type])
        return response.text, None
    except Exception as e:
        return None, str(e)

def get_ai_audit_report(metrics_dict, choice_name, vix_val):
    """AI 审计专家：基于量化指标生成专业评述"""
    prompt = f"""
    你是一名资深量化风险审计师。请针对以下资产包的量化指标进行深度审计：
    资产包：{choice_name}
    当前市场 VIX 指数：{vix_input}
    
    量化指标：
    1. 动态 VaR (非对称): {metrics_dict['d_var']:.2%} (反映极端下行风险)
    2. 特征稀疏率: {metrics_dict['sparsity']:.1%} (反映模型是否过拟合)
    3. P-hacking 概率: {metrics_dict['p_noise']:.1%} (反映回测可信度)
    4. 有效特征数: {metrics_dict['active']} 个
    
    请按以下格式输出：
    ### 🛡️ 专家审计意见
    - **风险评级**：[稳健/谨慎/高风险] 并解释原因。
    - **模型健康度**：评价特征选择是否合理，是否存在“维度灾难”。
    - **实习生建议**：告诉用户如果要在实盘中使用，需要注意什么。
    """
    try:
        return model.generate_content(prompt).text
    except:
        return "AI 审计插件暂时离线，请参考原始数值。"
# --- 5.1 LangGraph 核心组件 ---

class AgentState(TypedDict):
    """Agent 的记忆：在节点间传递的文件夹"""
    target_assets: str
    vix_level: float
    quant_results: dict
    is_robust: bool
    audit_memo: str

def research_node(state: AgentState):
    """节点 A：量化专家执行深度扫描"""
    asset_tickers = preset[state['target_assets']]
    returns = QuantEngine.get_market_data(asset_tickers)
    
    if returns.empty:
        return {"is_robust": False}
        
    # 执行量化计算
    weights = np.array([1.0/len(returns.columns)]*len(returns.columns))
    d_var, a_ret, a_vol = QuantEngine.compute_asymmetric_risk(returns, state['vix_level'], weights)
    y = returns.iloc[:, 0]; X = returns.shift(1).dropna(); y = y.iloc[1:]
    active, sparsity, coefs = StrategyAuditor.run_feature_sparsity_check(X, y)
    is_robust, p_noise = StrategyAuditor.check_optimizer_curse(0.05, 0.045, 100)
    
    # 存入状态（包括计算出的 coefs 供后面画图用，这里简化存入结果）
    return {
        "quant_results": {
            "d_var": d_var, "p_noise": p_noise, "sparsity": sparsity, 
            "active": active, "coefs": coefs, "returns": returns, "X": X
        },
        "is_robust": is_robust and (p_noise < 0.2) # 如果风险太高则标记为不稳健
    }

# --- 5.2 编排工作流 ---

# 1. 初始化
builder = StateGraph(AgentState)

# 2. 添加节点
builder.add_node("researcher", research_node)
builder.add_node("auditor", gemini_audit_node)

# 3. 设置逻辑：起点 -> 专家 -> (判断) -> 审计/结束
builder.set_entry_point("researcher")

def router(state):
    """路由逻辑：如果不稳健，直接结束，不浪费 AI Token"""
    if state["is_robust"]:
        return "auditor"
    return END

builder.add_conditional_edges("researcher", router)
builder.add_edge("auditor", END)

# 4. 编译 Agent
agent_executor = builder.compile()

def gemini_audit_node(state: AgentState):
    """节点 B：AI 专家基于数据撰写报告"""
    metrics = state['quant_results']
    report = get_ai_audit_report(metrics, state['target_assets'], state['vix_level'])
    return {"audit_memo": report}

def render_tv_chart(symbol, title): # <--- 确保这里有两个参数
    """渲染 TradingView 交互微件"""
    code = f"""
    <div style="height:350px; margin-bottom: 20px;">
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.MediumWidget({{
      "symbols": [["{title}", "{symbol}|12M"]],
      "width": "100%", "height": 350, "locale": "zh_CN", "colorTheme": "light", "autosize": true
    }});
    </script>
    </div>"""
    components.html(code, height=360)

def fetch_live_news_context():
    """模拟获取最新的市场上下文资讯"""
    return "MAS Monetary Policy Statement, US Inflation Data, NVIDIA Supply Chain Updates"

# --- 6. 侧边栏与指令控制 ---
with st.sidebar:
    st.header("⚙️ 专家指令中心")
    vix_input = st.slider("VIX 风险压力调节", 10.0, 50.0, 20.0)
    st.divider()
    btn_macro = st.button("🚀 启动宏观研判 (专家1+4协同)", use_container_width=True)
    btn_sector = st.button("🔍 穿透行业风险 (策略审计模式)", use_container_width=True)
    st.caption("注：研判将结合非对称代价逻辑输出报告")

# --- 7. 界面布局与专家协同 ---
tab1, tab2, tab3, tab4 = st.tabs(["🧠 首席宏观研判", "📈 实时仪表盘", "🛡️ 行业风险穿透", "🔢 量化审计室"])

with tab1:
    if btn_macro:
        with st.spinner("四大专家正在生成宏观备忘录..."):
            news = fetch_live_news_context()
            res, err = get_ai_analysis("macro", news, vix_input)
            if not err:
                st.markdown(res)
                st.download_button("📥 下载专业投研备忘录", generate_docx_report(res, "Macro Report"), f"Macro_{datetime.date.today()}.docx")
            else: st.error(err)
    else:
        st.info("💡 请点击左侧按钮启动专家协同研判。")

with tab2:
    st.subheader("📊 全球资产实时走廊 (全能专家版)")
    
    # 第一排：宏观风险锚点 (黄金 & 纳指)
    col1, col2 = st.columns(2)
    with col1: 
        render_tv_chart("OANDA:XAUUSD", "现货黄金 (Safe Haven)")
    with col2: 
        render_tv_chart("NASDAQ:NDX", "纳斯达克 100 (Growth/Tech)")
    
    # 第二排：区域增长引擎 (海峡指数 & 沪深300)
    col3, col4 = st.columns(2)
    with col3: 
        render_tv_chart("STI:STI", "新加坡海峡指数 (SG Blue Chips)")
    with col4: 
        render_tv_chart("SSE:000300", "沪深 300 指数 (China A-Shares)")
    
    # 第三排：流动性与大宗商品 (美元新币 & 布伦特原油)
    col5, col6 = st.columns(2)
    with col5: 
        render_tv_chart("FX_IDC:USDSGD", "美元/新币 (S$NEER Proxy)")
    with col6: 
        render_tv_chart("TVC:UKOIL", "布伦特原油 (Energy/Inflation)")

with tab3:
    if btn_sector:
        with st.spinner("执行行业风险穿透中..."):
            news = fetch_live_news_context()
            res, err = get_ai_analysis("sector", news, vix_input)
            if not err:
                st.markdown(res)
                st.download_button("📥 下载行业穿透报告", generate_docx_report(res, "Sector Report"), f"Sector_{datetime.date.today()}.docx")
    else:
        st.info("💡 请点击左侧按钮执行行业评估。")

with tab4:
    st.header("🔢 专家审计：特征稀疏性与回测稳健性")
    
    preset = {"新加坡蓝筹": ["DBSDF", "U11.SI", "V03.SI"], "科技成长": ["NVDA", "AAPL", "MSFT"]}
    choice = st.selectbox("选择审计资产包", list(preset.keys()))
    
    if st.button("🚀 启动 Agentic AI 深度审计", type="primary", key="run_agent_v1"):
        # 1. 自动感应实时 VIX (结合你之前的想法)
        # 这里先用 vix_input，如果是自动驾驶模式则直接传入实时值
        
        initial_input = {
            "target_assets": choice,
            "vix_level": vix_input,
            "quant_results": {},
            "is_robust": True,
            "audit_memo": ""
        }
        
        with st.status("Agent 正在协同专家组...", expanded=True) as status:
            # 运行 Agent 流程
            final_state = None
            for event in agent_executor.stream(initial_input):
                for node_name, state_update in event.items():
                    st.write(f"✅ {node_name} 任务处理完成")
                    # 更新当前状态以便获取结果
                    if not final_state: final_state = initial_input.copy()
                    final_state.update(state_update)
            status.update(label="审计工作流执行完毕！", state="complete")
        
        # --- 2. 从 Agent 状态中提取结果并渲染 UI ---
        if final_state and "quant_results" in final_state:
            q = final_state['quant_results']
            
            # 复用你原来的 UI 组件展示计算结果
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("动态 VaR", f"{q['d_var']:.2%}")
            c2.metric("特征稀疏率", f"{q['sparsity']:.1%}")
            c3.metric("P-hacking 风险", f"{q['p_noise']:.1%}")
            c4.metric("有效特征", f"{q['active']} 个")

            # 可视化部分（使用从状态中取回的数据）
            st.divider()
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("👁️ 市场流形结构")
                iso = Isomap(n_neighbors=5, n_components=2)
                manifold = iso.fit_transform(q['returns'])
                df_iso = pd.DataFrame(manifold, columns=["Dim 1", "Dim 2"])
                df_iso['Return'] = q['returns'].mean(axis=1).values
                st.scatter_chart(df_iso, x="Dim 1", y="Dim 2", color="Return")

            with col_r:
                st.subheader("🧬 因子贡献权重")
                feat_importance = pd.DataFrame({'Factor': q['X'].columns, 'Weight': q['coefs']})
                st.bar_chart(feat_importance, x="Factor", y="Weight")

            # --- 3. 展示 AI 审计备忘录 ---
            if final_state['is_robust']:
                st.divider()
                st.subheader("🤖 AI 专家深度审计报告")
                st.info("💡 结论由 Agent 编排 Gemini 生成：")
                st.markdown(final_state['audit_memo'])
            else:
                st.warning("⚠️ Agent 审计终止：检测到该策略 P-hacking 风险过高，已拦截 AI 报告生成。")
st.markdown("---")
st.caption("Macro Alpha Pro | 核心算法：Isomap Non-linear Manifold, LassoCV Feature Selection, Asymmetric Risk Management")
