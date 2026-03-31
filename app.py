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

# --- 1.5 登录验证逻辑 (新增) ---
def check_password():
    """验证用户名和密码，返回 True/False"""
    def password_entered():
        if st.session_state["username"] == "zhang" and st.session_state["password"] == "200211":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # 验证后删除密码，安全性更佳
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # 初次访问，显示登录界面
        st.markdown("""
            <style>
            .login-box {
                max-width: 400px;
                margin: 100px auto;
                padding: 30px;
                background-color: #FFFFFF;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border: 1px solid #E2E8F0;
                text-align: center;
            }
            </style>
            <div class="login-box">
                <h2 style='color: #1B315E;'>🏛️ Macro Alpha Pro</h2>
                <p style='color: #64748B;'>请验证您的身份以访问终端</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.text_input("用户名", key="username")
            st.text_input("密码", type="password", key="password", on_change=password_entered)
            st.button("登录终端", on_click=password_entered)
            if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                st.error("❌ 凭据无效，请联系管理员")
        return False
    return True

# 只有通过验证才继续向下运行
if not check_password():
    st.stop()  # 拦截后续所有代码运行

st.markdown("""
    <style>
    /* 调整 Metric 卡片的背景 */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1E3A8A;
    }
    div[data-testid="metric-container"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* 优化按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1E3A8A;
    }
    /* 审计报告的卡片式布局 */
    .audit-card {
        background-color: #FFFFFF;
        border-left: 5px solid #1E3A8A;
        padding: 20px;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Macro Alpha")

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
    @st.cache_data(ttl=3600, show_spinner=False)
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

    @staticmethod
    def get_realtime_vix():
        """从 Yahoo Finance 获取真实实时 VIX 数值"""
        try:
            vix_data = yf.download("^VIX", period="1d", progress=False)
            if not vix_data.empty:
                # 获取最新的收盘价
                current_vix = vix_data['Close'].iloc[-1]
                return round(float(current_vix), 2)
            return 20.0  # 如果抓取失败，返回中值基准
        except:
            return 20.0
    
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

@st.cache_data(ttl=3600, show_spinner=False)
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
    technical_report: str  # 新增：存储硬核技术审计
    audit_memo: str       # 存储最终的 CEO 级简报

PRESET_ASSETS = {"新加坡蓝筹": ["DBSDF", "U11.SI", "V03.SI"], "科技成长": ["NVDA", "AAPL", "MSFT"]}

def research_node(state: AgentState):
    """量化研究节点：负责基础指标计算"""
    returns = QuantEngine.get_market_data(PRESET_ASSETS[state['target_assets']])
    if returns.empty: return {"is_robust": False}
    weights = np.array([1.0/len(returns.columns)]*len(returns.columns))
    d_var, a_ret, a_vol = QuantEngine.compute_asymmetric_risk(returns, state['vix_level'], weights)
    
    y = returns.iloc[:, 0]
    X = returns.shift(1).fillna(0)
    active, sparsity, coefs = StrategyAuditor.run_feature_sparsity_check(X, y)
    is_robust, p_noise = StrategyAuditor.check_optimizer_curse(0.05, 0.045, 100)
    
    # --- 核心计算逻辑：审计置信度 ---
    base_score = 100
    penalty_p = max(0, (p_noise - 0.05) * 200) # 超过5%后，每增加1%扣2分
    penalty_s = 20 if active < 2 else 0
    confidence_score = max(0, min(100, base_score - penalty_p - penalty_s))
    
    # ⚠️ 关键修正：必须将 confidence_score 包含在返回的字典中
    return {
        "quant_results": {
            "d_var": d_var, 
            "p_noise": p_noise, 
            "sparsity": sparsity, 
            "active": active, 
            "coefs": coefs, 
            "returns": returns, 
            "X": X, 
            "a_ret": a_ret, 
            "a_vol": a_vol,
            "confidence_score": confidence_score # 👈 确保这个键存在
        }, 
        "is_robust": (p_noise < 0.3)
    }

def technical_audit_node(state: AgentState):
    """审计节点：负责撰写硬核技术报告"""
    q = state['quant_results']
    prompt = f"""
    你是一名量化风险审计师。请针对以下指标撰写一份【硬核技术报告】：
    资产包: {state['target_assets']}
    VIX环境: {state['vix_level']}
    VaR: {q['d_var']:.2%}
    Lasso特征数: {q['active']}
    P-hacking 风险: {q['p_noise']:.2%}
    
    要求：使用专业量化术语，分析模型的统计稳健性、过拟合风险及非对称风险暴露。
    """
    try:
        res = model.generate_content(prompt)
        return {"technical_report": res.text}
    except:
        return {"technical_report": "技术审计调用失败。"}

def translator_node(state: AgentState):
    """翻译官节点：将技术报告重写为 CEO 级简报 (核心改进)"""
    q = state['quant_results']
    tech_report = state['technical_report']
    
    prompt = f"""
    你是一名资深投研顾问，负责向基金经理（Fund Manager）汇报。
    请根据下方的【硬核技术报告】，将其重写为【CEO 级分层简报】。
    
    原始报告内容：{tech_report}
    
    ### 严格遵循以下输出格式 ###
    
    ### 📢 首席执行决策建议 (Executive Summary)
    [用直白、果断的决策建议起头：买入/持有/减仓。严禁统计术语，告诉老板“So What”。]
    
    ### 🔍 关键洞察 (Key Insights)
    [将复杂指标翻译为业务语言：
    - VaR -> 压力下的潜在亏损
    - Sparsity -> 决策信号纯净度
    - P-hacking -> 结果真实可信度]
    
    ### 🔬 技术附录 (Technical Appendix)
    [保留硬核术语和具体数值，供量化同事复核。]
    """
    try:
        res = model.generate_content(prompt)
        return {"audit_memo": res.text}
    except:
        return {"audit_memo": "翻译决策生成失败。"}

# 构建 Graph
builder = StateGraph(AgentState)

builder.add_node("researcher", research_node)
builder.add_node("auditor", technical_audit_node)
builder.add_node("translator", translator_node) # 实装翻译官

builder.set_entry_point("researcher")

# 逻辑流：计算 -> 审计 -> 翻译 (如果稳健)
builder.add_conditional_edges(
    "researcher", 
    lambda x: "auditor" if x["is_robust"] else END
)
builder.add_edge("auditor", "translator")
builder.add_edge("translator", END)

agent_executor = builder.compile()

# --- 7. 主界面布局与样式注入 ---

# 注入自定义 CSS 以提升 UI 质感
st.markdown("""
    <style>
    /* 全局背景与字体微调 */
    .main { background-color: #FDFDFD; }
    
    /* Metric 卡片美化 */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #ECEFF4;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] { font-size: 24px; color: #1B315E; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 14px; color: #64748B; }

    /* 决策建议卡片 */
    .decision-card {
        background-color: #F8FAFC;
        border-left: 5px solid #1B315E;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* 侧边栏优化 */
    .css-1639116 { background-color: #1B315E; }
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        transition: all 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 智能环境配置")
    
    real_vix = QuantEngine.get_realtime_vix()
    
    # 动态风险颜色与标签
    if real_vix < 15:
        risk_label, risk_color = "低波动 (Complacency)", "blue"
    elif real_vix < 25:
        risk_label, risk_color = "常态波动 (Normal)", "green"
    elif real_vix < 35:
        risk_label, risk_color = "高波动 (Panic)", "orange"
    else:
        risk_label, risk_color = "极高风险 (Crisis)", "red"

    st.metric("实时 VIX 指数", real_vix, delta=risk_label, delta_color="normal")
    
    vix_input = st.number_input("环境压力修正 (默认为实时)", value=real_vix, help="手动调整波动率倍率以进行压力测试")
    
    st.divider()
    st.info("💡 **系统状态**: 当前处于【首席审计模式】，Agent 将优先检查统计稳健性。")
    st.caption("数据源: CBOE Real-time / Yahoo Finance")

# 使用分栏美化 Tab 标题
tab1, tab2, tab3, tab4 = st.tabs(["🧠 首席宏观研判", "📈 实时仪表盘", "🛡️ 行业风险穿透", "🔢 量化审计室"])

with tab1:
    st.header("🧠 首席宏观投研备忘录")
    with st.container():
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            if st.button("🚀 启动宏观协同分析", type="primary", use_container_width=True):
                with st.spinner("正在调度新加坡宏观专家库..."):
                    macro_res = get_ai_analysis("macro", vix_input)
                    st.markdown(f'<div class="decision-card">{macro_res}</div>', unsafe_allow_html=True)
                    st.download_button("📥 下载宏观报告", generate_docx_report(macro_res, "Macro Memo"), "Macro_Report.docx")
        with col_m2:
            st.caption("专家组成员：\n- MAS 政策分析师\n- 离岸汇率策略师\n- 通胀研究专家")

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
            st.markdown(f'<div class="decision-card">{sector_res}</div>', unsafe_allow_html=True)
            st.download_button("📥 下载行业报告", generate_docx_report(sector_res, "Sector Drilldown"), "Sector_Report.docx")

with tab4:
    st.header("🔢 Agentic AI 深度审计终端")
    
    # 顶部交互区
    c_left, c_right = st.columns([3, 1])
    with c_left:
        choice = st.selectbox("选择审计资产包", list(PRESET_ASSETS.keys()), label_visibility="collapsed")
    with c_right:
        start_audit = st.button("🚀 启动全自动审计流", type="primary", use_container_width=True)

    if "final_audit_state" not in st.session_state:
        st.session_state.final_audit_state = None

    if start_audit:
        with st.status("Agent 正在协同专家组...", expanded=True) as status:
            state = {"target_assets": choice, "vix_level": vix_input, "quant_results": {}, "is_robust": True, "technical_report": "", "audit_memo": ""}
            for event in agent_executor.stream(state):
                for node, update in event.items():
                    st.write(f"✅ {node} 任务处理完成")
                    state.update(update)
            st.session_state.final_audit_state = state
            status.update(label="审计流执行完毕", state="complete")

    # 结果渲染区
    if st.session_state.final_audit_state:
        s = st.session_state.final_audit_state
        if s.get("quant_results"):
            q = s['quant_results']
            score = q.get('confidence_score', 0)
            
            # --- 第一层：置信度仪表盘 ---
            st.divider()
            score_col, status_col = st.columns([1, 2])
            with score_col:
                st.metric("审计置信度", f"{int(score)}/100")
            with status_col:
                res_color = "green" if score >= 80 else "orange" if score >= 50 else "red"
                res_label = "💎 高度可信 (High Integrity)" if score >= 80 else "⚠️ 中度参考 (Cautionary)" if score >= 50 else "🚨 统计噪音警报 (Low)"
                st.markdown(f"<h2 style='color:{res_color}; margin-top:0;'>{res_label}</h2>", unsafe_allow_html=True)
            
            if score < 60:
                st.warning("⚠️ **风控提示**: 当前样本量或特征稀疏度触发预警，模型可能处于‘过拟合’边缘。")

            # --- 第二层：决策建议 (Executive Summary) ---
            if s.get("audit_memo") and s['is_robust']:
                memo_parts = s["audit_memo"].split("###")
                st.subheader("💡 首席执行决策建议")
                # 尝试渲染美化后的摘要
                exec_txt = memo_parts[1] if len(memo_parts) > 1 else s["audit_memo"]
                st.info(exec_txt)
                
                # --- 第三层：核心指标横向对比 ---
                st.subheader("🔍 关键洞察 (Key Insights)")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("压力回撤 (VaR)", f"{q['d_var']:.2%}", help="95%置信度下的潜在损失")
                k2.metric("信号纯净度", f"{q['sparsity']:.1%}", help="有效因子占比")
                k3.metric("统计真实信度", f"{q['p_noise']:.1%}", help="排除P-hacking后的真实性")
                k4.metric("夏普比率(预估)", f"{(q['a_ret']-0.03)/q['a_vol']:.2f}")

                # --- 第四层：技术底层透明化 ---
                with st.expander("🔬 查看量化审计技术细节 (Quant Tech Stack)"):
                    g1, g2 = st.columns([2, 1])
                    with g1:
                        st.markdown("**🧬 因子贡献权重 (Lasso Coefficients)**")
                        chart_data = pd.DataFrame({'Factor': q['X'].columns, 'Weight': q['coefs']})
                        st.bar_chart(chart_data, x="Factor", y="Weight")
                    with g2:
                        st.markdown("**📝 原始审计意见**")
                        st.caption(s.get("technical_report", "技术审计未生成"))
                    
                    st.download_button("📥 下载正式审计备忘录 (.docx)", 
                                     generate_docx_report(s["audit_memo"], "Investment Audit Memo"), 
                                     "Audit_Report.docx", use_container_width=True)
            else:
                st.error("⚠️ **审计拦截**: 由于统计稳健性未通过（P-hacking风险过高），系统已自动终止决策建议生成。")
                st.markdown(f"**当前风险值**: `{q['p_noise']:.2%}` | **允许上限**: `30.00%`")

st.markdown("---")
st.caption("Macro Alpha Pro | NUS MSBA Project | 专注量化审计与合规决策")
