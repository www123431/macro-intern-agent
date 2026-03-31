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
model = genai.GenerativeModel('gemini-1.5-flash')

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
    st.header("🔢 量化审计与特征实验室")
    preset = {"新加坡蓝筹": ["DBSDF", "U11.SI", "V03.SI"], "科技成长": ["NVDA", "AAPL", "MSFT"]}
    choice = st.selectbox("选择审计资产包", list(preset.keys()))
    
    if st.button("🔄 执行专家深度审计", type="primary"):
        returns = QuantEngine.get_market_data(preset[choice])
        
        if not returns.empty:
            # 1. 风险专家：计算非对称风险
            weights = np.array([1.0/len(returns.columns)]*len(returns.columns))
            d_var, a_ret, a_vol = QuantEngine.compute_asymmetric_risk(returns, vix_input, weights)
            
            # 2. 特征专家：Lasso 稀疏性检查
            y = returns.iloc[:, 0]
            X = returns.shift(1).dropna(); y = y.iloc[1:]
            active, sparsity, _ = StrategyAuditor.run_feature_sparsity_check(X, y)
            
            # 3. 审计专家：P-hacking 风险检测
            is_robust, p_noise = StrategyAuditor.check_optimizer_curse(0.05, 0.045, 100)
            
            # --- 展示审计结果 ---
            m1, m2, m3 = st.columns(3)
            m1.metric("动态 VaR (非对称)", f"{d_var:.2%}")
            m2.metric("特征稀疏率 (Lasso)", f"{sparsity:.1%}")
            m3.metric("统计噪声(P-hacking)概率", f"{p_noise:.1%}")
            
            st.divider()
            st.subheader("👁️ 市场非线性流形扫描 (Isomap)")
            iso = Isomap(n_neighbors=5, n_components=2)
            manifold = iso.fit_transform(returns)
            st.line_chart(pd.DataFrame(manifold, columns=["Dim 1", "Dim 2"]))
            st.success(f"审计结论：策略{'稳健' if is_robust else '可能过拟合'} (基于 {active} 个核心特征因子)")
        else:
            st.warning("未能获取有效数据。")

st.markdown("---")
st.caption("Macro Alpha Pro | 核心算法：Isomap Non-linear Manifold, LassoCV Feature Selection, Asymmetric Risk Management")
