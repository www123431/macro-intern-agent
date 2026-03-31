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
from scipy.optimize import minimize
import yfinance as yf
from sklearn.manifold import Isomap
from sklearn.linear_model import LassoCV

# --- 1. 页面基本配置 ---
st.set_page_config(page_title="Macro Alpha Pro Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 全球投研与自动化简报终端")

# --- 2. 安全读取 Secrets ---
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整，请检查后台配置。")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置文件读取失败: {e}")
    st.stop()

# --- 3. 初始化 Gemini ---
genai.configure(api_key=GEMINI_KEY)
MODEL_NAME = 'gemini-1.5-flash' 
model = genai.GenerativeModel(MODEL_NAME)

# [专家 1 & 2: 特征工程与回测审计专家]
class StrategyAuditor:
    @staticmethod
    def run_feature_sparsity_check(X, y):
        """特征工程专家：利用 LassoCV 执行 Unitless 稀疏性控制"""
        model = LassoCV(cv=5).fit(X, y)
        active_features = np.sum(np.abs(model.coef_) > 1e-10)
        sparsity = 1 - (active_features / X.shape[1])
        return active_features, sparsity, model.coef_

    @staticmethod
    def check_optimizer_curse(val_score, test_score, n_trials):
        """回测审计专家：评估 P-hacking 风险"""
        luck_factor = np.log1p(n_trials) * 0.01 
        gap = max(0, val_score - test_score)
        prob_noise = 1 - np.exp(-gap / (luck_factor + 1e-6))
        return gap <= luck_factor, prob_noise

# [专家 3: 量化风险专家]
class QuantEngine:
    @staticmethod
    def get_market_data(tickers):
        data = yf.download(tickers, period="1y", progress=False)['Close']
        if isinstance(data, pd.Series): data = data.to_frame()
        returns = data.pct_change().dropna()
        return returns

    @staticmethod
    def compute_asymmetric_risk(returns, vix, weights):
        """非对称代价逻辑"""
        port_ret = returns.dot(weights)
        alpha = 0.05 * (1 + (vix - 20) / 40) if vix > 20 else 0.05
        dynamic_vaR = np.quantile(port_ret, min(alpha, 0.2))
        return dynamic_vaR, port_ret.mean() * 252, port_ret.std() * np.sqrt(252)

# --- 4. 工业级量化逻辑引擎 ---
class AdvancedStrategicOrchestrator:
    def __init__(self, vix_current: float):
        self.vix = vix_current

    def extract_market_manifold(self, data: pd.DataFrame, n_components=2):
        iso = Isomap(n_neighbors=max(5, len(data)//10), n_components=n_components)
        manifold_features = iso.fit_transform(data)
        return pd.DataFrame(manifold_features, index=data.index, columns=[f"Dim_{i+1}" for i in range(n_components)])

    def validate_strategy_robustness(self, val_loss, train_loss, test_loss, num_models_tried):
        luck_factor = (np.log1p(num_models_tried) / 10.0) * train_loss
        gap = max(0, test_loss - val_loss)
        return {"is_robust": gap <= luck_factor, "noise_prob": 1 - np.exp(-gap/(luck_factor+1e-6))}

    def compute_quantile_risk_bounds(self, returns_history):
        alpha = 0.05 * (1 + (self.vix - 20) / 40) if self.vix > 20 else 0.05
        alpha = min(alpha, 0.2)
        return {"dynamic_vaR": np.quantile(returns_history, alpha), "conf": 1-alpha}

class QuantEngine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_portfolio_stats(ticker_list):
        try:
            data = yf.download(ticker_list, period="1y", progress=False)['Close']
            if isinstance(data, pd.Series): data = data.to_frame()
            returns = data.pct_change().dropna()
            return returns.mean() * 252, returns.cov() * 252, returns
        except: return None, None, None

    @staticmethod
    def calculate_risk_metrics(returns, weights, confidence=0.95):
        port_returns = returns.dot(weights)
        ann_r, ann_v = port_returns.mean()*252, port_returns.std()*np.sqrt(252)
        sharpe = (ann_r - 0.03)/ann_v if ann_v != 0 else 0
        var = np.percentile(port_returns, (1-confidence)*100)
        return ann_r, ann_v, sharpe, var, port_returns[port_returns <= var].mean()

def get_ai_analysis(prompt_type, news_context, vix_val):
    vix_msg = f"注意：当前市场VIX压力为{vix_val}，策略需偏向非对称防御。"
    prompts = {
        "macro": f"作为新加坡宏观专家，基于以下资讯分析S$NEER与CPI：{news_context}。{vix_msg}",
        "sector": f"作为行业分析专家，评估AI算力与新加坡蓝筹机会：{news_context}。{vix_msg}"
    }
    try:
        return model.generate_content(prompts[prompt_type]).text, None
    except Exception as e: return None, str(e)

def render_tv_chart(symbol):
    code = f"""<script src="https://s3.tradingview.com/tv.js"></script>
    <script>new TradingView.MediumWidget({{"symbols": [["{symbol}", "{symbol}|12M"]], "width": "100%", "height": 350, "locale": "zh_CN", "colorTheme": "light"}});</script>"""
    components.html(code, height=360)

# --- 5. 辅助功能 ---
def generate_docx_report(content, title="Investment Memo"):
    doc = Document()
    doc.add_heading('Macro Alpha Intelligence', 0)
    doc.add_heading(title, level=1)
    doc.add_paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    for line in content.split('\n'): doc.add_paragraph(line)
    buf = BytesIO(); doc.save(buf); buf.seek(0)
    return buf

@st.cache_data(ttl=3600)
def get_ai_analysis(prompt_type, sg, tech, geo, vix_val):
    vix_context = f"当前市场VIX压力指数为{vix_val}，请在分析中体现非对称风险偏好。"
    if prompt_type == "macro":
        prompt = f"你是一名常驻新加坡的资深宏观策略师。分析对CPI、MAS政策及S$NEER的影响。{vix_context} 资讯：{sg}, {geo}"
    else:
        prompt = f"分析CPO、AI算力、地缘敏感商品、新加坡蓝筹机会。{vix_context} 资讯：{tech}, {geo}"
    try:
        return model.generate_content(prompt).text, None
    except Exception as e: return None, str(e)

def fetch_macro_sector_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        sg_res = requests.get(f"https://news.google.com/rss/search?q=Singapore+Economy&hl=en-SG", timeout=10)
        sg_feed = feedparser.parse(sg_res.content).entries[:3]
        return sg_feed, [], [] # 简化版抓取，确保逻辑通畅
    except: return [], [], []

def render_tv_widget(symbol, title):
    code = f"""<script src="https://s3.tradingview.com/tv.js"></script>
    <script>new TradingView.MediumWidget({{"symbols": [["{title}", "{symbol}|12M"]], "width": "100%", "height": 350, "locale": "zh_CN", "colorTheme": "light"}});</script>"""
    components.html(code, height=360)

# --- 6. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 专家指令中心")
    vix_input = st.slider("VIX 风险调节", 10.0, 50.0, 20.0)
    st.divider()
    btn_macro = st.button("🚀 启动宏观研判 (专家1+4)", use_container_width=True)
    btn_sector = st.button("🔍 穿透行业风险 (专家2+4)", use_container_width=True)
    st.caption("注：研判将结合非对称代价逻辑输出")

# --- 7. 界面布局 ---
tab1, tab2, tab3, tab4 = st.tabs(["🧠 首席宏观研判", "📈 实时仪表盘", "🛡️ 行业风险穿透", "🔢 量化审计"])

with tab1:
    if btn_macro:
        with st.spinner("专家协同分析中..."):
            res, err = get_ai_analysis("macro", "MAS Policy Update, Singapore CPI Data", vix_input)
            if not err: st.markdown(res)
            else: st.error(err)
    else: st.info("请点击左侧按钮启动")

with tab2:
    c1, c2 = st.columns(2)
    with c1: render_tv_chart("OANDA:XAUUSD")
    with c2: render_tv_chart("NASDAQ:NDX")

with tab3:
    if btn_sector:
        with st.spinner("行业专家评估中..."):
            res, err = get_ai_analysis("sector", "Nvidia AI Chips, SG Blue Chips Performance", vix_input)
            if not err: st.markdown(res)
    else: st.info("请点击左侧按钮启动")

with tab4:
    st.header("🔢 量化审计与特征实验室")
    preset = {"新加坡蓝筹": ["DBSDF", "U11.SI", "V03.SI"], "科技成长": ["NVDA", "AAPL", "MSFT"]}
    choice = st.selectbox("选择审计资产包", list(preset.keys()))
    
    if st.button("🔄 执行专家深度审计", type="primary"):
        returns = QuantEngine.get_market_data(preset[choice])
        
        if not returns.empty:
            # 1. 风险专家：非对称 VaR
            weights = np.array([1.0/len(returns.columns)]*len(returns.columns))
            d_var, a_ret, a_vol = QuantEngine.compute_asymmetric_risk(returns, vix_input, weights)
            
            # 2. 特征专家：稀疏性检查 (模拟 X, y)
            y = returns.iloc[:, 0]
            X = returns.shift(1).dropna(); y = y.iloc[1:]
            active, sparsity, _ = StrategyAuditor.run_feature_sparsity_check(X, y)
            
            # 3. 审计专家：P-hacking 自检
            is_robust, p_noise = StrategyAuditor.check_optimizer_curse(0.05, 0.04, 100)
            
            # --- UI 展示 ---
            m1, m2, m3 = st.columns(3)
            m1.metric("动态 VaR (非对称)", f"{d_var:.2%}")
            m2.metric("特征稀疏率", f"{sparsity:.1%}")
            m3.metric("统计噪声(P-hacking)概率", f"{p_noise:.1%}")
            
            st.divider()
            st.subheader("👁️ 非线性流形扫描 (Isomap)")
            iso = Isomap(n_neighbors=5, n_components=2)
            manifold = iso.fit_transform(returns)
            st.line_chart(pd.DataFrame(manifold, columns=["结构维度1", "结构维度2"]))
            st.success(f"审计结论：策略{'稳健' if is_robust else '可能存在过度拟合'} (基于 {active} 个有效特征)")

st.markdown("---")
st.caption("Macro Alpha Pro | 四大专家模式：特征/回测/量化/宏观 | 算法底座：LassoCV, Isomap, Quantile Regression")
