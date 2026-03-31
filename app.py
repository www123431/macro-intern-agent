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
    st.header("⚙️ 终端控制")
    vix_input = st.slider("VIX 压力测试", 10.0, 50.0, 20.0)
    st.sidebar.markdown("---")
    btn_macro = st.button("🚀 启动深度宏观研判", use_container_width=True)
    btn_sector = st.button("🔍 穿透行业风险", use_container_width=True)

# --- 7. 界面布局 ---
tab1, tab2, tab3, tab4 = st.tabs(["🧠 首席宏观研判", "📈 实时仪表盘", "🛡️ 行业风险穿透", "🔢 量化工作台"])

with tab1:
    if btn_macro:
        with st.spinner("策略师正在扫描全球图谱..."):
            sg, tech, geo = fetch_macro_sector_data()
            res, err = get_ai_analysis("macro", sg, tech, geo, vix_input)
            if not err:
                st.markdown(res)
                st.download_button("📥 下载专业备忘录", generate_docx_report(res, "Macro Report"), f"Macro_{datetime.date.today()}.docx")
            else: st.error(err)
    else: st.info("点击左侧按钮启动宏观分析")

with tab2:
    col1, col2 = st.columns(2)
    with col1: render_tv_widget("OANDA:XAUUSD", "Gold")
    with col2: render_tv_widget("NASDAQ:NDX", "Nasdaq 100")

with tab3:
    if btn_sector:
        with st.spinner("风险穿透中..."):
            sg, tech, geo = fetch_macro_sector_data()
            res, err = get_ai_analysis("sector", sg, tech, geo, vix_input)
            if not err:
                st.markdown(res)
                st.download_button("📥 下载行业报告", generate_docx_report(res, "Sector Report"), f"Sector_{datetime.date.today()}.docx")
    else: st.info("点击左侧按钮执行行业风险评估")

with tab4:
    st.header("🔢 量化实验场")
    asset_presets = {"🇸🇬 新加坡蓝筹": ["DBSDF", "U11.SI", "V03.SI"], "🇺🇸 纳指科技": ["NVDA", "AAPL", "MSFT"]}
    choice = st.selectbox("资产预设", list(asset_presets.keys()))
    if st.button("🔄 执行硬核审计"):
        mean_r, cov_m, raw_ret = QuantEngine.get_portfolio_stats(asset_presets[choice])
        if raw_ret is not None:
            orch = AdvancedStrategicOrchestrator(vix_input)
            weights = np.array([1.0/len(mean_r)]*len(mean_r))
            ann_r, ann_v, sharpe, var, cvar = QuantEngine.calculate_risk_metrics(raw_ret, weights)
            risk_bounds = orch.compute_quantile_risk_bounds(raw_ret.dot(weights))
            manifold_df = orch.extract_market_manifold(raw_ret)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("动态安全边界", f"{risk_bounds['dynamic_vaR']:.2%}")
            m2.metric("年化波动", f"{ann_v:.2%}")
            m3.metric("Sharpe", f"{sharpe:.2f}")
            st.line_chart(manifold_df)
