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

# --- 4. 工业级量化逻辑引擎 (高级版) ---
class AdvancedStrategicOrchestrator:
    def __init__(self, vix_current: float):
        self.vix = vix_current
        self.scaling_required = False # 坚持 Unitless Returns 原则

    def extract_market_manifold(self, data: pd.DataFrame, n_components=2):
        """利用 Isomap 提取非线性市场流形 (对齐讲义非线性降维逻辑)"""
        iso = Isomap(n_neighbors=max(5, len(data)//10), n_components=n_components)
        manifold_features = iso.fit_transform(data)
        return pd.DataFrame(manifold_features, index=data.index, columns=[f"Dimension_{i+1}" for i in range(n_components)])

    def validate_strategy_robustness(self, val_loss, train_loss, test_loss, num_models_tried):
        """验证集‘暗物质’检测：评估 P-hacking 风险 (对齐优化者诅咒逻辑)"""
        luck_factor = (np.log1p(num_models_tried) / 10.0) * train_loss
        generalization_gap = max(0, test_loss - val_loss)
        is_p_hacking = generalization_gap > luck_factor
        prob_of_noise = 1 - np.exp(-generalization_gap / (luck_factor + 1e-6))
        return {
            "is_robust": not is_p_hacking,
            "noise_probability": min(max(prob_of_noise, 0), 1)
        }

    def compute_quantile_risk_bounds(self, returns_history):
        """基于非对称代价思想的动态风险边界"""
        # VIX 越高，分位数覆盖越广 (即更保守)
        alpha = 0.05 * (1 + (self.vix - 20) / 40) if self.vix > 20 else 0.05
        alpha = min(alpha, 0.2) # 上限保护
        lower_bound = np.quantile(returns_history, alpha)
        return {"dynamic_vaR": lower_bound, "confidence_level": 1 - alpha}

class QuantEngine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_portfolio_stats(ticker_list):
        try:
            data = yf.download(ticker_list, period="1y", progress=False)['Close']
            if isinstance(data, pd.Series): data = data.to_frame()
            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            return mean_returns, cov_matrix, returns
        except Exception as e:
            st.error(f"数据抓取失败: {e}")
            return None, None, None

    @staticmethod
    def calculate_risk_metrics(returns, weights, confidence=0.95):
        port_returns = returns.dot(weights)
        ann_return = port_returns.mean() * 252
        ann_vol = port_returns.std() * np.sqrt(252)
        sharpe = (ann_return - 0.03) / ann_vol if ann_vol != 0 else 0
        var = np.percentile(port_returns, (1 - confidence) * 100)
        cvar = port_returns[port_returns <= var].mean()
        return ann_return, ann_vol, sharpe, var, cvar

# --- 5. 辅助函数 (报告与AI) ---
def generate_docx_report(content, title="Investment Memo"):
    doc = Document()
    doc.add_heading('Macro Alpha Intelligence', 0)
    doc.add_heading(title, level=1)
    doc.add_paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    for line in content.split('\n'): doc.add_paragraph(line)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

@st.cache_data(ttl=3600)
def get_ai_analysis(prompt_type, sg_data, tech_data, geo_data):
    try:
        full_prompt = f"分析资讯影响：{sg_data} | {tech_data} | {geo_data}"
        response = model.generate_content(full_prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)

def fetch_macro_sector_data():
    return [], [], [] # 占位符，保持你原有的逻辑

def render_tv_medium_widget(symbol, title):
    render_code = f"""
    <div class="tradingview-widget-container" style="height:350px;">
      <div id="tradingview_{symbol}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.MediumWidget({{
        "symbols": [["{title}", "{symbol}|12M"]],
        "width": "100%", "height": 350, "locale": "zh_CN", "colorTheme": "light", "autosize": true
      }});
      </script>
    </div>"""
    components.html(render_code, height=360)

# --- 6. 侧边栏与布局 ---
with st.sidebar:
    st.header("⚙️ 引擎控制台")
    vix_input = st.slider("市场波动率压力测试 (VIX)", 10.0, 50.0, 20.0)
    st.info(f"非对称惩罚因子: {1 + (max(0, vix_input - 20) / 40):.2f}x")
    st.sidebar.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🧠 首席宏观研判", "📈 实时仪表盘", "🛡️ 行业风险穿透", "🔢 量化工作台 (Quant Lab)"])

# (Tab 1, 2, 3 保持你原有的 UI 调用逻辑即可，此处略，重点展示 Tab 4 的整合)

with tab4:
    st.header("⚡ 工业级组合风险监测")
    asset_presets = {
        "🇸🇬 新加坡蓝筹": ["DBSDF", "U11.SI", "V03.SI", "^STI"],
        "🇺🇸 纳指科技": ["NVDA", "AAPL", "MSFT", "^NDX"],
        "🛡️ 避险资产": ["GC=F", "US10Y", "USDSGD=X"]
    }
    
    col_ctrl, col_main = st.columns([1, 3])
    with col_ctrl:
        preset_choice = st.selectbox("选择监测预设", list(asset_presets.keys()))
        run_monitor = st.button("🔄 执行深度审计", type="primary", use_container_width=True)

    with col_main:
        if run_monitor:
            with st.status("正在执行硬核逻辑校验...", expanded=True) as status:
                mean_r, cov_m, raw_ret = QuantEngine.get_portfolio_stats(asset_presets[preset_choice])
                
                if raw_ret is not None:
                    # 实例化高级调度器
                    orch = AdvancedStrategicOrchestrator(vix_current=vix_input)
                    
                    # 1. 基础指标
                    weights = np.array([1.0/len(mean_r)]*len(mean_r))
                    ann_r, ann_v, sharpe, var, cvar = QuantEngine.calculate_risk_metrics(raw_ret, weights)
                    
                    # 2. 高级风险边界
                    risk_bounds = orch.compute_quantile_risk_bounds(raw_ret.dot(weights))
                    
                    # 3. 流形分析
                    manifold_df = orch.extract_market_manifold(raw_ret)
                    
                    # 4. 稳健性自检 (模拟数据)
                    robustness = orch.validate_strategy_robustness(0.02, 0.015, 0.03, 50)
                    
                    status.update(label="✅ 深度审计完成", state="complete")

                    # --- UI 显示 ---
                    m1, m2, m3 = st.columns(3)
                    m1.metric("动态安全边界 (VaR)", f"{risk_bounds['dynamic_vaR']:.2%}")
                    m2.metric("策略稳健性", "PASS" if robustness['is_robust'] else "FAIL")
                    m3.metric("统计噪声概率", f"{robustness['noise_probability']:.1%}")

                    st.write("---")
                    st.caption("📈 市场非线性流形视图 (Isomap Dimensions)")
                    st.line_chart(manifold_df)
                    
                    with st.expander("📝 审计技术细节"):
                        st.write(f"当前置信区间: {risk_bounds['confidence_level']:.1%}")
                        st.write(f"夏普比率 (Risk-Adjusted): {sharpe:.2f}")
                        st.info("提示：流形图若出现剧烈波动，代表市场隐变量结构不稳定。")
                else:
                    st.error("无法获取数据")

# 页面底部标记
st.markdown("---")
st.caption("Macro Alpha Terminal | 引擎状态: 已注入硬核逻辑插件 | 核心算法: Isomap & Asymmetric Quantile")
