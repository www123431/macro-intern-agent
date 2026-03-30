import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import yfinance as yf
import plotly.graph_objects as go
import urllib.parse
import pandas as pd

# 1. 页面设置
st.set_page_config(page_title="Macro Alpha Pro Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 全球宏观与行业策略终端 (终极稳定版)")

# 2. 安全读取 Secrets
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整，请检查 Streamlit Cloud 的 Secrets 设置")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置读取失败: {e}")
    st.stop()

# 3. 初始化 Gemini
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# --- 核心功能：带缓存的 AI 生成函数 ---
@st.cache_data(ttl=600)  # 缓存 10 分钟
def get_ai_analysis(prompt_type, sg_data, tech_data, geo_data):
    try:
        if prompt_type == "macro":
            full_prompt = f"作为常驻新加坡的首席策略师，分析以下资讯并给出针对新加坡CPI和新元汇率的犀利见解。资讯：{sg_data}, {geo_data}"
        else:
            full_prompt = f"""
            你是一名常驻新加坡的首席行业策略师。基于以下最新动态：
            科技动态：{tech_data} | 地缘背景：{geo_data}
            请针对 CPO、AI 算力、大宗商品及新加坡蓝筹股给出深度解读，并标注投资建议（建议买入/退出/持续持有）。
            要求：以【板块名称】-【建议动作】-【深度解读】-【风险等级】结构输出。语气冷峻专业。
            """
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI 模块暂时无法响应: {str(e)}"

# 4. 增强版数据抓取 (用于 AI 分析)
def fetch_macro_sector_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Inflation")')
        sg_res = requests.get(f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en", headers=headers, timeout=10)
        sg_feed = feedparser.parse(sg_res.content)
        
        tech_query = urllib.parse.quote('("CPO" OR "Nvidia" OR "AI Computing") AND ("Market" OR "Risk")')
        tech_res = requests.get(f"https://gnews.io/api/v4/search?q={tech_query}&lang=en&max=3&apikey={GNEWS_KEY}", timeout=10).json()
        
        geo_query = urllib.parse.quote('("Geopolitics" OR "War") AND ("Oil" OR "Supply Chain")')
        geo_res = requests.get(f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=2&apikey={GNEWS_KEY}", timeout=10).json()

        return sg_feed.entries[:3], tech_res.get('articles', []), geo_res.get('articles', [])
    except:
        return [], [], []

# 5. 【核心修复】极致稳定版图表渲染 (全线 Python 原生驱动)
def render_stable_metric(ticker, label):
    try:
        # 抓取最近 30 天数据
        data = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if data.empty:
            st.error(f"{label} 数据源断开")
            return
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        delta = ((current_price - prev_price) / prev_price) * 100
        
        # 绘制专业面积图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines',
            line=dict(color='#2962FF', width=3),
            fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)'
        ))
        
        fig.update_layout(
            title=f"<b>{label}</b><br><span style='font-size:18px;'>{current_price:.2f} ({delta:+.2f}%)</span>",
            height=220, margin=dict(l=10, r=10, t=50, b=10),
            xaxis_visible=False, yaxis_gridcolor='rgba(200, 200, 200, 0.2)',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    except:
        st.write(f"❌ {label} 载入错误")

# --- 界面分栏 ---
tab1, tab2, tab3 = st.tabs(["🧠 首席宏观研判", "📈 实时全球仪表盘", "🛡️ 行业风险穿透"])

# TAB 1: 宏观研判
with tab1:
    st.info("👈 点击侧边栏按钮启动 AI 深度研判")
    if st.sidebar.button("🚀 启动深度宏观研判"):
        with st.spinner("策略师正在扫描全球图谱..."):
            sg, tech, geo = fetch_macro_sector_data()
            result = get_ai_analysis("macro", sg, tech, geo)
            st.markdown(result)

# TAB 2: 仪表盘 (全线修复版)
with tab2:
    st.subheader("📊 实时行情监测 (极致稳健模式)")
    
    # 按照你的需求排列资产
    assets = [
        {"ticker": "GC=F", "name": "✨ 伦敦金现货"},
        {"ticker": "^TNX", "name": "🇺🇸 10年美债收益率"},
        {"ticker": "^NDX", "name": "🇺🇸 纳斯达克 100"},
        {"ticker": "BZ=F", "name": "🛢️ 布伦特原油"},
        {"ticker": "000300.SS", "name": "🇨🇳 沪深 300"},
        {"ticker": "^STI", "name": "🇸🇬 海峡时报指数 (STI)"},
        {"ticker": "USDSGD=X", "name": "💵 美元/新元"},
    ]
    
    col1, col2, col3 = st.columns(3)
    for i, asset in enumerate(assets):
        with [col1, col2, col3][i % 3]:
            render_stable_metric(asset["ticker"], asset["name"])

# TAB 3: 行业风险穿透
with tab3:
    if st.sidebar.button("🔍 穿透行业风险"):
        with st.spinner("正在评估板块溢价与安全边际..."):
            sg, tech, geo = fetch_macro_sector_data()
            result = get_ai_analysis("sector", sg, tech, geo)
            st.markdown(result)

st.markdown("---")
st.caption("Macro Alpha v4.4 | 业务分析实习辅助工具 | 已切换至本地数据处理架构")
