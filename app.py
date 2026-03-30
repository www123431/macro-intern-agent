import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import yfinance as yf
import plotly.graph_objects as go
import urllib.parse
import streamlit.components.v1 as components

# 1. 页面基本设置
st.set_page_config(page_title="Macro Alpha Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 混合动力稳定版终端")

# 2. 安全读取 Secrets
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整，请检查 Streamlit Cloud 设置")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置错误: {e}")
    st.stop()

# 3. 初始化 Gemini
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# --- 核心功能：带缓存的 AI 生成函数 ---
@st.cache_data(ttl=600)
def get_ai_analysis(prompt_type, sg_data, tech_data, geo_data):
    try:
        if prompt_type == "macro":
            full_prompt = f"作为常驻新加坡的首席策略师，分析：{sg_data}, {geo_data}。重点研判新加坡CPI与新元汇率。"
        else:
            full_prompt = f"""
            分析以下板块并给建议（买入/退出/持有）：
            1. CPO与光模块 | 2. AI算力与半导体 | 3. 地缘敏感大宗商品 | 4. 新加坡本地蓝筹
            资讯：{tech_data} | {geo_data}
            """
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"分析暂时不可用: {str(e)}"

# 4. 增强版数据抓取
def fetch_macro_sector_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Inflation")')
        sg_res = requests.get(f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en", headers=headers, timeout=10)
        sg_feed = feedparser.parse(sg_res.content)
        
        tech_query = urllib.parse.quote('("CPO" OR "Nvidia") AND ("Market" OR "Risk")')
        tech_res = requests.get(f"https://gnews.io/api/v4/search?q={tech_query}&lang=en&max=3&apikey={GNEWS_KEY}", timeout=10).json()
        
        geo_query = urllib.parse.quote('("Geopolitics" OR "War") AND ("Oil" OR "Gold")')
        geo_res = requests.get(f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=2&apikey={GNEWS_KEY}", timeout=10).json()

        return sg_feed.entries[:3], tech_res.get('articles', []), geo_res.get('articles', [])
    except:
        return [], [], []

# 5. 【修复专用】Python 原生 Plotly 极致稳定图表 (用于美债、纳指)
def render_native_plotly_chart(ticker, name):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty:
            st.warning(f"{name} 数据源不可用")
            return
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        delta = ((current_price - prev_price) / prev_price) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', 
                                 line=dict(color='#2962FF', width=2), fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)'))
        fig.update_layout(title=f"📈 {name}: {current_price:.2f} ({delta:+.2f}%)", 
                          margin=dict(l=0, r=0, t=30, b=0), height=230, xaxis_visible=False, paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        st.error(f"{name} 载入失败")

# 6. 【保留专用】TradingView 极致兼容挂件 (用于油、金、汇、指数)
def render_tv_compatible_chart(symbol):
    render_code = f"""
    <div class="tradingview-widget-container" style="height:230px;">
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
      {{
        "symbol": "{symbol}", "width": "100%", "height": 220, "locale": "zh_CN",
        "dateRange": "12M", "colorTheme": "light", "trendLineColor": "rgba(41, 98, 255, 1)",
        "underLineColor": "rgba(41, 98, 255, 0.3)", "isTransparent": false
      }}
      </script>
    </div>"""
    components.html(render_code, height=230)

# --- 界面分栏 ---
tab1, tab2, tab3 = st.tabs(["🧠 首席宏观研判", "📈 混合动力仪表盘", "🛡️ 行业风险穿透"])

with tab1:
    if st.sidebar.button("🚀 启动深度宏观研判"):
        with st.spinner("策略师扫描中..."):
            sg, tech, geo = fetch_macro_sector_data()
            st.markdown(get_ai_analysis("macro", sg, tech, geo))

with tab2:
    st.subheader("📊 实时行情监测 (混合稳定型)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**🛢️ 布伦特原油 (UKOIL)**")
        render_tv_compatible_chart("TVC:UKOIL")
        st.write("**✨ 伦敦金 (XAUUSD)**")
        render_tv_compatible_chart("OANDA:XAUUSD")
    with col2:
        st.write("**🇺🇸 10年美债 (US10Y) - 稳定版**")
        render_native_plotly_chart("^TNX", "US10Y")
        st.write("**🇺🇸 纳斯达克 (NDX) - 稳定版**")
        render_native_plotly_chart("^NDX", "Nasdaq 100")
    with col3:
        st.write("**🇨🇳 沪深 300 (CSI300)**")
        render_tv_compatible_chart("SSE:000300")
        st.write("**💵 美元/新元 (USDSGD)**")
        render_tv_compatible_chart("FX_IDC:USDSGD")

with tab3:
    if st.sidebar.button("🔍 穿透行业风险"):
        with st.spinner("风险评估中..."):
            sg, tech, geo = fetch_macro_sector_data()
            st.markdown(get_ai_analysis("sector", sg, tech, geo))

st.markdown("---")
st.caption("Macro Alpha v4.3 | 局部稳定版修复 | 实习辅助工具")
