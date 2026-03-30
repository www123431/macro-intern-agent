import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import urllib.parse
import streamlit.components.v1 as components

# 1. 页面基本设置
st.set_page_config(page_title="Macro Alpha Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 全球宏观与行业策略终端")

# 2. 安全读取 Secrets
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 未配置完全")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置读取失败: {e}")
    st.stop()

# 3. 初始化 Gemini
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# --- 核心功能：AI 生成函数 ---
@st.cache_data(ttl=600)
def get_ai_analysis(prompt_type, sg_data, tech_data, geo_data):
    try:
        if prompt_type == "macro":
            full_prompt = f"作为常驻新加坡的首席策略师，分析：{sg_data}, {geo_data}。重点研判新加坡CPI与新元汇率。"
        else:
            full_prompt = f"分析以下板块并给建议（买入/退出/持有）：CPO、AI算力、地缘敏感大宗、新加坡蓝筹。资讯：{tech_data} | {geo_data}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI 模块暂时繁忙，请稍后再试。"

# 4. 数据抓取函数
def fetch_macro_sector_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Inflation")')
        sg_res = requests.get(f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en", headers=headers, timeout=10)
        sg_feed = feedparser.parse(sg_res.content)
        
        tech_query = urllib.parse.quote('("CPO" OR "Nvidia") AND ("Market" OR "Risk")')
        tech_res = requests.get(f"https://gnews.io/api/v4/search?q={tech_query}&lang=en&max=3&apikey={GNEWS_KEY}", timeout=10).json()
        
        geo_query = urllib.parse.quote('("Geopolitics" OR "War") AND ("Oil")')
        geo_res = requests.get(f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=2&apikey={GNEWS_KEY}", timeout=10).json()

        return sg_feed.entries[:3], tech_res.get('articles', []), geo_res.get('articles', [])
    except:
        return [], [], []

# 5. 【核心回溯】极致兼容版 TradingView 挂件
# 使用 Mini Symbol Overview，这种格式几乎不会触发“小鬼”报错
def render_tv_widget(symbol):
    render_code = f"""
    <div class="tradingview-widget-container" style="height:220px;">
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
      {{
        "symbol": "{symbol}",
        "width": "100%",
        "height": 220,
        "locale": "zh_CN",
        "dateRange": "12M",
        "colorTheme": "light",
        "trendLineColor": "rgba(41, 98, 255, 1)",
        "underLineColor": "rgba(41, 98, 255, 0.3)",
        "isTransparent": false,
        "autosize": true
      }}
      </script>
    </div>"""
    components.html(render_code, height=230)

# --- 界面分栏 ---
tab1, tab2, tab3 = st.tabs(["🧠 首席宏观研判", "📈 实时全球仪表盘", "🛡️ 行业风险穿透"])

with tab1:
    if st.sidebar.button("🚀 启动深度宏观研判"):
        with st.spinner("策略师扫描中..."):
            sg, tech, geo = fetch_macro_sector_data()
            st.markdown(get_ai_analysis("macro", sg, tech, geo))

with tab2:
    st.subheader("📊 实时行情监测 (TradingView 兼容版)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**✨ 伦敦金 (XAUUSD)**")
        render_tv_widget("OANDA:XAUUSD")
        st.write("**🛢️ 布伦特原油 (UKOIL)**")
        render_tv_widget("TVC:UKOIL")

    with col2:
        st.write("**🇺🇸 10年美债 (US10Y)**")
        render_tv_widget("TVC:US10Y")
        st.write("**🇺🇸 纳斯达克 (NDX)**")
        render_tv_widget("NASDAQ:NDX")

    with col3:
        st.write("**🇨🇳 沪深 300 (CSI300)**")
        render_tv_widget("SSE:000300")
        st.write("**🇸🇬 海峡时报指数 (STI)**")
        render_tv_widget("FTSE:STI")

    st.write("**💵 美元/新元 (USDSGD)**")
    render_tv_widget("FX_IDC:USDSGD")

with tab3:
    if st.sidebar.button("🔍 穿透行业风险"):
        with st.spinner("风险定价评估中..."):
            sg, tech, geo = fetch_macro_sector_data()
            st.markdown(get_ai_analysis("sector", sg, tech, geo))

st.markdown("---")
st.caption("Macro Alpha v4.5 | 回溯 TradingView 架构 | 实习决策辅助")
