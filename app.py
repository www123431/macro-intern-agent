import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import google.api_core.exceptions
import urllib.parse
import streamlit.components.v1 as components
import time

# 1. 页面设置
st.set_page_config(page_title="Macro Alpha Pro Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 全球宏观与行业策略终端")

# 2. 安全读取 Secrets
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整")
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
            full_prompt = f"作为首席策略师，分析：{sg_data}, {geo_data}。重点研判新加坡CPI与新元汇率。"
        else:
            full_prompt = f"""
            分析以下板块并给建议（买入/退出/持有）：
            1. CPO与光模块 | 2. AI算力与半导体 | 3. 地缘敏感大宗商品 | 4. 新加坡本地蓝筹
            资讯：{tech_data} | {geo_data}
            """
        response = model.generate_content(full_prompt)
        return response.text
    except google.api_core.exceptions.ResourceExhausted:
        return "ERROR_RATE_LIMIT"
    except Exception as e:
        return f"ERROR_UNKNOWN: {str(e)}"

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

# 5. 图表渲染
def render_tv_medium_widget(symbol, title):
    render_code = f"""
    <div class="tradingview-widget-container" style="height:350px;">
      <div id="tradingview_{symbol}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.MediumWidget({{
        "symbols": [["{title}", "{symbol}|12M"]],
        "chartOnly": false, "width": "100%", "height": 350, "locale": "zh_CN", "colorTheme": "light",
        "gridLineColor": "rgba(240, 243, 250, 0)", "trendLineColor": "#2962FF",
        "underLineColor": "rgba(41, 98, 255, 0.3)", "isTransparent": false, "autosize": true, "showFloatingTooltip": true
      }});
      </script>
    </div>"""
    components.html(render_code, height=360)

# --- 界面分栏 ---
tab1, tab2, tab3 = st.tabs(["🧠 首席宏观研判", "📈 实时全球仪表盘", "🛡️ 行业风险穿透"])

# TAB 1: 宏观研判
with tab1:
    if st.sidebar.button("🚀 启动深度宏观研判"):
        with st.spinner("策略师正在扫描全球图谱..."):
            sg, tech, geo = fetch_macro_sector_data()
            result = get_ai_analysis("macro", sg, tech, geo)
            if result == "ERROR_RATE_LIMIT":
                st.error("⚠️ API 调用过快，请稍后。")
            else:
                st.markdown(result)

# TAB 2: 仪表盘 (已更新 3 列布局，加入伦敦金)
with tab2:
    st.subheader("📊 实时行情监测")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**✨ 伦敦金现货 (XAUUSD)**")
        render_tv_medium_widget("OANDA:XAUUSD", "Gold")
        st.write("**🛢️ 布伦特原油 (UKOIL)**")
        render_tv_medium_widget("TVC:UKOIL", "Brent Oil")
        
    with col2:
        st.write("**🇺🇸 10年美债 (US10Y)**")
        render_tv_medium_widget("TVC:US10Y", "US 10Y Yield")
        st.write("**🇺🇸 纳斯达克 (NDX)**")
        render_tv_medium_widget("NASDAQ:NDX", "Nasdaq 100")
        
    with col3:
        st.write("**🇨🇳 沪深 300 (CSI300)**")
        render_tv_medium_widget("SSE:000300", "CSI 300")
        st.write("**💵 美元/新元 (USDSGD)**")
        render_tv_medium_widget("FX_IDC:USDSGD", "USD / SGD")

# TAB 3: 行业风险穿透
with tab3:
    st.subheader("🧭 行业板块风险提示与操作策略")
    if st.sidebar.button("🔍 穿透行业风险"):
        with st.spinner("评估中..."):
            sg, tech, geo = fetch_macro_sector_data()
            result = get_ai_analysis("sector", sg, tech, geo)
            if result == "ERROR_RATE_LIMIT":
                st.error("⚠️ API 频率超限。")
            else:
                st.markdown(result)

st.markdown("---")
st.caption("Macro Alpha v4.2 | 伦敦金实时接入 | 实习决策辅助")
