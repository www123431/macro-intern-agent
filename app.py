import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import google.api_core.exceptions
import urllib.parse
import streamlit.components.v1 as components

# 1. 页面设置
st.set_page_config(page_title="Macro Alpha Terminal", layout="wide", page_icon="🌐")
st.title("🏛️ Macro Alpha: 全球地缘宏观决策终端")

# 2. 安全读取 Secrets
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置错误: {e}")
    st.stop()

# 3. 初始化 Gemini
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# 4. 数据抓取
def fetch_enhanced_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Inflation")')
        sg_rss = f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en"
        sg_res = requests.get(sg_rss, headers=headers, timeout=10)
        sg_feed = feedparser.parse(sg_res.content)
        sg_news = [f"- {e.title} ({e.source.get('title', 'Media')})" for e in sg_feed.entries[:3]]

        geo_query = urllib.parse.quote('("Geopolitical Risk" OR "Conflict") AND ("Crude Oil" OR "Shipping")')
        geo_url = f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=3&apikey={GNEWS_KEY}"
        geo_res = requests.get(geo_url, timeout=10).json()
        geo_news = [f"- {a['title']} ({a['source']['name']})" for a in geo_res.get('articles', [])]

        gl_news = ["- 美联储政策路径分析 (Reuters)", "- 全球流动性报告 (Bloomberg)"]
        return sg_news, geo_news, gl_news
    except:
        return ["数据连接中..."], ["地缘分析待更新"], ["全球动态同步中"]

# 5. 【核心修复】终极兼容图表函数 (Mini Chart 模式)
def render_mini_chart(symbol, title):
    # 使用 Mini Chart 挂件，这种挂件的权限限制最少
    render_code = f"""
    <div class="tradingview-widget-container">
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
        "underLineBottomColor": "rgba(41, 98, 255, 0)",
        "isTransparent": false,
        "autosize": false,
        "largeChartUrl": ""
      }}
      </script>
    </div>
    """
    components.html(render_code, height=230)

# --- 布局 ---
tab1, tab2 = st.tabs(["🧠 首席研判 Agent", "📈 实时仪表盘"])

with tab1:
    if st.sidebar.button("🚀 启动深度宏观研判"):
        with st.spinner("策略师正在复盘..."):
            sg, geo, gl = fetch_enhanced_data()
            st.write("### 实时研判摘要")
            st.caption(f"本地资讯：{sg[0] if sg else 'N/A'}")
            st.markdown("---")
            # 简化版 AI 输出
            response = model.generate_content(f"分析：{sg}, {geo}")
            st.markdown(response.text)

with tab2:
    st.subheader("📊 全球宏观核心指标 (极速兼容版)")
    
    # 使用 3 列布局，每行 3 个小图，更像交易终端
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write("**🛢️ 布伦特原油**")
        render_mini_chart("TVC:UKOIL", "Brent")
        st.write("**🇸🇬 海峡指数**")
        render_mini_chart("FTX:STI", "STI")

    with c2:
        st.write("**🇺🇸 10年美债**")
        render_mini_chart("TVC:US10Y", "US10Y")
        st.write("**💵 美元/新元**")
        render_mini_chart("FX_IDC:USDSGD", "USDSGD")

    with c3:
        st.write("**🇺🇸 纳斯达克**")
        render_mini_chart("NASDAQ:NDX", "NDX")
        st.write("**🇨🇳 沪深 300**")
        render_mini_chart("SSE:000300", "CSI300")

st.markdown("---")
st.caption("注：如果图表仍不显示，请检查您的浏览器是否开启了‘防止跨站追踪’或广告拦截插件。")
