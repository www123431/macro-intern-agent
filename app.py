import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import google.api_core.exceptions
import urllib.parse
import streamlit.components.v1 as components

# 1. 页面基本设置
st.set_page_config(page_title="Macro Alpha Terminal", layout="wide", page_icon="🌐")
st.title("🏛️ Macro Alpha: 全球地缘宏观决策终端")

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
MODEL_NAME = 'gemini-flash-latest' 
model = genai.GenerativeModel(MODEL_NAME)

# 4. 深度数据抓取函数
def fetch_enhanced_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # A. 新加坡：定向搜索 MAS 与 顶级媒体
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Economic Outlook" OR "Inflation")')
        sg_rss = f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en"
        sg_res = requests.get(sg_rss, headers=headers, timeout=10)
        sg_feed = feedparser.parse(sg_res.content)
        sg_news = [f"- {e.title} ({e.source.get('title', 'Media')})" for e in sg_feed.entries[:3]]

        # B. 地缘政治：战争、制裁、能源与供应链
        geo_query = urllib.parse.quote('("Geopolitical Risk" OR "Conflict" OR "Sanctions") AND ("Crude Oil" OR "Supply Chain" OR "Shipping")')
        geo_url = f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=3&apikey={GNEWS_KEY}"
        geo_res = requests.get(geo_url, timeout=10).json()
        geo_news = [f"- {a['title']} ({a['source']['name']})" for a in geo_res.get('articles', [])]

        # C. 全球宏观：美联储动态与流动性
        global_query = urllib.parse.quote('("Federal Reserve" OR "Treasury Yield") AND ("Macro Strategy" OR "Recession")')
        global_url = f"https://gnews.io/api/v4/search?q={global_query}&lang=en&max=2&apikey={GNEWS_KEY}"
        gl_res = requests.get(global_url, timeout=10).json()
        gl_news = [f"- {a['title']} ({a['source']['name']})" for a in gl_res.get('articles', [])]

        return sg_news, geo_news, gl_news
    except Exception as e:
        return ["数据抓取超时"], ["地缘数据不可用"], ["全球数据不可用"]

# 5. TradingView 图表挂件函数
def render_tv_chart(symbol, theme="light"):
    render_code = f"""
    <div class="tradingview-widget-container" style="height:400px;">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.MediumWidget({{
        "symbols": [["{symbol}", "{symbol}"]],
        "chartOnly": false,
        "width": "100%",
        "height": 400,
        "locale": "zh_CN",
        "colorTheme": "{theme}",
        "gridLineColor": "rgba(240, 243, 250, 0)",
        "trendLineColor": "#2962FF",
        "underLineColor": "rgba(41, 98, 255, 0.3)",
        "isTransparent": false,
        "autosize": true,
        "showFloatingTooltip": true
      }});
      </script>
    </div>
    """
    components.html(render_code, height=410)

# --- 页面布局 ---
tab1, tab2 = st.tabs(["🧠 首席研判 Agent", "📈 实时全球仪表盘"])

# TAB 1: AI 研判
with tab1:
    st.sidebar.header("控制台")
    if st.sidebar.button("🚀 启动深度宏观研判"):
        with st.spinner("正在解析全球政经关联图谱..."):
            sg, geo, gl = fetch_enhanced_data()
            
            c1, c2, c3 = st.columns(3)
            with c1: 
                st.info("🇸🇬 新加坡变量")
                for i in sg: st.caption(i)
            with c2: 
                st.warning("⚔️ 地缘博弈")
                for i in geo: st.caption(i)
            with c3: 
                st.success("🌎 全球流动性")
                for i in gl: st.caption(i)
            
            st.divider()
            
            # --- 资深策略师 Prompt ---
            prompt = f"""
            你是一名常驻新加坡的全球宏观首席策略师。请基于以下资讯进行“政经一体化”分析：
            新加坡动态：{sg}
            地缘局势：{geo}
            全球环境：{gl}

            要求：
            1. 分析地缘冲突如何通过能源价格和海运成本传导至新加坡的 CPI。
            2. 研判美联储政策与地缘风险溢价对新元汇率（S$NEER）的挤压。
            3. 指出当前市场最严重的“认知偏差”或“预期差”。
            4. 为实习生准备一句穿透性的早会发言（冷峻、洞见）。
            """
            
            try:
                response = model.generate_content(prompt)
                st.markdown("### 🎙️ 首席策略师深度研判")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"分析中断: {e}")
    else:
        st.write("👈 请点击侧边栏按钮开始分析。")

# TAB 2: 实时走势
with tab2:
    st.subheader("📊 全球核心宏观指标 (实时)")
    
    # 第一行：避险与能源（地缘政治最敏感指标）
    row1_a, row1_b = st.columns(2)
    with row1_a:
        st.markdown("**🛢️ 布伦特原油 (Brent Oil)** - 地缘政治的晴雨表")
        render_tv_chart("TVC:UKOIL")
    with row1_b:
        st.markdown("**🇺🇸 10年期美债收益率 (US10Y)** - 全球资产定价之锚")
        render_tv_chart("TVC:US10Y")
    
    st.divider()
    
    # 第二行：股市资产
    row2_a, row2_b = st.columns(2)
    with row2_a:
        st.markdown("**🇨🇳 沪深 300 (CSI 300)**")
        render_tv_chart("SSE:000300")
    with row2_b:
        st.markdown("**🇺🇸 纳斯达克 100 (Nasdaq 100)**")
        render_tv_chart("NASDAQ:NDX")

    st.divider()

    # 第三行：新加坡专项
    row3_a, row3_b = st.columns(2)
    with row3_a:
        st.markdown("**🇸🇬 新加坡海峡时报指数 (STI)**")
        render_tv_chart("FTX:STI")
    with row3_b:
        st.markdown("**💵 美元/新元汇率 (USD/SGD)** - MAS 政策的核心观测点")
        render_tv_chart("FX_IDC:USDSGD")

# 页脚
st.markdown("---")
st.caption("Macro Alpha Agent v3.0 | 实时数据由 TradingView 提供 | 核心大脑 Gemini Flash")
