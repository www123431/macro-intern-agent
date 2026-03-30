import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import google.api_core.exceptions
import urllib.parse
import streamlit.components.v1 as components

# 1. 页面设置
st.set_page_config(page_title="Macro Alpha Pro Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 全球宏观与行业策略终端")

# 2. 安全读取 Secrets
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ 配置不完整")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置错误: {e}")
    st.stop()

# 3. 初始化 Gemini
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# 4. 增强版数据抓取
def fetch_macro_sector_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # 抓取新加坡本地与宏观
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Inflation")')
        sg_feed = feedparser.parse(requests.get(f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en", headers=headers).content)
        
        # 抓取 AI 算力与科技板块动态
        tech_query = urllib.parse.quote('("CPO" OR "Nvidia" OR "AI Computing" OR "Semiconductor") AND ("Market Risk" OR "Forecast")')
        tech_res = requests.get(f"https://gnews.io/api/v4/search?q={tech_query}&lang=en&max=3&apikey={GNEWS_KEY}").json()
        
        # 抓取地缘与能源
        geo_query = urllib.parse.quote('("Geopolitics" OR "War") AND ("Oil" OR "Supply Chain")')
        geo_res = requests.get(f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=2&apikey={GNEWS_KEY}").json()

        return sg_feed.entries[:3], tech_res.get('articles', []), geo_res.get('articles', [])
    except:
        return [], [], []

# 5. 图表渲染函数
def render_tv_medium_widget(symbol, title):
    render_code = f"""
    <div class="tradingview-widget-container" style="height:380px;">
      <div id="tradingview_{symbol}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.MediumWidget({{
        "symbols": [["{title}", "{symbol}|12M"]],
        "chartOnly": false, "width": "100%", "height": 380, "locale": "zh_CN", "colorTheme": "light",
        "gridLineColor": "rgba(240, 243, 250, 0)", "trendLineColor": "#2962FF",
        "underLineColor": "rgba(41, 98, 255, 0.3)", "isTransparent": false, "autosize": true, "showFloatingTooltip": true
      }});
      </script>
    </div>"""
    components.html(render_code, height=390)

# --- 界面分栏 ---
tab1, tab2, tab3 = st.tabs(["🧠 首席宏观研判", "📈 实时全球仪表盘", "🛡️ 行业风险穿透"])

# TAB 1: 宏观研判
with tab1:
    if st.sidebar.button("🚀 启动深度宏观研判"):
        with st.spinner("策略师正在扫描全球图谱..."):
            sg, tech, geo = fetch_macro_sector_data()
            prompt = f"作为首席策略师，分析以下宏观资讯并给出针对新加坡CPI和新元汇率的犀利见解：{sg}, {geo}"
            res = model.generate_content(prompt)
            st.markdown(res.text)

# TAB 2: 仪表盘 (保持之前的稳定配置)
with tab2:
    st.subheader("📊 实时行情监测")
    col_l, col_r = st.columns(2)
    with col_l:
        st.write("**🛢️ 布伦特原油 (UKOIL)**")
        render_tv_medium_widget("TVC:UKOIL", "Brent Oil")
        st.write("**🇨🇳 沪深 300 (CSI300)**")
        render_tv_medium_widget("SSE:000300", "CSI 300")
    with col_r:
        st.write("**🇺🇸 10年美债 (US10Y)**")
        render_tv_medium_widget("TVC:US10Y", "US 10Y Yield")
        st.write("**🇺🇸 纳斯达克 (NDX)**")
        render_tv_medium_widget("NASDAQ:NDX", "Nasdaq 100")

# TAB 3: 行业风险穿透 (全新界面)
with tab3:
    st.subheader("🧭 行业板块风险提示与操作策略")
    st.info("基于地缘变动与算力周期，穿透行业底层逻辑。")
    
    if st.sidebar.button("🔍 穿透行业风险"):
        with st.spinner("正在评估板块溢价与安全边际..."):
            sg, tech, geo = fetch_macro_sector_data()
            
            sector_prompt = f"""
            你是一名常驻新加坡的首席行业策略师。基于以下最新动态：
            科技动态：{tech}
            地缘/宏观背景：{geo}

            请针对以下板块给出深度解读，并标注投资建议（建议买入/退出/持续持有）：

            1. **CPO (共封装光学) 与光模块**：
               - 核心逻辑：分析 800G/1.6T 需求对通信板块的重构。
               - 风险提示：技术迭代风险及订单兑现度。
            2. **AI 算力与半导体**：
               - 核心逻辑：关注 Nvidia 及新加坡本地封测链。
               - 策略：面对出口管制与估值高企，给出明确动作。
            3. **地缘敏感型大宗商品**：
               - 核心逻辑：能源及航运板块在战争背景下的防御性。
            4. **新加坡本地蓝筹 (REITs & Banks)**：
               - 核心逻辑：高利率长久化对新加坡本地资产的定价挤压。

            请以【板块名称】-【建议动作】-【深度解读】-【风险等级】的结构输出。
            要求：语气冷峻、数据感强、具有实战意义。
            """
            
            sector_res = model.generate_content(sector_prompt)
            st.markdown(sector_res.text)
    else:
        st.write("👈 请点击侧边栏的“🔍 穿透行业风险”按钮获取详细板块建议。")

st.markdown("---")
st.caption("Macro Alpha v4.0 | 实习辅助工具 | 严禁作为直接投资依据")
