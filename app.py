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
        st.error("❌ Secrets 配置不完整，请在 Streamlit Cloud 的 Secrets 中检查 GNEWS_KEY 和 GEMINI_API_KEY")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置读取失败: {e}")
    st.stop()

# 3. 初始化 Gemini
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# --- 核心功能：带缓存的 AI 生成函数 ---
@st.cache_data(ttl=600)  # 缓存 10 分钟 (600秒)
def get_ai_analysis(prompt_type, sg_data, tech_data, geo_data):
    try:
        if prompt_type == "macro":
            full_prompt = f"作为常驻新加坡的首席策略师，分析以下宏观资讯并给出针对新加坡CPI和新元汇率的犀利见解。资讯：{sg_data}, {geo_data}"
        else:
            full_prompt = f"""
            你是一名常驻新加坡的首席行业策略师。基于以下最新动态：
            科技动态：{tech_data} | 地缘背景：{geo_data}
            请针对以下板块给出深度解读，并标注投资建议（建议买入/退出/持续持有）：
            1. CPO (共封装光学) 与光模块 (800G/1.6T周期)
            2. AI 算力与半导体 (Nvidia/TSMC及本地封测)
            3. 地缘敏感型大宗商品 (能源/航运)
            4. 新加坡本地蓝筹 (REITs & Banks 高息影响)
            要求：以【板块名称】-【建议动作】-【深度解读】-【风险等级】结构输出。语气冷峻专业。
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
        
        tech_query = urllib.parse.quote('("CPO" OR "Nvidia" OR "AI Computing") AND ("Market" OR "Risk")')
        tech_res = requests.get(f"https://gnews.io/api/v4/search?q={tech_query}&lang=en&max=3&apikey={GNEWS_KEY}", timeout=10).json()
        
        geo_query = urllib.parse.quote('("Geopolitics" OR "War") AND ("Oil" OR "Supply Chain")')
        geo_res = requests.get(f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=2&apikey={GNEWS_KEY}", timeout=10).json()

        return sg_feed.entries[:3], tech_res.get('articles', []), geo_res.get('articles', [])
    except:
        return [], [], []

# 5. 图表渲染
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
            result = get_ai_analysis("macro", sg, tech, geo)
            
            if result == "ERROR_RATE_LIMIT":
                st.error("⚠️ API 调用过快，请 1 分钟后重试。缓存已开启，10 分钟内重复查看不消耗额度。")
            elif "ERROR_UNKNOWN" in result:
                st.error(f"❌ 系统异常: {result}")
            else:
                st.markdown(result)
    else:
        st.info("👈 点击左侧侧边栏按钮启动 AI 宏观分析。结果将自动缓存 10 分钟。")

# TAB 2: 仪表盘
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

# TAB 3: 行业风险穿透
with tab3:
    st.subheader("🧭 行业板块风险提示与操作策略")
    if st.sidebar.button("🔍 穿透行业风险"):
        with st.spinner("正在评估板块溢价与安全边际..."):
            sg, tech, geo = fetch_macro_sector_data()
            result = get_ai_analysis("sector", sg, tech, geo)
            
            if result == "ERROR_RATE_LIMIT":
                st.error("⚠️ API 频率达到上限。请稍等片刻，利用这段时间复盘仪表盘数据。")
            elif "ERROR_UNKNOWN" in result:
                st.error(f"❌ 系统异常: {result}")
            else:
                st.markdown(result)
    else:
        st.write("👈 点击侧边栏按钮获取 CPO、AI 算力等板块的深度建议。")

st.markdown("---")
st.caption("Macro Alpha v4.1 | 已启用智能频率控制与数据持久化缓存 | 实习决策辅助")
