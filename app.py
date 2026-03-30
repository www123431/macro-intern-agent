import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import google.api_core.exceptions
import urllib.parse
import streamlit.components.v1 as components
from docx import Document
from docx.shared import Inches
from io import BytesIO
import datetime

# --- 1. 页面配置 ---
st.set_page_config(page_title="Macro Alpha Pro Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 专业基金投研终端 (v5.0)")

# --- 2. 安全读取 Secrets ---
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置读取失败: {e}")
    st.stop()

# --- 3. 初始化 Gemini ---
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# --- 4. 专业报告生成函数 (The Game Changer) ---
def generate_report(analysis_text, report_type="Macro"):
    doc = Document()
    doc.add_heading(f'Macro Alpha - {report_type} Investment Memo', 0)
    
    date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    doc.add_paragraph(f'Generated on: {date_str}')
    doc.add_paragraph('Analyst: Macro Alpha AI Agent')
    
    doc.add_heading('Executive Summary', level=1)
    doc.add_paragraph(analysis_text)
    
    doc.add_heading('Risk Disclaimer', level=2)
    doc.add_paragraph('This report is generated for internal reference only.')
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- 5. 核心分析函数 (带缓存) ---
@st.cache_data(ttl=600)
def get_ai_analysis(prompt_type, sg_data, tech_data, geo_data):
    try:
        # 更加专业的 Prompt 注入
        if prompt_type == "macro":
            full_prompt = f"作为常驻新加坡的首席策略师，基于以下资讯进行深度穿透，重点研判新元汇率(S$NEER)和通胀走势：{sg_data}, {geo_data}"
        else:
            full_prompt = f"分析 CPO、AI算力、地缘敏感商品及新加坡蓝筹。按【板块】-【建议】-【逻辑】输出：{tech_data} | {geo_data}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"ERROR: {str(e)}"

# --- 6. 数据抓取 ---
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

# --- 7. TradingView 渲染器 (保持 v4.2 稳定版) ---
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
        "underLineColor": "rgba(41, 98, 255, 0.3)", "isTransparent": false, "autosize": true
      }});
      </script>
    </div>"""
    components.html(render_code, height=360)

# --- 8. 界面逻辑 ---
tab1, tab2, tab3 = st.tabs(["🧠 首席宏观研判", "📈 实时全球仪表盘", "🛡️ 行业风险穿透"])

# 侧边栏：操作面板
st.sidebar.header("🕹️ 控制中心")
run_macro = st.sidebar.button("🚀 启动深度研判")
run_sector = st.sidebar.button("🔍 穿透行业风险")

with tab1:
    if run_macro:
        with st.spinner("策略师正在扫描全球图谱..."):
            sg, tech, geo = fetch_macro_sector_data()
            result = get_ai_analysis("macro", sg, tech, geo)
            st.markdown(result)
            
            # 专业动作：提供报告下载
            report_buf = generate_report(result, "Macro Analysis")
            st.download_button(
                label="📥 下载专业投资备忘录 (Word)",
                data=report_buf,
                file_name=f"Macro_Alpha_Report_{datetime.date.today()}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

with tab2:
    st.subheader("📊 实时行情监测")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**✨ 伦敦金 (XAUUSD)**")
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

with tab3:
    if run_sector:
        with st.spinner("正在评估行业溢价..."):
            sg, tech, geo = fetch_macro_sector_data()
            result = get_ai_analysis("sector", sg, tech, geo)
            st.markdown(result)
            
            report_buf = generate_report(result, "Sector Strategy")
            st.sidebar.download_button(
                label="📥 下载行业策略简报",
                data=report_buf,
                file_name=f"Sector_Strategy_{datetime.date.today()}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

st.markdown("---")
st.caption("Macro Alpha v5.0 | 自动化研报系统已启用 | 实习辅助决策")
