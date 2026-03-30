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

# --- 1. 页面基本配置 ---
st.set_page_config(page_title="Macro Alpha Pro Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 全球投研与自动化简报终端")

# --- 2. 安全读取 Secrets ---
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整，请在 Streamlit Cloud 后台检查 GNEWS_KEY 和 GEMINI_API_KEY")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置文件读取失败: {e}")
    st.stop()

# --- 3. 初始化 Gemini (使用你提供的新 API Key) ---
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-3-flash') # 使用最新的 Flash 模型

# --- 4. 自动化 Word 报告生成函数 ---
def generate_docx_report(content, title="Investment Memo"):
    doc = Document()
    doc.add_heading('Macro Alpha Intelligence', 0)
    doc.add_heading(title, level=1)
    
    # 添加报告元数据
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    p = doc.add_paragraph()
    p.add_run(f'Report Date: {now}\n').bold = True
    p.add_run('Classification: Internal / Confidential\n').italic = True
    p.add_run(f'Analyst: Macro Alpha AI Agent')

    doc.add_heading('Market Analysis & Strategic Insights', level=2)
    # 处理换行符，确保在 Word 中显示正常
    for line in content.split('\n'):
        doc.add_paragraph(line)
    
    doc.add_page_break()
    doc.add_paragraph("Disclaimer: This AI-generated report is for professional reference only.")
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- 5. AI 生成逻辑 (带 1 小时缓存与限流重试) ---
@st.cache_data(ttl=3600) 
def get_ai_analysis(prompt_type, sg_data, tech_data, geo_data):
    for attempt in range(2): 
        try:
            if prompt_type == "macro":
                full_prompt = f"""
                你是一名常驻新加坡的资深宏观策略师。请基于以下资讯进行深度穿透：
                1. 研判对新加坡 CPI 和 MAS 货币政策的潜在影响。
                2. 分析新元汇率(S$NEER)的波动预期。
                资讯：{sg_data}, {geo_data}
                请以专业、冷静、有洞察力的语气回答。
                """
            else:
                full_prompt = f"""
                你是一名行业分析师。分析以下板块的投资机会与风险：
                CPO与光模块、AI算力、地缘敏感商品、新加坡本地蓝筹。
                给出明确的建议（买入/退出/持有）及逻辑支撑。
                资讯：{tech_data} | {geo_data}
                """
            
            response = model.generate_content(full_prompt)
            return response.text
        except google.api_core.exceptions.ResourceExhausted:
            if attempt == 0:
                st.warning("⚠️ 触发 API 频率限制，正在为您自动重试（等待 30 秒）...")
                time.sleep(30)
                continue
            return "❌ API 配额暂时耗尽，请稍后再试。"
        except Exception as e:
            return f"⚠️ 发生错误: {str(e)}"

# --- 6. 增强版数据抓取 ---
def fetch_macro_sector_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # 新加坡相关
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Inflation")')
        sg_res = requests.get(f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en", headers=headers, timeout=10)
        sg_feed = feedparser.parse(sg_res.content)
        
        # 科技与 AI 相关
        tech_query = urllib.parse.quote('("CPO" OR "Nvidia" OR "AI infrastructure") AND ("Market" OR "Risk")')
        tech_res = requests.get(f"https://gnews.io/api/v4/search?q={tech_query}&lang=en&max=3&apikey={GNEWS_KEY}", timeout=10).json()
        
        # 地缘与大宗
        geo_query = urllib.parse.quote('("Geopolitics" OR "Oil Price") AND ("Supply Chain")')
        geo_res = requests.get(f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=2&apikey={GNEWS_KEY}", timeout=10).json()

        return sg_feed.entries[:3], tech_res.get('articles', []), geo_res.get('articles', [])
    except:
        return [], [], []

# --- 7. TradingView 挂件渲染 (回归 v4.2 稳定逻辑) ---
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

# --- 8. 界面 Tab 布局 ---
tab1, tab2, tab3 = st.tabs(["🧠 首席宏观研判", "📈 实时全球仪表盘", "🛡️ 行业风险穿透"])

# --- TAB 1: 宏观分析 ---
with tab1:
    st.info("💡 建议在开盘前运行，生成今日宏观备忘录。")
    if st.sidebar.button("🚀 启动深度宏观研判"):
        with st.spinner("策略师正在扫描全球图谱..."):
            sg, tech, geo = fetch_macro_sector_data()
            result = get_ai_analysis("macro", sg, tech, geo)
            st.markdown(result)
            
            # 生成并提供 Word 下载
            report_data = generate_docx_report(result, "Morning Macro Intelligence Memo")
            st.download_button(
                label="📥 点击下载：今日宏观投资备忘录 (Word)",
                data=report_data,
                file_name=f"Macro_Report_{datetime.date.today()}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

# --- TAB 2: 仪表盘 (3列经典布局) ---
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

# --- TAB 3: 行业穿透 ---
with tab3:
    st.subheader("🧭 板块风险与操作策略")
    if st.sidebar.button("🔍 穿透行业风险"):
        with st.spinner("正在评估板块溢价..."):
            sg, tech, geo = fetch_macro_sector_data()
            result = get_ai_analysis("sector", sg, tech, geo)
            st.markdown(result)
            
            # 生成行业报告
            report_data = generate_docx_report(result, "Sector Strategy Analysis")
            st.download_button(
                label="📥 点击下载：行业风险策略报告 (Word)",
                data=report_data,
                file_name=f"Sector_Strategy_{datetime.date.today()}.docx"
            )

st.markdown("---")
st.caption(f"Macro Alpha v5.1 | 系统运行正常 | 当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
