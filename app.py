import streamlit as st
import requests
import feedparser
import google.generativeai as genai

# 1. 页面基本设置
st.set_page_config(page_title="Macro Intern Agent", layout="wide")
st.title("📊 宏观经济每日分析 Agent")

# --- 修改后的第 2 & 3 部分 ---
try:
    # 自动兼容你可能写的两种名字
    AV_KEY = st.secrets.get("AV_KEY")
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    # 这里优先读取你刚才说的 GEMINI_API_KEY
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")

    if not all([AV_KEY, GNEWS_KEY, GEMINI_KEY]):
        st.error("Secrets 配置不完整，请检查 .streamlit/secrets.toml")
        st.stop()
except Exception as e:
    st.error(f"读取配置时出错: {e}")
    st.stop()

# 配置 Gemini
genai.configure(api_key=GEMINI_KEY)
# 使用最新的 1.5 系列模型
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 数据抓取函数 ---
def fetch_data():
    # A. MAS RSS (新加坡本地)
    mas_feed = feedparser.parse("https://www.mas.gov.sg/rss/news-and-publications")
    mas_news = [f"- {e.title}" for e in mas_feed.entries[:3]]
    
    # B. GNews (全球宏观)
    g_url = f"https://gnews.io/api/v4/search?q=macroeconomics&lang=en&max=3&apikey={GNEWS_KEY}"
    g_res = requests.get(g_url).json()
    global_news = [f"- {a['title']} ({a['source']['name']})" for a in g_res.get('articles', [])]
    
    return mas_news, global_news

# --- 侧边栏与触发 ---
if st.sidebar.button("💡 生成今日宏观分析"):
    with st.spinner("Agent 正在阅读新闻并结合 CFA 框架进行分析..."):
        mas, gbl = fetch_data()
        
        # 展示原始数据
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🇸🇬 新加坡金管局动态")
            st.write("\n".join(mas))
        with col2:
            st.subheader("🌎 全球宏观要闻")
            st.write("\n".join(gbl))
            
        # AI 分析部分
        st.divider()
        st.subheader("🧠 Agent 深度解读 (CFA Level I Context)")
        
        prompt = f"""
        你是一名资深的金融宏观分析师。请基于以下最新资讯：
        新加坡动态：{mas}
        全球要闻：{gbl}
        
        请按照以下结构给出分析：
        1. 核心趋势总结 (Executive Summary)
        2. 潜在的市场风险或机遇 (基于 CFA 货币政策与通胀框架)
        3. 实习生建议：在今天的早会上，我应该重点关注哪些指标？
        """
        
        response = model.generate_content(prompt)
        st.markdown(response.text)
else:
    st.info("点击左侧按钮，让 Agent 开始工作。")
