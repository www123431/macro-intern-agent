import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import google.api_core.exceptions
import urllib.parse
import time

# 1. 页面设置
st.set_page_config(page_title="Macro Alpha Agent v2.0", layout="wide", page_icon="🌐")
st.title("🧪 Macro Alpha: 资深地缘宏观策略分析")

# 2. 安全读取配置
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置读取错误: {e}")
    st.stop()

# 3. 初始化 Gemini (采用最新 Flash)
genai.configure(api_key=GEMINI_KEY)
MODEL_NAME = 'gemini-flash-latest' 
model = genai.GenerativeModel(MODEL_NAME)

# 4. 深度数据抓取 (三维驱动：新加坡、地缘战争、全球博弈)
def fetch_enhanced_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # A. 新加坡：官方 vs 媒体解读
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Inflation" OR "Economic Outlook")')
        sg_rss = f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en"
        sg_res = requests.get(sg_rss, headers=headers, timeout=10)
        sg_feed = feedparser.parse(sg_res.content)
        sg_news = [f"- {e.title} ({e.source.get('title', 'Media')})" for e in sg_feed.entries[:3]]

        # B. 地缘政治：战争局势、能源冲突、供应链武器化
        # 关键词侧重于：冲突、制裁、原油、运输航道
        geo_query = urllib.parse.quote('("Geopolitical Risk" OR "Conflict" OR "Sanctions") AND ("Crude Oil" OR "Supply Chain" OR "Shipping")')
        geo_url = f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=3&apikey={GNEWS_KEY}"
        geo_res = requests.get(geo_url, timeout=10).json()
        geo_news = [f"- {a['title']} ({a['source']['name']})" for a in geo_res.get('articles', [])]

        # C. 全球宏观：央行博弈与预期差
        global_query = urllib.parse.quote('("Federal Reserve" OR "ECB") AND ("Interest Rates" OR "Macro Strategy")')
        global_url = f"https://gnews.io/api/v4/search?q={global_query}&lang=en&max=3&apikey={GNEWS_KEY}"
        gl_res = requests.get(global_url, timeout=10).json()
        gl_news = [f"- {a['title']} ({a['source']['name']})" for a in gl_res.get('articles', [])]

        return sg_news, geo_news, gl_news
    except Exception as e:
        st.error(f"数据抓取失败: {e}")
        return ["暂无SG数据"], ["暂无地缘数据"], ["暂无全球数据"]

# 5. UI 与 核心研判逻辑
st.sidebar.header("Alpha 控制面板")
st.sidebar.markdown("---")
if st.sidebar.button("🚀 启动穿透式研判"):
    with st.spinner("正在构建多维地缘宏观模型..."):
        sg_data, geo_data, gl_data = fetch_enhanced_data()
        
        # 实时资讯分栏展示
        cols = st.columns(3)
        with cols[0]:
            st.subheader("🇸🇬 新加坡变量")
            for i in sg_data: st.caption(i)
        with cols[1]:
            st.subheader("⚔️ 地缘/战争博弈")
            for i in geo_data: st.caption(i)
        with cols[2]:
            st.subheader("🌎 全球流动性")
            for i in gl_data: st.caption(i)
            
        st.divider()
        
        # --- 高级分析师 Prompt (Zoltan Pozsar 风格) ---
        alpha_prompt = f"""
        你是一名顶级宏观策略分析师，擅长剖析“地缘政治-大宗商品-货币政策”的联动效应。
        当前资讯环境：
        - 新加坡：{sg_data}
        - 地缘局势：{geo_data}
        - 全球宏观：{gl_data}

        请展现深度洞察，拒绝平庸，按以下逻辑进行穿透式输出：

        1. 【地缘通胀传导】：地缘冲突（如油价波动、航道风险）是否正在对冲掉央行的加息努力？指出这些冲突如何通过“输入型成本”直接威胁新加坡的通胀目标。
        2. 【风险定价预期差】：目前市场对战争局势的定价是否充分？请指出哪些被大众忽略的“二阶效应”（例如：制裁导致的本币清算体系变化）。
        3. 【新加坡避险地位】：在动荡局势下，分析新加坡作为“避险天堂”产生的资金回流效应。这种流动性对新元汇率（S$NEER）及其金管局政策空间的挤压是什么？
        4. 【策略师金句】：给出一句极具冲击力、适合在投行早会展示的穿透性结论。

        要求：语气冷峻、专业、拒绝术语堆砌，直戳商业本质。使用中文输出。
        """
        
        try:
            response = model.generate_content(alpha_prompt)
            st.markdown("### 🧠 首席策略师深度解读")
            st.markdown(response.text)
        except google.api_core.exceptions.ResourceExhausted:
            st.error("⚠️ API 频率超限，请等候一分钟再试。")
        except Exception as e:
            st.error(f"分析失败: {e}")
else:
    st.info("💡 请点击左侧按钮，从首席策略师视角分析新加坡、全球宏观及地缘战争局势。")

# 页脚
st.markdown("---")
st.caption("Macro Alpha Agent v2.0 | 集成地缘政治风险定价模型 | 专供金融研判使用")
