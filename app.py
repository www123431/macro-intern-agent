import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import google.api_core.exceptions
import urllib.parse

# 1. 页面基本设置
st.set_page_config(page_title="Macro Alpha Agent", layout="wide", page_icon="📈")
st.title("🧪 Macro Alpha: 资深金融时政分析 Agent")

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
MODEL_NAME = 'gemini-flash-latest' 
model = genai.GenerativeModel(MODEL_NAME)

# 4. 深度数据抓取 (混合官方与媒体视角)
def fetch_alpha_data():
    try:
        # 新加坡：混合官方与顶级媒体，搜索更具冲击力的关键词
        # 搜索：(MAS官网 OR 海峡时报) + (货币政策 OR 经济前景 OR 银行业压力)
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Economic Outlook" OR "Inflation" OR "Banking")')
        sg_rss = f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en"
        
        # 全球：侧重宏观博弈
        global_query = urllib.parse.quote('("Federal Reserve" OR "ECB" OR "China Stimulus") AND "Macro Strategy"')
        global_url = f"https://gnews.io/api/v4/search?q={global_query}&lang=en&max=3&apikey={GNEWS_KEY}"

        # 执行抓取
        headers = {'User-Agent': 'Mozilla/5.0'}
        sg_res = requests.get(sg_rss, headers=headers, timeout=10)
        sg_feed = feedparser.parse(sg_res.content)
        
        gl_res = requests.get(global_url, timeout=10).json()
        
        sg_news = [f"- {e.title} ({e.source.get('title', 'Media')})" for e in sg_feed.entries[:4]]
        gl_news = [f"- {a['title']} ({a['source']['name']})" for a in gl_res.get('articles', [])]
        
        return sg_news, gl_news
    except Exception as e:
        return [f"抓取异常: {e}"], ["获取失败"]

# 5. 主逻辑
if st.sidebar.button("🚀 开启深度宏观研判"):
    with st.spinner("正在构建宏观博弈模型..."):
        sg_data, gl_data = fetch_alpha_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🇸🇬 新加坡核心变量")
            for i in sg_data: st.caption(i)
        with col2:
            st.subheader("🌎 全球宏观博弈")
            for i in gl_data: st.caption(i)
            
        st.divider()
        
        # --- 资深分析师 Prompt ---
        alpha_prompt = f"""
        你是一名拥有20年经验的全球宏观首席策略师。请基于以下资讯进行穿透式分析：
        新加坡变量：{sg_data}
        全球环境：{gl_data}

        请展现专业时政分析家的素养，拒绝教科书式的陈述，按以下逻辑输出：

        一、 【破译信号】：不要复述新闻，请指出这些资讯背后的“预期差”在哪里？哪些是干扰杂音，哪些是真正的深层变量？
        二、 【全球传导链】：分析全球宏观环境（如美联储或大国政策）如何通过利率、资本流向或贸易条件，对新加坡这个小型开放经济体产生“非线性”影响？
        三、 【政策博弈】：基于当前动态，推测新加坡金管局（MAS）下一次政策窗口期的博弈重心是什么？（是防通胀的韧性，还是保增长的压力？）
        四、 【资深建议】：如果今天你要在投行内部早会上发言，请给出一句能够穿透迷雾的核心结论。

        注意：语气要冷峻、专业、充满洞见。
        """
        
        try:
            response = model.generate_content(alpha_prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error(f"分析失败: {e}")
else:
    st.info("点击左侧按钮，从资深策略师视角研判全球与本地宏观局势。")

st.markdown("---")
st.caption("Macro Alpha Agent | 高级金融时政研判工具")
