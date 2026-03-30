import streamlit as st
import requests
import feedparser
import google.generativeai as genai
import google.api_core.exceptions
import time

# 1. 页面基本设置
st.set_page_config(page_title="Macro Intern Agent", layout="wide", page_icon="📊")
st.title("📊 宏观经济每日分析 Agent")

# 2. 配置与 Secrets 读取
try:
    AV_KEY = st.secrets.get("AV_KEY")
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    # 兼容两种可能的 Key 命名
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")

    if not all([AV_KEY, GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整，请检查 .streamlit/secrets.toml 或 Streamlit Cloud 设置")
        st.stop()
except Exception as e:
    st.error(f"❌ 读取配置时出错: {e}")
    st.stop()

# 3. 初始化 Gemini (使用你列表里最稳健的模型)
genai.configure(api_key=GEMINI_KEY)
# 建议使用 gemini-flash-latest，它会自动指向当前最稳定的 Flash 版本
MODEL_NAME = 'gemini-flash-latest' 
model = genai.GenerativeModel(MODEL_NAME)

# 4. 数据抓取函数
def fetch_data():
    mas_news = []
    global_news = []
    diag_info = [] # 用于存储诊断信息

    # --- 1. 深度排查 MAS ---
    mas_urls = [
        "https://www.mas.gov.sg/rss/news",
        "https://www.mas.gov.sg/rss/media-releases",
        "https://www.mas.gov.sg/rss/monetary-policy"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/rss+xml, application/xml, text/xml, */*'
    }

    for url in mas_urls:
        try:
            res = requests.get(url, headers=headers, timeout=15)
            status = res.status_code
            content_snippet = res.text[:150].replace('\n', ' ')
            
            # 诊断记录
            diag_info.append(f"URL: {url} | Status: {status} | Snippet: {content_snippet}")

            if status == 200:
                feed = feedparser.parse(res.content)
                if feed.entries:
                    mas_news = [f"- {e.title}" for e in feed.entries[:3]]
                    break # 抓到一个有效的就跳出
        except Exception as e:
            diag_info.append(f"URL: {url} | Error: {str(e)}")

    # 如果所有 RSS 都挂了，最后尝试直接爬网页 (备选方案)
    if not mas_news:
        diag_info.append("所有 RSS 链接均未返回有效数据，尝试备选方案...")
        mas_news = ["MAS 官网 RSS 接口当前不可用，请稍后重试。"]

    # --- 2. 抓取 GNews (保持不变) ---
    try:
        g_url = f"https://gnews.io/api/v4/search?q=macroeconomics&lang=en&max=3&apikey={GNEWS_KEY}"
        g_res = requests.get(g_url, timeout=10).json()
        global_news = [f"- {a['title']} ({a['source']['name']})" for a in g_res.get('articles', [])]
    except Exception as e:
        global_news = [f"GNews 获取失败: {e}"]

    # 在界面上展示诊断信息（仅供排查时使用，稳定后可删除）
    with st.expander("🛠️ MAS 抓取诊断日志 (排查完可删除)"):
        for info in diag_info:
            st.write(info)

    return mas_news, global_news

# 5. 侧边栏与主触发逻辑
st.sidebar.header("控制面板")
if st.sidebar.button("💡 生成今日宏观分析"):
    with st.spinner("Agent 正在阅读新闻并结合 CFA 框架进行分析..."):
        mas, gbl = fetch_data()
        
        # 展示原始数据
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🇸🇬 新加坡金管局动态")
            for item in mas: st.write(item)
        with col2:
            st.subheader("🌎 全球宏观要闻")
            for item in gbl: st.write(item)
            
        st.divider()
        st.subheader(f"🧠 Agent 深度解读 ({MODEL_NAME})")
        
        # 构建 Prompt
        prompt = f"""
        你是一名资深的金融宏观分析师。请基于以下最新资讯：
        新加坡动态：{mas}
        全球要闻：{gbl}
        
        请按照以下结构给出专业分析（使用中文）：
        1. 核心趋势总结 (Executive Summary)
        2. 潜在的市场风险或机遇 (请结合 CFA Level I 中的货币政策、财政政策或通胀框架进行分析)
        3. 实习生建议：在今天的早会上，我应该重点关注哪些指标或话题？
        """
        
        # 6. 带错误处理的 AI 生成请求
        try:
            response = model.generate_content(prompt)
            if response.text:
                st.markdown(response.text)
            else:
                st.warning("模型返回了空内容，请重试。")
                
        except google.api_core.exceptions.ResourceExhausted:
            st.error("⚠️ [API 额度已耗尽] 免费版 Gemini 每分钟请求次数有限，请等待约 60 秒后再点击。")
        except google.api_core.exceptions.InvalidArgument:
            st.error("⚠️ [参数错误] 可能是模型名称无效，请检查代码中的 MODEL_NAME。")
        except Exception as e:
            st.error(f"⚠️ 发生未知错误: {e}")

else:
    st.info("👋 你好！点击左侧按钮，让 Agent 为你汇总今日宏观资讯并进行 CFA 视角的解读。")
    st.caption(f"当前运行模型: {MODEL_NAME}")

# 页脚
st.markdown("---")
st.caption("Powered by Gemini API | Designed for Business Analytics Interns")
