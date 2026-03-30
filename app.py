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
import numpy as np
from scipy.stats import norm

# --- 1. 页面基本配置 ---
st.set_page_config(page_title="Macro Alpha Pro Terminal", layout="wide", page_icon="🏛️")
st.title("🏛️ Macro Alpha: 全球投研与自动化简报终端")

# --- 2. 安全读取 Secrets ---
try:
    GNEWS_KEY = st.secrets.get("GNEWS_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
    if not all([GNEWS_KEY, GEMINI_KEY]):
        st.error("❌ Secrets 配置不完整，请检查后台配置。")
        st.stop()
except Exception as e:
    st.error(f"❌ 配置文件读取失败: {e}")
    st.stop()

# --- 3. 初始化 Gemini ---
genai.configure(api_key=GEMINI_KEY)
# 修正模型名称为最稳定的版本，或根据你的新 Key 调整
MODEL_NAME = 'gemini-1.5-flash' 
model = genai.GenerativeModel(MODEL_NAME)

# --- 4. 自动化 Word 报告生成函数 ---
def generate_docx_report(content, title="Investment Memo"):
    doc = Document()
    doc.add_heading('Macro Alpha Intelligence', 0)
    doc.add_heading(title, level=1)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    p = doc.add_paragraph()
    p.add_run(f'Report Date: {now}\n').bold = True
    p.add_run('Classification: Internal / Confidential\n').italic = True
    p.add_run(f'Analyst: Macro Alpha AI Agent ({MODEL_NAME})')

    doc.add_heading('Market Analysis & Strategic Insights', level=2)
    for line in content.split('\n'):
        doc.add_paragraph(line)
    
    doc.add_page_break()
    doc.add_paragraph("Disclaimer: This AI-generated report is for professional reference only.")
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- 5. 优雅的 AI 生成逻辑 ---
@st.cache_data(ttl=3600) 
def get_ai_analysis(prompt_type, sg_data, tech_data, geo_data):
    for attempt in range(2): 
        try:
            if prompt_type == "macro":
                full_prompt = f"你是一名常驻新加坡的资深宏观策略师。分析对新加坡CPI、MAS政策及新元汇率(S$NEER)的影响。资讯：{sg_data}, {geo_data}"
            else:
                full_prompt = f"分析 CPO、AI算力、地缘敏感商品、新加坡蓝筹的投资机会。资讯：{tech_data} | {geo_data}"
            
            response = model.generate_content(full_prompt)
            return response.text, None # 返回结果和无错误
            
        except google.api_core.exceptions.ResourceExhausted:
            if attempt == 0:
                time.sleep(30)
                continue
            return None, "Quota_Exceeded"
        except google.api_core.exceptions.NotFound:
            return None, "Model_Not_Found"
        except Exception as e:
            return None, str(e)

# --- 6. 数据抓取 ---
def fetch_macro_sector_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        sg_query = urllib.parse.quote('(site:mas.gov.sg OR site:straitstimes.com) ("Monetary Policy" OR "Inflation")')
        sg_res = requests.get(f"https://news.google.com/rss/search?q={sg_query}&hl=en-SG&gl=SG&ceid=SG:en", headers=headers, timeout=10)
        sg_feed = feedparser.parse(sg_res.content)
        
        tech_query = urllib.parse.quote('("CPO" OR "Nvidia" OR "AI") AND ("Market" OR "Risk")')
        tech_res = requests.get(f"https://gnews.io/api/v4/search?q={tech_query}&lang=en&max=3&apikey={GNEWS_KEY}", timeout=10).json()
        
        geo_query = urllib.parse.quote('("Geopolitics" OR "Oil Price") AND ("Supply Chain")')
        geo_res = requests.get(f"https://gnews.io/api/v4/search?q={geo_query}&lang=en&max=2&apikey={GNEWS_KEY}", timeout=10).json()

        return sg_feed.entries[:3], tech_res.get('articles', []), geo_res.get('articles', [])
    except:
        return [], [], []

# --- 7. TradingView 挂件 ---
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
    
# --- 1. 量化计算引擎 (基于 Week 4, 5, 9 课程内容) ---
class QuantEngine:
    @staticmethod
    def black_scholes_analysis(S, K, T, r, sigma, option_type='call'):
        """Week 4 & 5: BSM 模型与 Delta 对冲计算"""
        if T <= 0: return 0, 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
        return price, delta

    @staticmethod
    def calculate_ewma_vol(returns, lam=0.94):
        """Week 5: EWMA 动态波动率预测 (RiskMetrics 标准)"""
        vols = np.zeros(len(returns))
        vols[0] = np.var(returns)
        for t in range(1, len(returns)):
            vols[t] = lam * vols[t-1] + (1 - lam) * (returns[t-1]**2)
        return np.sqrt(vols[-1])

    @staticmethod
    def capm_return(rf, beta, rm):
        """Week 9: CAPM 预期回报率计算"""
        return rf + beta * (rm - rf)

    @staticmethod
    def calculate_var_cvar(returns, confidence_level=0.95):
        """Week 9: 风险价值 (VaR) 与 预期缺口 (CVaR)"""
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = returns[returns <= var].mean()
        return var, cvar

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_realtime_metrics(ticker_list):
        """抓取真实数据计算协方差矩阵 (Portfolio Perspective)"""
        data = yf.download(ticker_list, period="1y")['Close']
        returns = data.pct_change().dropna()
        
        # 计算年化均值和协方差
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        return mean_returns, cov_matrix, returns

    @staticmethod
    def portfolio_performance(weights, mean_returns, cov_matrix):
        """组合层面表现计算"""
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, std

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_portfolio_data(tickers):
        """实时抓取组合数据并计算基础量化指标"""
        data = yf.download(tickers, period="1y")['Close']
        returns = data.pct_change().dropna()
        return returns

    @staticmethod
    def calculate_risk_metrics(returns, weights, confidence=0.95):
        """计算组合 VaR 和 CVaR (Week 9 核心)"""
        port_returns = returns.dot(weights)
        var = np.percentile(port_returns, (1 - confidence) * 100)
        cvar = port_returns[port_returns <= var].mean()
        
        # 年化指标
        ann_return = port_returns.mean() * 252
        ann_vol = port_returns.std() * np.sqrt(252)
        sharpe = (ann_return - 0.03) / ann_vol # 假设无风险利率 3%
        
        return ann_return, ann_vol, sharpe, var, cvar
        
    @staticmethod
    def optimize_portfolio(mean_returns, cov_matrix):
        """寻找最大夏普比率的权重组合 (BA 核心课内容)"""
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        
        def min_func_sharpe(weights, mean_returns, cov_matrix):
            return -QuantEngine.portfolio_performance(weights, mean_returns, cov_matrix)[0] / \
                   QuantEngine.portfolio_performance(weights, mean_returns, cov_matrix)[1]

        result = minimize(min_func_sharpe, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
        
# --- 8. 界面布局 ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 首席宏观研判", 
    "📈 实时全球仪表盘", 
    "🛡️ 行业风险穿透", 
    "🔢 量化工作台 (Quant Lab)"
])

with tab1:
    if st.sidebar.button("🚀 启动深度宏观研判"):
        with st.spinner("策略师正在扫描全球图谱..."):
            sg, tech, geo = fetch_macro_sector_data()
            result, err = get_ai_analysis("macro", sg, tech, geo)
            
            if err:
                if err == "Model_Not_Found":
                    st.warning("⚠️ **系统维护中**：正在优化 AI 模型路径。请稍后再试或联系系统管理员。")
                elif err == "Quota_Exceeded":
                    st.error("📉 **流量预警**：当前投研需求激增，配额已暂时耗尽，请于下一时段重试。")
                else:
                    with st.expander("📝 投研诊断报告"):
                        st.write(f"技术细节：{err}")
                    st.info("💡 建议：请检查 API Key 权限或稍后刷新页面。")
            else:
                st.markdown(result)
                report_data = generate_docx_report(result, "Morning Macro Intelligence Memo")
                st.download_button(label="📥 下载专业投资备忘录 (Word)", data=report_data, file_name=f"Macro_Report_{datetime.date.today()}.docx")

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

with tab3:
    if st.sidebar.button("🔍 穿透行业风险"):
        with st.spinner("评估中..."):
            sg, tech, geo = fetch_macro_sector_data()
            result, err = get_ai_analysis("sector", sg, tech, geo)
            if not err:
                st.markdown(result)
                report_data = generate_docx_report(result, "Sector Strategy Report")
                st.download_button(label="📥 下载行业分析简报", data=report_data, file_name=f"Sector_Strategy_{datetime.date.today()}.docx")
            else:
                st.warning("⚠️ 评估模块暂时无法访问，请检查连接。")
with tab4:
    st.header("⚡ 组合风险实时监测 (Live Portfolio Monitor)")
    
    # 1. 定义专业资产包 (Advisors’ Clique 常用)
    asset_presets = {
        "🇸🇬 新加坡蓝筹 (SG Blue Chips)": ["DBSDF", "U11.SI", "V03.SI", "^STI"],
        "🇺🇸 纳指科技 (Nasdaq Tech)": ["NVDA", "AAPL", "MSFT", "^NDX"],
        "🛡️ 避险资产 (Safe Haven)": ["GC=F", "US10Y", "USDSGD=X"]
    }
    
    # 2. 界面布局
    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.subheader("🛠️ 控制面板")
        preset_choice = st.selectbox("选择监测预设", list(asset_presets.keys()))
        current_tickers = asset_presets[preset_choice]
        
        st.write(f"**成分股:** \n`{', '.join(current_tickers)}`")
        
        # 增加一个开关，控制是否开启“实时追踪模式”
        run_monitor = st.button("🔄 执行实时回测", use_container_width=True, type="primary")

    with col_main:
        if run_monitor:
            with st.status("正在同步全球数据中心...", expanded=True) as status:
                # A. 抓取实时数据
                mean_r, cov_m, raw_ret = QuantEngine.get_portfolio_stats(current_tickers)
                # 默认等权重计算
                weights = np.array([1.0 / len(current_tickers)] * len(current_tickers))
                ann_r, ann_v, sharpe, var, cvar = QuantEngine.calculate_risk_metrics(raw_ret, weights)
                
                status.update(label="✅ 数据同步完成", state="complete", expanded=False)

                # B. 核心指标仪表盘 (简洁排版)
                st.subheader(f"📊 {preset_choice} 实时指标")
                m1, m2, m3 = st.columns(3)
                m1.metric("预期年化收益", f"{ann_r:.2%}")
                m2.metric("年化波动率", f"{ann_v:.2%}")
                # 风险预警：如果 VaR 超过 1.5%，显示红色
                var_status = "inverse" if abs(var) > 0.015 else "normal"
                m3.metric("95% 隔夜 VaR", f"{var:.2%}", delta_color=var_status)

                # C. 专业走势对比图 (Normalized)
                st.write("---")
                st.caption("🚀 资产收益率对齐 (Baseline: 100)")
                # 归一化处理：让不同价格的资产在同一起跑线对比
                norm_df = (1 + raw_ret).cumprod() * 100
                st.line_chart(norm_df)
                
        else:
            # 💡 修复行：使用官方 st.info 替代报错的 light_box
            st.info("💡 **操作指引**：请在左侧选择资产包，并点击“执行实时回测”以获取最新风险归因分析。")
