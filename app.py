import streamlit as st
import google.generativeai as genai

st.title("API Debugger")

# 1. 检查 Key 是否读到
key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
if not key:
    st.error("未找到 API Key")
else:
    st.success(f"Key 已读取 (长度: {len(key)})")

# 2. 列出该 Key 真正有权限调用的模型
try:
    genai.configure(api_key=key)
    st.write("---")
    st.subheader("你当前 Key 可用的模型列表：")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            st.code(m.name)
    
    if not available_models:
        st.warning("该 Key 似乎没有任何可用的生成模型权限。")
except Exception as e:
    st.error(f"无法获取模型列表，可能是 Key 无效或区域限制: {e}")
