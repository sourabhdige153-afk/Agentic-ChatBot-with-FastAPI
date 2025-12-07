import streamlit as st
from utils.constant import API_URL, MODEL_NAMES_OPENAI, MODEL_NAMES_GROQ
import requests

st.set_page_config(page_title="LangGraph Agent UI", page_icon="🤖", layout="centered")
st.title("🤖 AI ChatBot Agents")
st.write("create and interact with AI ChatBot agents")

system_prompt = st.text_area("Define System Prompt: ", height=70, placeholder="Type Your system rpompt here...")

provider = st.radio("Select Model Provideer:", ("Groq", "OpenAI"))

if provider == "Groq":
    model_name = st.selectbox("Select Groq Model: ", MODEL_NAMES_GROQ)
else:
    model_name = st.selectbox("Select OpenAI Model: ", MODEL_NAMES_OPENAI)
    
web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Enter Your Query: ", height= 150, placeholder="Ask anything...")

if st.button("Ask Agent: "):
    st.header("Agent Response:")
    if user_query.strip():
        
        payload = {
            "model_name": model_name,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": web_search
        }
        
        response = requests.post(API_URL, json=payload, verify=False)
        if response.status_code == 200:
            response = response.json()
            st.markdown(f"**Response:** {response}")
        else:
            response = response.json()['error']
            st.error(response)
    else:
        response ="Please enter a valid query."
        st.error(response)
