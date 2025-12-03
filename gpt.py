import streamlit as st
from langchain_openai import ChatOpenAI
import httpx
import os
from datetime import datetime

# =========================================
# Page Setup
# =========================================
st.set_page_config(page_title="Local ChatGPT UI", layout="wide")

# Title
st.markdown("<h1 style='text-align:center; margin-bottom:20px;'>💬 Local ChatGPT UI</h1>", unsafe_allow_html=True)

# =========================================
# LLM Setup
# =========================================
os.environ["TIKTOKEN_CACHE_DIR"] = "./token"
client = httpx.Client(verify=False)

OPENAPIKEY = "sk-_hlWKUmXssdcqD4O7AI3hQ"

# Available models (ADD MORE IF NEEDED)
MODEL_OPTIONS = {
    "DeepSeek V3": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-DeepSeek-V3-0324",
        "api_key":OPENAPIKEY
    },

    "DeepSeek R1": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-DeepSeek-R1",
        "api_key":OPENAPIKEY
    },

    "GPT Turbo": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-gpt-35-turbo",
        "api_key":OPENAPIKEY
    },
    "GPT 4o": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-gpt-4o",
        "api_key":OPENAPIKEY
    },

    "GPT 4o mini": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-gpt-4o-mini",
        "api_key":OPENAPIKEY
    },

    "Text Embedding 3 large": {
        "base_url": "https://genailab.tcs.in",
        "model": "genailab-maas-text-embedding-3-large",
        "api_key":OPENAPIKEY
    },

    "whisper": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-whisper",
        "api_key":OPENAPIKEY
    },

    "Llama 3.2-90B Vision Instruct": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct",
        "api_key":OPENAPIKEY
    },

    "Llama-3.3-70B-Instruct": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-Llama-3.3-70B-Instruct",
        "api_key":OPENAPIKEY
    },

    "Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8",
        "api_key":OPENAPIKEY
    },
    
    "Phi-3.5-vision-instruct": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-Phi-3.5-vision-instruct",
        "api_key":OPENAPIKEY
    },

    "Phi-4-reasoning": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure_ai/genailab-maas-Phi-4-reasoning",
        "api_key":OPENAPIKEY
    }

    
}

st.sidebar.header("⚙️ Settings")

# Model Dropdown
selected_model = st.sidebar.selectbox(
    "Choose Model:",
    list(MODEL_OPTIONS.keys())
)


# =========================================
# Theme Toggle
# =========================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

st.sidebar.button(f"Toggle to { 'Dark' if st.session_state.theme=='light' else 'Light' } Mode", on_click=toggle_theme)

# =========================================
# Create LLM Instance
# =========================================

def create_llm_for_selected_model():
    llm_config = MODEL_OPTIONS[selected_model]
    key_to_use = OPENAPIKEY

    return ChatOpenAI(
        base_url=llm_config["base_url"],
        model=llm_config["model"],
        api_key= key_to_use,
        http_client=client
    )


# =========================================
# Chat History
# =========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================
# Chat Window Layout
# =========================================
st.markdown("""
<style>
.chatbox {
    height: 70vh;
    overflow-y: scroll;
    padding: 15px;
    border: 1px solid #cccccc;
    border-radius: 12px;
    background-color: rgba(240,240,240,0.4);
}
.user-bubble {
    text-align:right; 
    background-color:#DCF8C6; 
    padding:10px; 
    border-radius:12px; 
    margin:10px 0; 
    max-width:80%; 
    float:right;
}
.bot-bubble {
    text-align:left; 
    background-color:#F1F0F0; 
    padding:10px; 
    border-radius:12px; 
    margin:10px 0; 
    max-width:80%;
    float:left;
}
.copy-btn {
    font-size:10px; 
    margin-top:5px;
}
</style>
""", unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    st.markdown('<div class="chatbox">', unsafe_allow_html=True)

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(
                f"""
                <div class="user-bubble">
                    <b>You:</b><br>{chat['content']}<br>
                </div>
                <div style='clear:both;'></div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div class="bot-bubble">
                    <b>{selected_model}:</b><br>{chat['content']}<br>
                </div>
                <div style='clear:both;'></div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# User Input (Pinned at Bottom)
# =========================================
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Message:", height=80)
    submitted = st.form_submit_button("Send")

def extract_text(resp):
    try:
        # dict-like
        if isinstance(resp, dict):
            for k in ("result", "content", "text", "output"):
                if k in resp and resp[k]:
                    return resp[k]
            if "choices" in resp and isinstance(resp["choices"], (list, tuple)) and resp["choices"]:
                c = resp["choices"][0]
                if isinstance(c, dict):
                    return c.get("text") or c.get("message", {}).get("content") or str(c)
                return str(c)
        # list/tuple of messages
        if isinstance(resp, (list, tuple)):
            parts = []
            for item in resp:
                if isinstance(item, dict):
                    parts.append(item.get("content") or item.get("text") or str(item))
                else:
                    parts.append(getattr(item, "content", None) or getattr(item, "text", None) or str(item))
            return "\n".join([p for p in parts if p])
        # object with attributes (AIMessage etc.)
        content = getattr(resp, "content", None) or getattr(resp, "text", None)
        if content:
            return content
    except Exception:
        pass
    return str(resp)

if submitted and user_input and OPENAPIKEY:
    effective_key = MODEL_OPTIONS[selected_model].get("api_key") or OPENAPIKEY
    if not effective_key:
        st.warning("Please provide an API key in the sidebar or add a default key in the script.")
    else:
        timestamp = datetime.now().strftime("%H:%M")
        # add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input, "time": timestamp})

        # Create LLM instance (use sidebar key if present)
        llm = create_llm_for_selected_model()
        # timestamp = datetime.now().strftime("%H:%M")

        # Model Response
        with st.spinner(f"{selected_model} is typing..."):
            response = llm.invoke(user_input)

    
        assistant_text = extract_text(response)

        st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_text,
        "time": timestamp
        })

        # st.experimental_rerun()

# =========================================
# Dark / Light Mode CSS
# =========================================
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    body { background-color: #121212 !important; color: #e0e0e0 !important; }
    textarea, input, .stTextInput, .stTextArea { background-color: #333 !important; color: #fff !important; }
    </style>
    """, unsafe_allow_html=True)
