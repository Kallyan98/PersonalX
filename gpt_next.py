import streamlit as st
from langchain_openai import ChatOpenAI
import httpx
import os
import io
import base64
from PIL import Image
import pandas as pd
import tempfile
from datetime import datetime



# =========================================
# Page Setup
# =========================================
st.set_page_config(page_title="Local ChatGPT UI", layout="wide")

st.markdown("""
<style>
/* round icon button styling */
.stButton>button {
    height: 38px !important;
    width: 38px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 22px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: white !important;
    border: 1px solid #dadada !important;
}
</style>
""", unsafe_allow_html=True)



# === session-state initialization (MUST be early) ===
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []     # list of {"name":..., "file":...}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_display_label" not in st.session_state:
    st.session_state.selected_display_label = None
if "selected_model_internal_key" not in st.session_state:
    st.session_state.selected_model_internal_key = None


st.markdown("<h1 style='text-align:center; margin-bottom:20px;'>💬 Local ChatGPT UI</h1>", unsafe_allow_html=True)

# =========================================
# LLM Setup
# =========================================
os.environ["TIKTOKEN_CACHE_DIR"] = "./token"
# NOTE: verify=False is insecure; keep only for local dev/self-signed certs
client = httpx.Client(verify=False)

OPENAPIKEY = "sk-_hlWKUmXssdcqD4O7AI3hQ"  # <-- replace with your key if needed

# INTERNAL model map (internal keys are the values we use to create LLMs)
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
        "model": "azure/genailab-maas-gpt-35-turbo",
        "api_key":OPENAPIKEY
    },

    "GPT 4o": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure/genailab-maas-gpt-4o",
        "api_key":OPENAPIKEY
    },

    "GPT 4o mini": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure/genailab-maas-gpt-4o-mini",
        "api_key":OPENAPIKEY
    },

    "Text Embedding 3 large": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure/genailab-maas-text-embedding-3-large",
        "api_key": OPENAPIKEY
    },

    "whisper": {
        "base_url": "https://genailab.tcs.in",
        "model": "azure/genailab-maas-whisper",
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

# Friendly display labels list (display label -> internal key)
MODEL_DISPLAY_LIST = [
    ("DeepSeek V3", "DeepSeek V3"),
    ("DeepSeek R1", "DeepSeek R1"),
    ("GPT-3.5 Turbo (fast)", "GPT Turbo"),
    ("GPT-4o", "GPT 4o"),
    ("GPT 4o Mini", "GPT 4o mini"),
    ("Text Embedding 3 Large", "Text Embedding 3 large"),
    ("Whisper (speech)", "whisper"),
    ("Llama 3.2-90B Vision", "Llama 3.2-90B Vision Instruct"),
    ("Llama 3.3-70B Instruct", "Llama-3.3-70B-Instruct"),
    ("Llama-4 Maverick 17B", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
    ("Phi 3.5 Vision", "Phi-3.5-vision-instruct"),
    ("Phi 4 Reasoning", "Phi-4-reasoning"),
]

MODEL_DISPLAY_LABELS = [label for label, internal in MODEL_DISPLAY_LIST]
DISPLAY_TO_INTERNAL = {label: internal for label, internal in MODEL_DISPLAY_LIST}


if not st.session_state.selected_display_label:
    st.session_state.selected_display_label = MODEL_DISPLAY_LABELS[0]
    st.session_state.selected_model_internal_key = DISPLAY_TO_INTERNAL[st.session_state.selected_display_label]
# =========================================
# Create LLM Instance (lazy)
# =========================================
def create_llm_for_selected_model():
    sel_key = st.session_state.get("selected_model_internal_key", DISPLAY_TO_INTERNAL[MODEL_DISPLAY_LABELS[0]])
    # Ensure sel_key exists in MODEL_OPTIONS
    if sel_key not in MODEL_OPTIONS:
        # fallback to first option
        sel_key = MODEL_DISPLAY_LIST[0][1]
        st.session_state.selected_model_internal_key = sel_key
        st.session_state.selected_display_label = MODEL_DISPLAY_LABELS[0]
    llm_config = MODEL_OPTIONS[sel_key]
    key_to_use = llm_config.get("api_key") or OPENAPIKEY
    return ChatOpenAI(base_url=llm_config["base_url"], model=llm_config["model"], api_key=key_to_use, http_client=client)  

# =========================================
# Chat Window Layout (render existing chat_history)
# =========================================
st.markdown(
    """
<style>
.chatbox {
    height: 60vh;
    overflow-y: auto;
    padding: 15px;
    border: 1px solid #ececec;
    border-radius: 12px;
    background-color: rgba(250,250,250,0.6);
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
</style>
""",
    unsafe_allow_html=True,
)

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
                """,
                unsafe_allow_html=True,
            )
        else:
            bot_label = st.session_state.get("selected_display_label", MODEL_DISPLAY_LABELS[0])
            st.markdown(
                f"""
                <div class="bot-bubble">
                    <b>{bot_label}:</b><br>{chat['content']}<br>
                </div>
                <div style='clear:both;'></div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)



# =========================================
# Controls row (OUTSIDE the form) - plus, inline model selector, spacer, mic
# =========================================
# cols_controls = st.columns([0.10, 0.24, 0.52, 0.06, 0.08])

# # Plus button toggles uploader panel
# with cols_controls[0]:
#     if st.button("➕", key="plus_button_outside"):
#         st.session_state.show_uploader = not st.session_state.show_uploader

# # Inline model selector (keeps in sync with sidebar choice)
# with cols_controls[1]:
#     inline_display_label = st.selectbox(
#         "",
#         MODEL_DISPLAY_LABELS,
#         index=MODEL_DISPLAY_LABELS.index(st.session_state.get("selected_display_label", MODEL_DISPLAY_LABELS[0])),
#         key="inline_model_select_outside",
#     )
#     # store selection in session_state
#     st.session_state.selected_display_label = inline_display_label
#     st.session_state.selected_model_internal_key = DISPLAY_TO_INTERNAL[inline_display_label]


# with cols_controls[2]:
#     st.write("")  # spacer

# # Microphone recorder (attempt to import audio_recorder package)
# with cols_controls[3]:
#     recorded = None
#     try:
#         # audio_recorder may return bytes, dict or file-like depending on package version
#         from audio_recorder_streamlit import audio_recorder
#         recorded = audio_recorder(text="Press to record", key="audio_rec_outside")
#     except Exception:
#         if st.button("🎤", key="mic_fallback_outside"):
#             st.info("Browser recording not available. Upload an audio file using + uploader.")

# with cols_controls[4]:
#     st.write("")

controls = st.container()
with controls:
    col_plus, col_model, col_spacer, col_mic, col_mic_label = st.columns(
        [0.10, 0.30, 0.42, 0.07, 0.13]
    )

    # + BUTTON (circle styled via CSS)
    with col_plus:
        if st.button("➕", key="plus_button_outside"):   # Full-width plus looks cleaner
            st.session_state.show_uploader = not st.session_state.show_uploader

    # MODEL SELECTOR
    with col_model:
        inline_display_label = st.selectbox(
            "",
            MODEL_DISPLAY_LABELS,
            index=MODEL_DISPLAY_LABELS.index(
                st.session_state.get("selected_display_label", MODEL_DISPLAY_LABELS[0])
            ),
            key="inline_model_select_outside",
        )
        st.session_state.selected_display_label = inline_display_label
        st.session_state.selected_model_internal_key = DISPLAY_TO_INTERNAL[
            inline_display_label
        ]

    # SPACER
    with col_spacer:
        st.write("")

    # MIC ICON
    with col_mic:
        recorded = None
        try:
            from audio_recorder_streamlit import audio_recorder
            recorded = audio_recorder(icon_size="30px", key="audio_rec_outside")
        except Exception:
            st.button("🎤", key="mic_fallback_outside")

    # # MIC LABEL → “Press to record”
    # with col_mic_label:
    #     st.markdown(
    #         "<div style='margin-top:10px; font-size:14px;'>Press to record</div>",
    #         unsafe_allow_html=True,
    #     )

# Show uploader panel outside the form as well (so files can be selected before typing)
if st.session_state.show_uploader:
    st.markdown("**Upload files (xlsx, csv, pdf, docx, png, jpg, jpeg, gif, mp3, wav):**")
    uploaded = st.file_uploader(
        "Choose files",
        type=["xlsx", "xls", "csv", "pdf", "docx", "png", "jpg", "jpeg", "gif", "mp3", "wav"],
        accept_multiple_files=True,
        key="uploader_multi_outside",
    )
    if uploaded:
        for f in uploaded:
            if f.name not in [x["name"] for x in st.session_state.uploaded_files]:
                st.session_state.uploaded_files.append({"name": f.name, "file": f})
        st.success(f"Queued {len(uploaded)} file(s). They will be attached to the next message.")

# =========================================
# Helper: extract text from uploaded files
# =========================================
def extract_text_from_file(filelike):
    name = filelike.name.lower()
    raw_text = ""
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(filelike)
            raw_text = df.head(200).to_csv(index=False)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filelike)
            raw_text = df.head(200).to_csv(index=False)
        elif name.endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(filelike)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            raw_text = "\n".join(pages)
        elif name.endswith(".docx"):
            import docx
            doc = docx.Document(filelike)
            raw_text = "\n".join([p.text for p in doc.paragraphs])
        elif name.endswith((".png", ".jpg", ".jpeg", ".gif")):
            try:
                import pytesseract
                img = Image.open(filelike)
                raw_text = pytesseract.image_to_string(img)
                if not raw_text.strip():
                    raw_text = f"[image: {filelike.name}]"
            except Exception:
                raw_text = f"[image: {filelike.name}]"
        elif name.endswith((".mp3", ".wav")):
            raw_text = f"[audio file uploaded: {filelike.name}]"
        else:
            try:
                raw_text = filelike.getvalue().decode("utf-8", errors="ignore")
            except Exception:
                raw_text = f"[binary file attached: {filelike.name}]"
    except Exception as e:
        raw_text = f"[failed to extract text from {filelike.name}: {e}]"
    return raw_text

# If there are queued uploads, show them with attach/remove actions
if st.session_state.uploaded_files:
    st.markdown("**Queued attachments:**")
    for idx, info in enumerate(st.session_state.uploaded_files):
        cols2 = st.columns([0.8, 0.08, 0.12])
        with cols2[0]:
            st.write(info["name"])
        with cols2[1]:
            if st.button("Remove", key=f"remove_{idx}"):
                st.session_state.uploaded_files.pop(idx)
                st.experimental_rerun()
        with cols2[2]:
            if st.button("Attach to next message", key=f"attach_{idx}"):
                filelike = info["file"]
                txt = extract_text_from_file(filelike)
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"[Attachment: {info['name']}]\n{txt}",
                    "time": datetime.now().strftime("%H:%M"),
                })
                st.success(f"Attached {info['name']} as system context.")
                st.session_state.uploaded_files.pop(idx)
                st.experimental_rerun()

# =========================================
# Input form (only message area + submit) - this MUST contain the form_submit_button
# =========================================
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Message:", height=90)
    submitted = st.form_submit_button("Send")

# =========================================
# Robust extractor for possible response shapes
# =========================================
def extract_text(resp):
    try:
        if isinstance(resp, dict):
            for k in ("result", "content", "text", "output"):
                if k in resp and resp[k]:
                    return resp[k]
            if "choices" in resp and isinstance(resp["choices"], (list, tuple)) and resp["choices"]:
                c = resp["choices"][0]
                if isinstance(c, dict):
                    return c.get("text") or c.get("message", {}).get("content") or str(c)
                return str(c)
        if isinstance(resp, (list, tuple)):
            parts = []
            for item in resp:
                if isinstance(item, dict):
                    parts.append(item.get("content") or item.get("text") or str(item))
                else:
                    parts.append(getattr(item, "content", None) or getattr(item, "text", None) or str(item))
            return "\n".join([p for p in parts if p])
        content = getattr(resp, "content", None) or getattr(resp, "text", None)
        if content:
            return content
    except Exception:
        pass
    return str(resp)

# =========================================
# Handle submit: process attachments/audio, create combined prompt, call model
# =========================================
if submitted and user_input:
    # 1) Process recorded audio (if any)
    if recorded:
        try:
            # normalize recorded to file-like named "recorded.wav"
            if isinstance(recorded, dict) and "audio" in recorded:
                audio_b64 = recorded["audio"].split(",")[-1]
                audio_bytes = base64.b64decode(audio_b64)
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "recorded.wav"
            elif isinstance(recorded, (bytes, bytearray)):
                audio_file = io.BytesIO(recorded)
                audio_file.name = "recorded.wav"
            else:
                audio_file = recorded
            # try whisper transcription if available
            try:
                import whisper
                model = whisper.load_model("small")  # choose model size per resources
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    audio_file.seek(0)
                    tf.write(audio_file.read())
                    temp_name = tf.name
                result = model.transcribe(temp_name)
                transcript = result.get("text", "").strip()
                if transcript:
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": transcript,
                        "time": datetime.now().strftime("%H:%M"),
                    })
            except Exception:
                # fallback: add audio file as queued uploaded file for external STT
                st.session_state.uploaded_files.append({"name": getattr(audio_file, "name", "recorded.wav"), "file": audio_file})
                st.warning("Local whisper not available; saved audio into uploaded files for later processing.")
        except Exception as e:
            st.warning(f"Audio processing failed: {e}")

    # 2) Process queued uploads to produce system context chunks
    system_context_chunks = []
    if st.session_state.uploaded_files:
        for item in list(st.session_state.uploaded_files):
            try:
                filelike = item["file"]
                txt = extract_text_from_file(filelike)
                system_context_chunks.append(f"[Attachment: {item['name']}]\n{txt}")
                st.session_state.uploaded_files.remove(item)
            except Exception as e:
                system_context_chunks.append(f"[Attachment: {item['name']} extraction failed: {e}]")

    # 3) Build combined prompt for the model
    combined_prompt = ""
    if system_context_chunks:
        combined_prompt += "\n\n".join(system_context_chunks) + "\n\n"
    combined_prompt += user_input

    # 4) Add user message to local chat history
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({"role": "user", "content": user_input, "time": timestamp})

    # 5) Create llm and call it with combined prompt
    llm = create_llm_for_selected_model()
    bot_label = st.session_state.get("selected_display_label", MODEL_DISPLAY_LABELS[0])
    with st.spinner(f"{bot_label} is typing..."):
        # invoke with combined_prompt so attachments are visible to the model
        response = llm.invoke(combined_prompt)

    assistant_text = extract_text(response)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text, "time": timestamp})

# =========================================
# Dark / Light Mode CSS
# =========================================
if st.session_state.theme == "dark":
    st.markdown(
        """
    <style>
    body { background-color: #121212 !important; color: #e0e0e0 !important; }
    textarea, input, .stTextInput, .stTextArea { background-color: #333 !important; color: #fff !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )
