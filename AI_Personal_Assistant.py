# from langchain_openai import ChatOpenAI 
# from langchain.schema import HumanMessage, AIMessage
# import streamlit as st
# import os 
# import httpx 

# client = httpx.Client(verify=False)

# llm = ChatOpenAI(
# base_url="https://genailab.XXXXXXXXXXX.in",
# model = "azure_ai/genailab-maas-DeepSeek-V3-0324",
# #model = "azure/genailab-maas-DeepSeek-R1",
# api_key="XXXXXXXXXXXXXXXXXX", 
# # Will be provided during event. And this key is for Hackathon purposes only and should not be used for any unauthorized purposes
# http_client = client
# )

# # -------------------------
# # Streamlit UI
# # -------------------------
# st.set_page_config(page_title="Healthcare Assistant")
# st.title("💊 Your personal healthcare buddy")

# print("Assistant: Hello! 👋 How may I help you today?")
# print("(Type 'Exit' anytime to close this chat.)")

# # Initialize conversation history
# messages = []

# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() in ["exit", "quit", "close"]:
#         print("\nAssistant: 👋 Goodbye! Have a great day!")
#         break

#     # Add user message to history
#     messages.append(HumanMessage(content=user_input))

#     # Get model response
#     response = llm.invoke(messages)

#     # Print the assistant's reply
#     print("\nAssistant:", response.content)
#     print("\n💬 (Type 'Exit' to end the chat or ask another question.)")

#     # Add assistant reply to history
#     messages.append(AIMessage(content=response.content))


import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import httpx

# -------------------------
# Streamlit App Setup
# -------------------------
st.set_page_config(page_title="🧠 AI Assistant", page_icon="🤖", layout="centered")

# st.markdown("""
#     <style>
#     .stApp h1 {
#         white-space: nowrap;
#         overflow: hidden;
#         text-overflow: ellipsis;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.title("🤖 Your Personal Healthcare Buddy")
st.markdown(
    """
    <h1 style="font-size:32px; white-space: nowrap;">🤖 Your Personal AI Assistant</h1>
    """,
    unsafe_allow_html=True
)
st.markdown("Welcome! Ask me anything.")

# -------------------------
# LLM Configuration
# -------------------------
client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.xxxxxxxxx.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="XXXXXXXXXXXXXXXXXXXXXX",
    http_client=client
)

# -------------------------
# Initialize Session State
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="👋 Hello! I'm your AI assistant. How may I help you today?")
    ]

# -------------------------
# Display Chat History
# -------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# -------------------------
# Chat Input Box
# -------------------------
if user_input := st.chat_input("Type your message here..."):
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Get model response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 💭"):
            try:
                response = llm.invoke(st.session_state.messages)
                st.markdown(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"⚠️ Something went wrong: {e}")



