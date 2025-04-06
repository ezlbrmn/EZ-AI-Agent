import os
import streamlit as st
import requests
from dotenv import load_dotenv

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load .env
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
ENDPOINT = os.getenv("DEEPSEEK_ENDPOINT")
MODEL = os.getenv("DEEPSEEK_MODEL")

# Load vector store
vectorstore = FAISS.load_local(
    "./vectorstore",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# UI Setup
st.set_page_config(page_title="EZ-AI SecArch Chat", layout="wide")
st.title("🛡️ EZ-AI SecArch Helper – Interactive Security Agent")

# Session-based chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt
user_prompt = st.chat_input("Describe your GenAI use case or ask a follow-up question...")

if user_prompt:
    # Show user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # RAG grounding
    related_docs = vectorstore.similarity_search(user_prompt, k=4)
    doc_text = "\n\n".join([doc.page_content for doc in related_docs])

    # Build chat history + latest prompt
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[-10:]  # keep last 10 exchanges
    ]

    # Inject context chunk
    system_prompt = (
        "You are a cybersecurity expert trained in OWASP Top 10 for LLMs, SANS AI Controls, and OWASP Agentic Threats. "
        "Use the internal guidelines below to inform your analysis.\n\n"
        f"INTERNAL SECURITY GUIDELINES:\n{doc_text}"
    )

    chat_messages = [{"role": "system", "content": system_prompt}] + chat_history

    # Call DeepSeek
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    payload = {
        "model": MODEL,
        "messages": chat_messages,
        "temperature": 0.4,
        "max_tokens": 2000
    }

    with st.spinner("DeepSeek R1 is thinking..."):
        response = requests.post(ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        output = response.json()["choices"][0]["message"]["content"]

        # Remove <think> blocks
        if "<think>" in output and "</think>" in output:
            start = output.find("<think>")
            end = output.find("</think>") + len("</think>")
            output = output.replace(output[start:end], "").strip()

        st.chat_message("assistant").markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
