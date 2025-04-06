import os
import streamlit as st
import requests
from dotenv import load_dotenv

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Load .env credentials
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
ENDPOINT = os.getenv("DEEPSEEK_ENDPOINT")
MODEL = os.getenv("DEEPSEEK_MODEL")

# UI setup
st.set_page_config(page_title="EZ-AI SecArch Chat", layout="wide")
st.title("🛡️ EZ-AI SecArch Helper – Interactive Security Agent")

# Load initial vectorstore once
if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = FAISS.load_local(
            "./vectorstore",
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
    except:
        st.session_state.vectorstore = None

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# File upload (optional)
uploaded_files = st.file_uploader("📎 Upload PDFs (e.g. NIST, ISO, Internal Controls)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    new_docs = []
    for file in uploaded_files:
        loader = PyPDFLoader(file)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = file.name
            page.metadata["page"] = page.metadata.get("page", 0)
        new_docs.extend(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_store = FAISS.from_documents(new_docs, embeddings)
    new_store.save_local("vectorstore")
    st.session_state.vectorstore = new_store
    st.success("✅ Uploaded and indexed new documents.")

# Mode toggle
mode = st.radio("💡 Response Mode", ["Strict (only use docs)", "Flexible (allow external knowledge)"], index=0)

# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt
user_prompt = st.chat_input("Describe your GenAI use case or ask a follow-up question...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload documents before asking questions.")
        st.stop()

    related_docs = st.session_state.vectorstore.similarity_search(user_prompt, k=8)

    with st.expander("📂 Retrieved Documents (Used by Agent)", expanded=False):
        for i, doc in enumerate(related_docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            st.markdown(f"**Chunk {i+1}**  \n📄 Source: `{source}` – Page: `{page}`\n\n> {doc.page_content.strip()}")

    rag_chunks = [f">>> {doc.page_content.strip()} <<<" for doc in related_docs]
    doc_text = "\n\n".join(rag_chunks)

    grounding_instruction = (
        "Only use the information inside >>> <<<. Do not invent or assume anything outside of it."
        if mode.startswith("Strict")
        else "Use the information inside >>> <<< as primary context, but you may incorporate your cybersecurity expertise when helpful."
    )

    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[-10:]
    ]

    system_prompt = (
    "You are a cybersecurity expert trained in OWASP Top 10 for LLMs, SANS AI Controls, and OWASP Agentic Threats.\n"
    f"{grounding_instruction}\n\n"
    "When responding, perform the following tasks:\n"
    "1. Identify and list **clear security risks** in the described GenAI or LLM application.\n"
    "2. For each risk, propose a **detailed mitigation plan**.\n"
    "3. Reference **specific controls** from OWASP, SANS, NIST, or ISO when applicable.\n"
    "4. Use **markdown formatting**, numbered lists, and bullet points for clarity.\n"
    "5. Be concise but comprehensive — focus on critical issues first.\n\n"
    f"{doc_text}"
)

    chat_messages = [{"role": "system", "content": system_prompt}] + chat_history

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    payload = {
        "model": MODEL,
        "messages": chat_messages,
        "temperature": 0.3,
        "max_tokens": 4096
    }

    with st.spinner("🤖 DeepSeek R1 analyzing..."):
        response = requests.post(ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        output = response.json()["choices"][0]["message"]["content"]

        if "<think>" in output and "</think>" in output:
            start = output.find("<think>")
            end = output.find("</think>") + len("</think>")
            output = output.replace(output[start:end], "").strip()

        st.chat_message("assistant").markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
    else:
        st.error(f"❌ Error: {response.status_code} - {response.text}")
