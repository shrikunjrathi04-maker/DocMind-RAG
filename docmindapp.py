import os

# Tell Hugging Face / sentence-transformers not to load TensorFlow (we only need PyTorch).
# This avoids the protobuf version conflict caused by an installed TensorFlow.
# These lines must stay above all other imports.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import uuid
import tempfile

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    from langchain_classic.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------------
# Models and prompt (same settings as your main.py / create_database.py)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI Assistant.
Use only the provided context to answer the question.


If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


def build_retriever(uploaded_file):
    """PDF -> pages -> chunks -> embeddings -> Chroma -> MMR retriever."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        data = PyPDFLoader(tmp_path)
        docs = data.load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if not chunks:
        return None, 0, 0

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        collection_name=f"doc_{uuid.uuid4().hex}",
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    return retriever, len(docs), len(chunks)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
defaults = {
    "retriever": None,
    "doc_name": None,
    "doc_pages": 0,
    "doc_chunks": 0,
    "messages": [],
    "ended": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(900px 500px at 0% -10%, rgba(56, 189, 248, 0.16), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(139, 92, 246, 0.18), transparent 60%),
        #0a0b10;
    color: #e8e9f0;
}
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 860px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(14, 15, 22, 0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}
.side-title { font-size: 1.15rem; font-weight: 700; color: #fff; margin-bottom: 0.2rem; }
.side-sub { font-size: 0.85rem; color: #9a9bb0; margin-bottom: 1rem; }

[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.04);
    border: 1.5px dashed rgba(56, 189, 248, 0.5);
    border-radius: 16px;
}

/* Hero */
.hero { text-align: center; margin: 0.8rem 0 1.8rem; }
.hero .badge {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: #7dd3fc; background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.4);
}
.hero h1 {
    font-size: 3rem; font-weight: 800; margin: 0.9rem 0 0.4rem;
    background: linear-gradient(120deg, #ffffff 0%, #7dd3fc 45%, #c4b5fd 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: #9a9bb0; font-size: 1.02rem; margin: 0; }

/* Steps / empty state */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin-top: 1.4rem; }
.step {
    padding: 1.2rem; border-radius: 18px; text-align: center;
    background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.09);
}
.step .n { font-size: 1.8rem; }
.step .t { font-weight: 700; color: #fff; margin-top: 0.4rem; }
.step .d { font-size: 0.85rem; color: #9a9bb0; margin-top: 0.2rem; }

/* Doc card */
.doc-card {
    padding: 0.9rem 1.1rem; border-radius: 16px; margin-top: 0.8rem;
    background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3);
}
.doc-card .name { font-weight: 600; color: #fff; word-break: break-all; }
.doc-card .meta { font-size: 0.8rem; color: #9a9bb0; margin-top: 0.2rem; }

/* Buttons */
.stButton > button {
    width: 100%; border-radius: 12px; padding: 0.6rem 1rem; font-weight: 600;
    color: #06121a; border: none;
    background: linear-gradient(120deg, #38bdf8, #a78bfa);
    transition: all .2s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(56,189,248,0.35); color: #06121a; }

/* Chat */
[data-testid="stChatMessage"] {
    padding: 0.9rem 1.1rem; border-radius: 18px; margin-bottom: 0.8rem;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(56,189,248,0.2), rgba(255,255,255,0.04));
    border-color: rgba(56,189,248,0.4);
}
[data-testid="stChatMessage"] p { line-height: 1.65; font-size: 0.98rem; }
[data-testid="stChatInput"] { border-radius: 16px; border: 1px solid rgba(255,255,255,0.12); }
[data-testid="stChatInput"]:focus-within {
    border-color: #38bdf8; box-shadow: 0 0 0 3px rgba(56,189,248,0.22);
}
.hint { text-align: center; color: #6f7087; font-size: 0.8rem; margin-top: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar: PDF upload
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-title">📄 Your Document</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-sub">Upload a PDF and ask questions about it.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("⚡ Process Document"):
            with st.spinner("Reading and indexing your PDF..."):
                try:
                    retriever, pages, chunks = build_retriever(uploaded_file)
                except Exception as e:
                    retriever, pages, chunks = None, 0, 0
                    st.error(f"Could not process the PDF: {e}")

            if retriever is None:
                st.error("No readable text was found in this PDF.")
            else:
                st.session_state.retriever = retriever
                st.session_state.doc_name = uploaded_file.name
                st.session_state.doc_pages = pages
                st.session_state.doc_chunks = chunks
                st.session_state.messages = []
                st.session_state.ended = False
                st.rerun()

    if st.session_state.doc_name:
        st.markdown(
            f"""
<div class="doc-card">
    <div class="name">✅ {st.session_state.doc_name}</div>
    <div class="meta">{st.session_state.doc_pages} pages · {st.session_state.doc_chunks} chunks indexed</div>
</div>
""",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
    <span class="badge">📄 DocMind</span>
    <h1>Chat with your PDF</h1>
    <p>Answers come only from the document you upload.</p>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state.retriever is None:
    st.markdown(
        """
<div class="steps">
    <div class="step"><div class="n">📤</div><div class="t">1. Upload</div><div class="d">Add a PDF in the sidebar</div></div>
    <div class="step"><div class="n">⚡</div><div class="t">2. Process</div><div class="d">Click Process Document</div></div>
    <div class="step"><div class="n">💬</div><div class="t">3. Ask</div><div class="d">Ask questions about it</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "📄"):
        st.write(message["content"])

if st.session_state.ended:
    st.info("Chat ended.")
    if st.button("Start a new chat"):
        st.session_state.messages = []
        st.session_state.ended = False
        st.rerun()
    st.stop()

query = st.chat_input("You :")
st.markdown('<div class="hint">Type 0 to end the chat</div>', unsafe_allow_html=True)

if query:
    if query == "0":
        st.session_state.ended = True
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.write(query)

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Searching the document..."):
            docs = st.session_state.retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.invoke({"context": context, "question": query})

            response = get_llm().invoke(final_prompt)
        st.write(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})