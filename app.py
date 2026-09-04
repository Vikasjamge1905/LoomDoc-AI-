"""
Document-based RAG Chatbot using Gemini API
-------------------------------------------------
Upload a document (PDF / DOCX / TXT) -> it gets split into chunks ->
chunks are embedded and stored in a local FAISS vector index ->
when you ask a question, the most relevant chunks are retrieved and
sent to Gemini along with your question so it can answer
based ONLY on your document.

The Gemini API key is read from the GEMINI_API_KEY environment variable.
"""

import os
import streamlit as st
from google import genai
from streamlit.errors import StreamlitSecretNotFoundError
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# PDF / DOCX readers
from pypdf import PdfReader
import docx

# ============================================================
# 1. CONFIG — provide the Gemini API key through the environment
# ============================================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, StreamlitSecretNotFoundError):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.6-flash"

CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between chunks
TOP_K = 4                # number of chunks to retrieve per question

# ============================================================
# 2. Clients (created once)
# ============================================================
@st.cache_resource
def get_gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=GEMINI_API_KEY)

@st.cache_resource
def get_embedder():
    # Small, fast, runs locally — no API key needed for embeddings
    return SentenceTransformer("all-MiniLM-L6-v2")

gemini_client = get_gemini_client()
embedder = get_embedder()

# ============================================================
# 3. Document loading
# ============================================================
def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(file) -> str:
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs)

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def load_document(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return read_docx(uploaded_file)
    elif name.endswith(".txt"):
        return read_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")

# ============================================================
# 4. Chunking
# ============================================================
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]

# ============================================================
# 5. Build FAISS index from chunks
# ============================================================
def build_index(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve(index, chunks, question, top_k=TOP_K):
    q_emb = embedder.encode([question]).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ============================================================
# 6. Ask Gemini with retrieved context
# ============================================================
def ask_grok(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided document context. If the answer isn't in the context, say "
        "you don't know based on the document. Be concise and accurate."
    )
    user_prompt = f"Document context:\n{context}\n\nQuestion: {question}"

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "temperature": 0.2,
        },
    )
    return response.text

# ============================================================
# 7. Streamlit UI
# ============================================================
st.set_page_config(page_title="Document Q&A Chatbot (Gemini RAG)", page_icon="📄")
st.title("📄 Document Q&A Chatbot (Gemini + RAG)")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None and st.session_state.chunks is None:
    with st.spinner("Reading and indexing document..."):
        raw_text = load_document(uploaded_file)
        chunks = chunk_text(raw_text)
        index, _ = build_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.index = index
    st.success(f"Document indexed into {len(chunks)} chunks. Ask away!")

if st.session_state.chunks is not None:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input("Ask a question about the document...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant_chunks = retrieve(st.session_state.index, st.session_state.chunks, question)
                answer = ask_grok(question, relevant_chunks)
                st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))

    if st.button("Reset / Upload a new document"):
        st.session_state.chunks = None
        st.session_state.index = None
        st.session_state.chat_history = []
        st.rerun()
else:
    st.info("Upload a document above to start chatting with it.")
