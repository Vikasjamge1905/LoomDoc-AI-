# Document RAG Chatbot (Gemini)

A simple document-based chatbot: upload a PDF/DOCX/TXT file, and ask
questions about it. It uses RAG (Retrieval-Augmented Generation):

1. Your document is split into chunks.
2. Chunks are embedded locally (no API key needed) using `sentence-transformers`.
3. Embeddings are stored in a FAISS vector index.
4. When you ask a question, the most relevant chunks are retrieved
   and sent to **Gemini** along with your question, so the model
   answers based on your document instead of guessing.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure your Gemini API key

Set the key in the environment before starting Streamlit (PowerShell):

```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key"
streamlit run app.py
```

Use the key from https://aistudio.google.com/app/apikey. The app does
not store the key in the source code.

## 3. Run the app

```bash
streamlit run app.py
```

This opens a local web page (usually http://localhost:8501). Upload
a document, wait for it to be indexed, then ask questions in the
chat box.

## Notes / things you can customize

- `GEMINI_MODEL` — change to a Gemini model your API key can access.
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — control how the document is split.
  Smaller chunks = more precise retrieval but less context per chunk.
- `TOP_K` — how many chunks are retrieved per question. Increase for
  longer/more complex documents.
- Supports PDF, DOCX, and TXT out of the box. To support more types
  (e.g. `.pptx`, `.csv`), add another `read_xxx()` function and wire
  it into `load_document()`.
- **Security warning:** never commit your API key to a public repository.
