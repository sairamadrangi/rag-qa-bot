# 📄 Document Q&A Bot (RAG Pipeline)

A command-line and Streamlit-based Q&A bot that answers natural language questions from a collection of AI/ML documents using Retrieval-Augmented Generation (RAG). Answers are grounded strictly in the source documents and include citations (filename + page number).

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core language |
| LangChain | 1.x | RAG pipeline orchestration |
| ChromaDB | 1.x | Vector database (persistent) |
| sentence-transformers | 5.x | Embedding model (all-MiniLM-L6-v2) |
| Groq API | — | LLM inference (openai/gpt-oss-20b) |
| Streamlit | 1.x | Web UI |
| pypdf | 6.x | PDF text extraction |

## Architecture Overview

```
data/ (PDF/TXT files)
      |
      v
[1] Ingestion - load PDFs/TXT, strip page-number lines
      |
      v
[2] Chunking - RecursiveCharacterTextSplitter (900 chars, 150 overlap)
      |
      v
[3] Embedding - sentence-transformers/all-MiniLM-L6-v2 (batched)
      |
      v
[4] Vector Store - ChromaDB, persisted to disk (chroma_db/)
      |
      v
      (query time)
      |
User question -> embed query -> similarity search (top-k) -> retrieved chunks
      |
      v
[5] Generation - Groq LLM answers ONLY from retrieved context,
                 replies "I don't know based on the documents." otherwise
      |
      v
Answer + Source citations (filename, page)
```

Indexing (`index.py`) and querying (`query.py` / `app.py`) are fully separate scripts. The database is built once and reused on every query run.

## Chunking Strategy

**Fixed-size chunking with overlap** (`RecursiveCharacterTextSplitter`), 900 characters per chunk with 150 characters of overlap. This size keeps each chunk small enough for precise retrieval while the overlap prevents sentences from being cut off at chunk boundaries, which would otherwise lose context near the edges.

## Embedding Model & Vector Database

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` - free, runs locally (no API cost), fast, and a well-established baseline for semantic search.
- **Vector database:** ChromaDB - lightweight, persists to disk automatically, and integrates directly with LangChain, so indexing and querying can be cleanly separated.

## Setup Instructions

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd rag-qa-bot

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key (see Environment Variables below)

# 5. Build the vector database (run once, or whenever documents change)
python index.py

# 6. Ask questions - pick ONE:
python query.py          # command-line version
streamlit run app.py     # web UI version
```

## Environment Variables

Create a `.env` file in the project root (never commit this file):

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at https://console.groq.com/keys

## Example Queries

| Question | Answer theme |
|---|---|
| "What is self-attention?" | Explains the attention mechanism from the Transformer paper |
| "What is unsupervised learning?" | Definition from the machine learning document |
| "What is RAG?" | Explains Retrieval-Augmented Generation from the RAG paper |
| "What is a context window?" | Definition from the embeddings/LLM documents |
| "Who won the 2023 cricket world cup?" | Not in the documents - bot replies "I don't know based on the documents." |

## Known Limitations

- Retrieval quality depends on chunk size; very short or very specific questions spanning multiple chunks may retrieve incomplete context.
- The embedding model runs locally and is lightweight, so it may miss subtle semantic matches that a larger embedding model would catch.
- Currently supports PDF and TXT; DOCX is not implemented.
- No conversation memory - each question is answered independently of prior turns.