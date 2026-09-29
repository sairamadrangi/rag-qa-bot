from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

DB_DIR = "chroma_db"
TOP_K = 3

st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📄"
)

st.title("📄 Document Q&A Bot")
st.markdown("Ask questions about the AI/ML documents in the knowledge base.")


@st.cache_resource
def load_db():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding
    )


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


vector_db = load_db()
llm = load_llm()

# keep chat history visible on screen
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question...")

if query:
    # show user message
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # retrieve relevant chunks
    results = vector_db.similarity_search(query, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc in results)

    prompt = f"""You are a helpful AI assistant.

Answer only using the information in the context below.
If the answer is not in the context, reply exactly:
"I don't know based on the documents."

Context:
{context}

Question:
{query}

Answer:"""

    with st.spinner("Thinking..."):
        response = llm.invoke(prompt)

    answer = response.content

    # build sources list, no duplicates
    seen = set()
    source_lines = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        label = f"{source}, page {page}"
        if label not in seen:
            source_lines.append(f"- {label}")
            seen.add(label)

    full_reply = answer + "\n\n**Sources:**\n" + "\n".join(source_lines)

    st.chat_message("ai").markdown(full_reply)
    st.session_state.messages.append({"role": "ai", "content": full_reply})