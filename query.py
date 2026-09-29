import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

DB_DIR = "chroma_db"
TOP_K = 3  # how many chunks to retrieve per question


def main():
    print("Loading database...")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )
    print("Ready! Ask a question, or type 'exit' to quit.\n")

    while True:
        query = input("Ask: ").strip()

        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        if not query:
            continue

        # Retrieve relevant chunks
        results = vector_db.similarity_search(query, k=TOP_K)

        # Build context from retrieved chunks
        context = "\n\n".join(doc.page_content for doc in results)

        # Prompt
        prompt = f"""You are a helpful AI assistant.

Answer only using the information in the context below.
If the answer is not in the context, reply exactly:
"I don't know based on the documents."

Context:
{context}

Question:
{query}

Answer:"""

        response = llm.invoke(prompt)

        print("\nAnswer:")
        print(response.content)

        print("\nSources:")
        seen = set()
        for doc in results:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            label = f"{source}, page {page}"
            if label not in seen:
                print(f"- {label}")
                seen.add(label)

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()