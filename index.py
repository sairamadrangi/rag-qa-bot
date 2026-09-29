import os
import re
import glob
import shutil

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
DB_DIR = "chroma_db"


def clean_text(text):
    # remove lines that are just a page number, like "12"
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # collapse extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_all_documents():
    all_docs = []

    for filepath in sorted(glob.glob(f"{DATA_DIR}/*")):
        filename = os.path.basename(filepath)

        if filepath.endswith(".pdf"):
            pages = PyPDFLoader(filepath).load()
        elif filepath.endswith(".txt"):
            pages = TextLoader(filepath, encoding="utf-8").load()
        else:
            continue

        for page in pages:
            page.page_content = clean_text(page.page_content)
            page.metadata["source"] = filename
            if "page" not in page.metadata:
                page.metadata["page"] = 0

        print(f"Loaded {filename}: {len(pages)} page(s)")
        all_docs.extend(pages)

    return all_docs


def main():
    print("Step 1: Loading documents...")
    docs = load_all_documents()
    print(f"Total documents loaded: {len(docs)}")

    if len(docs) == 0:
        print("No documents found! Check that your 'data' folder has files in it.")
        return

    print("Step 2: Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    print("Step 3: Creating embeddings (this downloads a model the first time, be patient)...")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Step 4: Saving to database...")
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=DB_DIR
    )

    print(f"Done! Database saved in '{DB_DIR}' folder.")


if __name__ == "__main__":
    main()