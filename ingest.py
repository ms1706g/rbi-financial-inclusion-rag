import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PDF_PATH = "data/RBI.pdf"
CHROMA_PATH = "chroma_db"


def ingest_documents():

    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    chunks = [
    chunk for chunk in chunks
    if len(chunk.page_content.strip()) >= 200
]

    print(f"Created {len(chunks)} chunks after filtering")

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database only if it doesn't exist
    if os.path.exists(CHROMA_PATH):

        print("Chroma database already exists.")
        print("Skipping ingestion.")

    else:

        print("Creating Chroma database...")

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )

        print("Chroma database created successfully.")


if __name__ == "__main__":
    ingest_documents()