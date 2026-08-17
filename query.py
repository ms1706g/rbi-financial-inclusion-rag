from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import ollama

import os
from groq import Groq
import ollama

if os.getenv("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"):
    # Render → Groq
else:
    # Local → Ollama

CHROMA_PATH = "chroma_db"


def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vectorstore


def ask_question(question, k=4):

    vectorstore = get_vectorstore()

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(
        question,
        k=k
    )

    # Build context
    context = "\n\n".join(
        result.page_content
        for result in results
    )

    # Prompt
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the answer cannot be found in the context, say:
"I don't know based on the provided document."
3. Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    try:
        response = ollama.generate(
            model="phi3:mini",
            prompt=prompt
        )

        answer = response["response"]

    except Exception as e:
        answer = (
            "Unable to generate an answer. "
            "Please make sure Ollama is running and the model is available."
        )

        sources = []

        return answer, sources

    # Sources
    sources = []
    seen_sources = set()

    for result in results:

        source = result.metadata.get(
            "source",
            "Unknown source"
        )

        page = result.metadata.get("page")

        if page is not None:
            page += 1

        source_key = (source, page)

        if source_key not in seen_sources:
            sources.append({
                "source": source,
                "page": page
            })

            seen_sources.add(source_key)

    return answer, sources
