from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import ollama
from groq import Groq


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

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    context = "\n\n".join(
        result.page_content
        for result in results
    )

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

    try:

        groq_api_key = os.getenv("GROQ_API_KEY")

        if groq_api_key:

            client = Groq(
                api_key=groq_api_key
            )

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )

            answer = response.choices[0].message.content

        else:

            response = ollama.generate(
                model="phi3:mini",
                prompt=prompt
            )

            answer = response["response"]

    except Exception:

        answer = (
            "Unable to generate an answer. "
            "Please check the LLM configuration."
        )

        return answer, []

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