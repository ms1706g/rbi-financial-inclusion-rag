# =========================================================
# query.py
# RBI FINANCIAL INCLUSION RAG
# =========================================================

import os

import ollama
from groq import Groq
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_PATH = "chroma_db"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_MODEL = "openai/gpt-oss-20b"

OLLAMA_MODEL = "phi3:mini"


# =========================================================
# GET VECTORSTORE
# =========================================================

@lru_cache(maxsize=1)
def get_vectorstore():

    print("[RAG] Loading embedding model and vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vectorstore


# =========================================================
# RETRIEVE CONTEXT
# =========================================================

def retrieve_context(question: str, k: int = 4):
    """
    Retrieve the most relevant chunks from the
    RBI Financial Inclusion document.

    Returns:
        context: Combined text from retrieved chunks
        results: Original Chroma documents
    """

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    # -----------------------------------------------------
    # No results
    # -----------------------------------------------------

    if not results:
        return "", []

    # -----------------------------------------------------
    # Combine retrieved documents
    # -----------------------------------------------------

    context = "\n\n".join(
        result.page_content
        for result in results
    )

    return context, results


# =========================================================
# BUILD PROMPT
# =========================================================

def build_prompt(question: str, context: str) -> str:
    """
    Build a document-grounded prompt.
    """

    prompt = f"""
You are a document question-answering assistant.

You must answer the user's question using ONLY the
provided context from the RBI Financial Inclusion document.

IMPORTANT RULES:

1. Use ONLY the provided context.

2. Do NOT use outside knowledge.

3. Do NOT use your own knowledge about RBI, India,
   economics, finance, banking, or current events.

4. Do NOT invent facts.

5. Do NOT assume information that is not explicitly
   supported by the context.

6. If the answer cannot be found in the context,
   respond exactly:

I don't know based on the provided document.

7. Keep the answer concise and factual.

8. If the context contains conflicting information,
   mention the conflict instead of guessing.

9. Answer the user's exact question.

---------------------------------------------------------
CONTEXT
---------------------------------------------------------

{context}

---------------------------------------------------------
QUESTION
---------------------------------------------------------

{question}

---------------------------------------------------------
ANSWER
---------------------------------------------------------
"""

    return prompt


# =========================================================
# GENERATE ANSWER WITH GROQ
# =========================================================

def generate_with_groq(prompt: str) -> str:
    """
    Generate an answer using Groq.
    """

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set."
        )

    client = Groq(
        api_key=groq_api_key
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=500
    )

    return (
        response.choices[0]
        .message
        .content
        .strip()
    )


# =========================================================
# GENERATE ANSWER WITH OLLAMA
# =========================================================

def generate_with_ollama(prompt: str) -> str:
    """
    Generate an answer using local Ollama.
    """

    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt
    )

    return response["response"].strip()


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(prompt: str) -> str:
    """
    Generate answer using Groq if GROQ_API_KEY exists.

    Otherwise fall back to Ollama.
    """

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    try:

        # -------------------------------------------------
        # Prefer Groq
        # -------------------------------------------------

        if groq_api_key:

            return generate_with_groq(
                prompt
            )

        # -------------------------------------------------
        # Fallback to Ollama
        # -------------------------------------------------

        return generate_with_ollama(
            prompt
        )

    except Exception as e:

        print(
            "\n[RAG] LLM generation error:",
            str(e)
        )

        return (
            "Unable to generate an answer. "
            "Please check the LLM configuration."
        )


# =========================================================
# EXTRACT SOURCES
# =========================================================

def extract_sources(results):
    """
    Extract unique source document and page information
    from Chroma results.
    """

    sources = []

    seen_sources = set()

    for result in results:

        metadata = result.metadata or {}

        # -------------------------------------------------
        # Source
        # -------------------------------------------------

        source = metadata.get(
            "source",
            "Unknown source"
        )

        # -------------------------------------------------
        # Page
        # -------------------------------------------------

        page = metadata.get(
            "page"
        )

        # -------------------------------------------------
        # Chroma/PDF page numbering
        #
        # If your metadata starts at 0, convert to 1-based.
        # -------------------------------------------------

        if isinstance(page, int):

            page = page + 1

        # -------------------------------------------------
        # Unique source key
        # -------------------------------------------------

        source_key = (
            source,
            page
        )

        if source_key in seen_sources:
            continue

        sources.append(
            {
                "source": source,
                "page": page
            }
        )

        seen_sources.add(
            source_key
        )

    return sources


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(question: str, k: int = 4):
    """
    Main RAG function.

    Flow:

        Question
            ↓
        Chroma similarity search
            ↓
        Relevant document chunks
            ↓
        Context
            ↓
        LLM
            ↓
        Answer + Sources

    Returns:

        answer, sources
    """

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    question = question.strip()

    if not question:

        return (
            "I don't know based on the provided document.",
            []
        )

    try:

        # =================================================
        # RETRIEVE DOCUMENTS
        # =================================================

        context, results = retrieve_context(
            question,
            k=k
        )

        # -------------------------------------------------
        # No relevant documents
        # -------------------------------------------------

        if not results or not context.strip():

            return (
                "I don't know based on the provided document.",
                []
            )

        # =================================================
        # BUILD PROMPT
        # =================================================

        prompt = build_prompt(
            question,
            context
        )

        # =================================================
        # GENERATE ANSWER
        # =================================================

        answer = generate_answer(
            prompt
        )

        # =================================================
        # EXTRACT SOURCES
        # =================================================

        sources = extract_sources(
            results
        )

        # =================================================
        # RETURN
        # =================================================

        return answer, sources

    except Exception as e:

        print(
            "\n[RAG] Error:",
            str(e)
        )

        return (
            "Unable to search the RBI document. "
            f"Error: {str(e)}",
            []
        )

# =========================================================
# TEST RAG DIRECTLY
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RBI FINANCIAL INCLUSION RAG")
    print("=" * 70)

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:

            print(
                "\nGoodbye!"
            )

            break

        if not question:
            continue

        # -------------------------------------------------
        # Ask question
        # -------------------------------------------------

        answer, sources = ask_question(
            question
        )

        # -------------------------------------------------
        # Print answer
        # -------------------------------------------------

        print(
            "\n" + "-" * 70
        )

        print(
            "Answer:"
        )

        print(
            answer
        )

        # -------------------------------------------------
        # Print sources
        # -------------------------------------------------

        print(
            "\nSources:"
        )

        if sources:

            for source in sources:

                print(
                    f"- {source['source']} | "
                    f"Page {source['page']}"
                )

        else:

            print(
                "- No sources found"
            )

        print(
            "-" * 70
        )