# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from openai import OpenAI
# import ollama


# client = OpenAI()
# # 1. Load PDF
# loader = PyPDFLoader("data/RBI.pdf")
# documents = loader.load()

# print(f"Number of documents: {len(documents)}")

# # 2. Create text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=800,
#     chunk_overlap=100
# )

# # 3. Split documents into chunks
# chunks = text_splitter.split_documents(documents)

# for i, chunk in enumerate(chunks[:5]):
#     print(f"\n--- Chunk {i} ---")
#     print(f"Length: {len(chunk.page_content)} characters")
#     print(chunk.page_content[:300])

# print(f"Number of chunks: {len(chunks)}")

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# print("\nChunks from page 18:")

# for i, chunk in enumerate(chunks):
#     if chunk.metadata.get("page") == 18:
#         print(f"\n--- Chunk index: {i} ---")
#         print(f"Length: {len(chunk.page_content)}")
#         print(repr(chunk.page_content))

# import os
# from langchain_chroma import Chroma

# if os.path.exists("chroma_db"):
#     print("\nLoading existing Chroma database...")

#     vectorstore = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

# else:
#     print("\nCreating new Chroma database...")

#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory="chroma_db"
#     )

#     print("Chroma vector store created successfully!")
# # 4. Inspect first chunk
# print("\nFirst chunk:")
# print(chunks[0].page_content)

# print("\nMetadata:")
# print(chunks[0].metadata)

# # Test embedding
# vector = embeddings.embed_query("What is financial inclusion?")

# print("\nEmbedding vector:")
# print(vector[:10])

# print("\nEmbedding dimensions:")
# print(len(vector))

# # Test semantic search
# query = "What is financial inclusion?"

# results = vectorstore.similarity_search(
#     query,
#     k=6
# )

# print("\n\nRetrieved chunks:")

# for i, result in enumerate(results):
#     print(f"\n--- Result {i + 1} ---")
#     print(result.page_content)
#     print("Metadata:", result.metadata)

#     from collections import Counter

# chunk_texts = [chunk.page_content.strip() for chunk in chunks]

# duplicates = {
#     text: count
#     for text, count in Counter(chunk_texts).items()
#     if count > 1
# }

# print(f"\nUnique chunks: {len(set(chunk_texts))}")
# print(f"Duplicate chunk texts: {len(duplicates)}")

# for text, count in list(duplicates.items())[:5]:
#     print(f"\nDuplicate found {count} times:")
#     print(text[:300])


# context = "\n\n".join(
#     result.page_content for result in results
# )

# question = input("\nAsk a question: ")

# prompt = f"""
# You are a document question-answering assistant.

# Answer the user's question using ONLY the provided context.

# Rules:
# 1. Do not use outside knowledge.
# 2. If the answer cannot be found in the context, say:
#    "I don't know based on the provided document."
# 3. Keep the answer concise and factual.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

# response = ollama.generate(
#     model="phi3:mini",
#     prompt=prompt
# )

# print("\nAnswer:")
# print(response["response"])

# result.metadata["source"]
# result.metadata["page"]





















from query import ask_question


print("RBI Financial Inclusion RAG")
print("Type 'exit' to quit.\n")


while True:

    question = input("Ask a question: ")

    if question.lower() == "exit":
        break

    answer, sources = ask_question(question)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for source in sources:

        if source["page"] is not None:
            print(f"- {source['source']} | Page {source['page']}")
        else:
            print(f"- {source['source']}")

    print()