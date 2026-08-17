from query import get_vectorstore


TEST_QUESTIONS = [
    "What is financial inclusion?",
    "Why is financial inclusion important for economic growth?",
    "What are the major challenges to financial inclusion?",
    "What is the role of customer protection in financial inclusion?",
    "What are the strategic objectives of financial inclusion in India?"
]


vectorstore = get_vectorstore()


for question in TEST_QUESTIONS:

    results = vectorstore.similarity_search(
        question,
        k=4
    )

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")

    for i, result in enumerate(results):

        print(f"\nResult {i + 1}")
        print(f"Page: {result.metadata.get('page', 'Unknown') + 1}")
        print(result.page_content[:300])