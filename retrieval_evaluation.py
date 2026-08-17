from query import retrieve_context


# =========================================================
# RETRIEVAL TEST CASES
# =========================================================

TEST_CASES = [

    {
        "question": "What is financial inclusion?",
        "expected_terms": [
            "formal finance",
            "economic growth",
            "poverty alleviation"
        ]
    },

    {
        "question": "Why is financial inclusion important for economic growth?",
        "expected_terms": [
            "economic growth",
            "poverty",
            "income inequality"
        ]
    },

    {
        "question": "What is the role of customer protection in financial inclusion?",
        "expected_terms": [
            "customer protection",
            "grievance redressal"
        ]
    },

    {
        "question": "What are the strategic objectives of financial inclusion in India?",
        "expected_terms": [
            "strategic objectives",
            "universal access",
            "basic bouquet"
        ]
    },

    {
        "question": "What are the major challenges to financial inclusion?",
        "expected_terms": [
            "challenges",
            "barriers",
            "financial inclusion"
        ]
    }

]


# =========================================================
# RETRIEVAL EVALUATION
# =========================================================

def evaluate_retrieval(question, expected_terms, k=4):

    context, results = retrieve_context(
        question,
        k=k
    )

    context_lower = context.lower()

    matched_terms = []

    for term in expected_terms:

        if term.lower() in context_lower:

            matched_terms.append(term)

    if expected_terms:

        score = (
            len(matched_terms)
            / len(expected_terms)
            * 100
        )

    else:

        score = 0


    return {
        "question": question,
        "score": score,
        "matched_terms": matched_terms,
        "total_terms": len(expected_terms),
        "results": results
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    all_scores = []

    for index, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        result = evaluate_retrieval(
            question=test_case["question"],
            expected_terms=test_case["expected_terms"],
            k=4
        )

        all_scores.append(
            result["score"]
        )

        print("\n")
        print("=" * 80)
        print(
            f"TEST {index}/{len(TEST_CASES)}"
        )
        print("=" * 80)

        print(
            "Question:",
            result["question"]
        )

        print(
            "Retrieval Score:",
            f"{result['score']:.2f}%"
        )

        print(
            "Matched Terms:",
            result["matched_terms"]
        )

        print(
            "Retrieved Pages:"
        )

        seen_pages = set()

        for document in result["results"]:

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page"
            )

            if page is not None:

                page += 1

            page_key = (
                source,
                page
            )

            if page_key not in seen_pages:

                print(
                    f"- {source} | Page {page}"
                )

                seen_pages.add(
                    page_key
                )


    # =====================================================
    # SUMMARY
    # =====================================================

    average_score = (
        sum(all_scores)
        / len(all_scores)
    )

    print("\n")
    print("=" * 80)
    print("RETRIEVAL SUMMARY")
    print("=" * 80)

    print(
        f"Average Retrieval Score: "
        f"{average_score:.2f}%"
    )


if __name__ == "__main__":

    main()