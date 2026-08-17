import re
import time

from agent import run_agent
from query import retrieve_context
from grounding import check_groundedness


# =========================================================
# EVALUATION DATASET
# =========================================================

TEST_CASES = [

    {
        "question": "What is financial inclusion?",
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "formal financial services",
            "economic growth"
        ]
    },

    {
        "question": "What are the strategic objectives of financial inclusion in India?",
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "universal access",
            "basic bouquet"
        ]
    },

    {
        "question": "What is 125 multiplied by 48?",
        "expected_tool": "calculator",
        "expected_keywords": [
            "6000"
        ]
    },

    {
        "question": "What is 15% of 800?",
        "expected_tool": "calculator",
        "expected_keywords": [
            "120"
        ]
    },

    {
        "question": "Which product has the highest revenue?",
        "expected_tool": "sql_tool",
        "expected_keywords": [
            "laptop",
            "900000"
        ]
    },

    {
        "question": "What is the total revenue?",
        "expected_tool": "sql_tool",
        "expected_keywords": [
            "2215000"
        ]
    },

    {
        "question": "Which category generated the most revenue?",
        "expected_tool": "sql_tool",
        "expected_keywords": [
            "electronics",
            "1950000"
        ]
    },

    {
        "question": "What is the capital of France?",
        "expected_tool": "none",
        "expected_keywords": [
            "i don't know based on the provided document"
        ]
    },

    # -----------------------------------------------------
    # Adversarial / Hallucination Tests
    # -----------------------------------------------------

    {
        "question": "What is the RBI repo rate mentioned for 2026?",
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document"
        ]
    },

    {
        "question": "What was India's GDP growth rate in 2026 according to the RBI?",
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document"
        ]
    },

    {
        "question": "What is the current RBI Governor's name?",
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document"
        ]
    },

    {
        "question": "What is the population of India according to the RBI document?",
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document"
        ]
    },

    {
        "question": "What is India's inflation rate in 2026?",
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document"
        ]
    }

]


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize_text(text):

    text = text.lower()

    # Remove currency symbols
    text = text.replace("₹", "")

    # Remove commas from numbers
    text = text.replace(",", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# ANSWER EVALUATION
# =========================================================

def evaluate_answer(answer, expected_keywords):

    normalized_answer = normalize_text(answer)

    for keyword in expected_keywords:

        normalized_keyword = normalize_text(
            keyword
        )

        if normalized_keyword not in normalized_answer:

            return False

    return True


# =========================================================
# EVALUATE ONE TEST CASE
# =========================================================

def evaluate_case(test_case):

    question = test_case["question"]

    expected_tool = test_case["expected_tool"]

    expected_keywords = test_case["expected_keywords"]

    # -----------------------------------------------------
    # Start timer
    # -----------------------------------------------------

    start_time = time.perf_counter()

    # -----------------------------------------------------
    # Run REAL AGENT
    # -----------------------------------------------------

    result = run_agent(
        question,
        return_metadata=True
    )

    # -----------------------------------------------------
    # Stop timer
    # -----------------------------------------------------

    latency = time.perf_counter() - start_time

    answer = result["answer"]

    tools_used = result["tools_used"]

    # -----------------------------------------------------
    # Determine actual tool
    # -----------------------------------------------------

    if tools_used:

        actual_tool = tools_used[0]

    else:

        actual_tool = "none"

    # -----------------------------------------------------
    # Tool Selection Accuracy
    # -----------------------------------------------------

    tool_correct = (
        actual_tool == expected_tool
    )

    # -----------------------------------------------------
    # Answer Accuracy
    # -----------------------------------------------------

    answer_correct = evaluate_answer(
        answer,
        expected_keywords
    )

    # -----------------------------------------------------
    # Grounding Evaluation
    # -----------------------------------------------------

    grounding_result = None

    # Only evaluate grounding for RBI questions
    if "rbi_rag_tool" in tools_used:

        context, _ = retrieve_context(
            question,
            k=4
        )

        grounding_result = check_groundedness(
            answer,
            context
        )

    # -----------------------------------------------------
    # Return Results
    # -----------------------------------------------------

    return {

        "question": question,

        "expected_tool": expected_tool,

        "actual_tool": actual_tool,

        "tools_used": tools_used,

        "tool_correct": tool_correct,

        "answer_correct": answer_correct,

        "answer": answer,

        "grounding": grounding_result,

        "latency": latency

    }


# =========================================================
# RUN EVALUATION
# =========================================================

def run_evaluation():

    print("=" * 80)
    print("AGENT EVALUATION")
    print("=" * 80)

    results = []

    for index, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        print("\n")
        print("=" * 80)
        print(
            f"TEST {index}/{len(TEST_CASES)}"
        )
        print("=" * 80)

        print(
            "Question:",
            test_case["question"]
        )

        result = evaluate_case(
            test_case
        )

        results.append(result)

        # -------------------------------------------------
        # Tool information
        # -------------------------------------------------

        print(
            "\nExpected tool:",
            result["expected_tool"]
        )

        print(
            "Actual tool:",
            result["actual_tool"]
        )

        print(
            "All tools used:",
            result["tools_used"]
        )

        print(
            "Tool selection:",
            "PASS"
            if result["tool_correct"]
            else "FAIL"
        )

        # -------------------------------------------------
        # Answer information
        # -------------------------------------------------

        print(
            "Answer:",
            "PASS"
            if result["answer_correct"]
            else "FAIL"
        )

        # -------------------------------------------------
        # Grounding information
        # -------------------------------------------------

        if result["grounding"] is not None:

            print(
                "Grounded:",
                "PASS"
                if result["grounding"]["grounded"]
                else "FAIL"
            )

            print(
                "Grounding Score:",
                f"{result['grounding']['score']:.2f}%"
            )

            print(
                "Supported Sentences:",
                result["grounding"]["supported_sentences"]
            )

            print(
                "Total Sentences:",
                result["grounding"]["total_sentences"]
            )

        # -------------------------------------------------
        # Latency
        # -------------------------------------------------

        print(
            "Latency:",
            f"{result['latency']:.2f}s"
        )


    # =====================================================
    # METRICS
    # =====================================================

    total = len(results)

    # -----------------------------------------------------
    # Tool Selection Accuracy
    # -----------------------------------------------------

    tool_accuracy = (
        sum(
            result["tool_correct"]
            for result in results
        )
        / total
        * 100
    )

    # -----------------------------------------------------
    # Answer Accuracy
    # -----------------------------------------------------

    answer_accuracy = (
        sum(
            result["answer_correct"]
            for result in results
        )
        / total
        * 100
    )

    # -----------------------------------------------------
    # Average Latency
    # -----------------------------------------------------

    average_latency = (
        sum(
            result["latency"]
            for result in results
        )
        / total
    )

    # -----------------------------------------------------
    # Grounding Metrics
    # -----------------------------------------------------

    grounding_results = [

        result["grounding"]

        for result in results

        if result["grounding"] is not None

    ]

    if grounding_results:

        grounded_count = sum(
            result["grounded"]
            for result in grounding_results
        )

        grounding_accuracy = (
            grounded_count
            / len(grounding_results)
            * 100
        )

        average_grounding_score = (
            sum(
                result["score"]
                for result in grounding_results
            )
            / len(grounding_results)
        )

    else:

        grounding_accuracy = 0

        average_grounding_score = 0


    # =====================================================
    # FAILED TOOL SELECTION CASES
    # =====================================================

    failed_tool_cases = [

        result

        for result in results

        if not result["tool_correct"]

    ]


    # =====================================================
    # FAILED ANSWER CASES
    # =====================================================

    failed_answer_cases = [

        result

        for result in results

        if not result["answer_correct"]

    ]


    # =====================================================
    # EVALUATION SUMMARY
    # =====================================================

    print("\n")
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Total Tests: {total}"
    )

    print(
        f"Tool Selection Accuracy: "
        f"{tool_accuracy:.2f}%"
    )

    print(
        f"Answer Accuracy: "
        f"{answer_accuracy:.2f}%"
    )

    print(
        f"Groundedness: "
        f"{grounding_accuracy:.2f}%"
    )

    print(
        f"Average Grounding Score: "
        f"{average_grounding_score:.2f}%"
    )

    print(
        f"Average Latency: "
        f"{average_latency:.2f}s"
    )


    # =====================================================
    # FAILED TOOL CASES
    # =====================================================

    if failed_tool_cases:

        print("\n")
        print("=" * 80)
        print("FAILED TOOL SELECTION CASES")
        print("=" * 80)

        for result in failed_tool_cases:

            print(
                "\nQuestion:",
                result["question"]
            )

            print(
                "Expected:",
                result["expected_tool"]
            )

            print(
                "Actual:",
                result["actual_tool"]
            )

            print(
                "All tools used:",
                result["tools_used"]
            )

    else:

        print(
            "\nTool selection: ALL TESTS PASSED"
        )


    # =====================================================
    # FAILED ANSWER CASES
    # =====================================================

    if failed_answer_cases:

        print("\n")
        print("=" * 80)
        print("FAILED ANSWER CASES")
        print("=" * 80)

        for result in failed_answer_cases:

            print(
                "\nQuestion:",
                result["question"]
            )

            print(
                "Actual answer:",
                result["answer"]
            )

    else:

        print(
            "\nAnswer evaluation: ALL TESTS PASSED"
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_evaluation()