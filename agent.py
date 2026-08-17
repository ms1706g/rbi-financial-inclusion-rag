import os
import re

from database import execute_query
from query import ask_question

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage,
)

from langchain_core.tools import tool
from langchain_groq import ChatGroq


# =========================================================
# CONSTANTS
# =========================================================

UNKNOWN_DOCUMENT_ANSWER = (
    "I don't know based on the provided document."
)

DATABASE_UNAVAILABLE_ANSWER = (
    "The requested information is not available in the database."
)


# =========================================================
# CALCULATOR TOOL
# =========================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Examples:

        125 * 48

        15 * 800 / 100

        1000 / 4

        2 ** 10
    """

    try:

        expression = expression.strip()

        if not expression:

            return (
                "Error: Empty mathematical expression."
            )

        # -------------------------------------------------
        # Allowed characters
        # -------------------------------------------------

        allowed_chars = (
            "0123456789+-*/().% "
        )

        if not all(
            char in allowed_chars
            for char in expression
        ):

            return (
                "Error: Invalid mathematical expression."
            )

        # -------------------------------------------------
        # Convert percentage
        #
        # 20% -> 20/100
        # -------------------------------------------------

        expression = expression.replace(
            "%",
            "/100",
        )

        # -------------------------------------------------
        # Safe evaluation
        # -------------------------------------------------

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {},
        )

        return f"Result: {result}"

    except ZeroDivisionError:

        return (
            "Error: Division by zero."
        )

    except Exception as e:

        return (
            "Error: Unable to calculate the expression. "
            f"{str(e)}"
        )


# =========================================================
# RBI RAG TOOL
# =========================================================

@tool
def rbi_rag_tool(question: str) -> str:
    """
    Search the RBI Financial Inclusion document.

    Returns the answer along with source document
    and page information.
    """

    try:

        answer, sources = ask_question(
            question
        )

        # -------------------------------------------------
        # Format sources
        # -------------------------------------------------

        if sources:

            source_text = "\n".join(
                (
                    f"- {source.get('source', 'Unknown')} | "
                    f"Page {source.get('page', 'Unknown')}"
                )
                for source in sources
            )

        else:

            source_text = (
                "- No sources found"
            )

        # -------------------------------------------------
        # Return
        # -------------------------------------------------

        return (
            f"Answer:\n"
            f"{answer}\n\n"
            f"Sources:\n"
            f"{source_text}"
        )

    except Exception as e:

        return (
            "Error while searching the RBI document: "
            f"{str(e)}"
        )


# =========================================================
# SQL VALIDATION
# =========================================================

def validate_read_only_sql(query: str):
    """
    Validate that SQL is read-only.

    Returns:

        (True, cleaned_query)

    OR

        (False, error_message)
    """

    if not query:

        return (
            False,
            "Error: Empty SQL query.",
        )

    # -----------------------------------------------------
    # Remove -- comments
    # -----------------------------------------------------

    query_without_line_comments = re.sub(
        r"--[^\n]*",
        "",
        query,
    )

    # -----------------------------------------------------
    # Remove /* */ comments
    # -----------------------------------------------------

    query_without_comments = re.sub(
        r"/\*.*?\*/",
        "",
        query_without_line_comments,
        flags=re.DOTALL,
    ).strip()

    if not query_without_comments:

        return (
            False,
            "Error: Empty SQL query.",
        )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    normalized = (
        query_without_comments
        .lower()
        .strip()
    )

    # -----------------------------------------------------
    # Remove final semicolon for checking
    # -----------------------------------------------------

    if normalized.endswith(";"):

        normalized = normalized[:-1].strip()

    # -----------------------------------------------------
    # Reject multiple statements
    # -----------------------------------------------------

    statements = [
        statement.strip()
        for statement in normalized.split(";")
        if statement.strip()
    ]

    if len(statements) != 1:

        return (
            False,
            "Error: Multiple SQL statements are not allowed.",
        )

    # -----------------------------------------------------
    # Only SELECT / WITH
    # -----------------------------------------------------

    if not (
        normalized.startswith("select")
        or normalized.startswith("with")
    ):

        return (
            False,
            "Error: Only read-only SELECT queries are allowed.",
        )

    # -----------------------------------------------------
    # Forbidden SQL operations
    # -----------------------------------------------------

    forbidden_keywords = {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "replace",
        "grant",
        "revoke",
        "merge",
    }

    sql_words = set(
        re.findall(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
            normalized,
        )
    )

    dangerous_words = (
        forbidden_keywords
        .intersection(sql_words)
    )

    if dangerous_words:

        return (
            False,
            "Error: Only read-only SQL queries are allowed.",
        )

    return (
        True,
        query_without_comments,
    )


# =========================================================
# SQL TOOL
# =========================================================

@tool
def sql_tool(query: str) -> str:
    """
    Execute a READ-ONLY SQL query against the sales database.

    Use this for questions about:

    - sales
    - products
    - revenue
    - quantities
    - prices
    - categories
    - customers
    - orders
    - business statistics
    """

    try:

        # -------------------------------------------------
        # Validate SQL
        # -------------------------------------------------

        is_valid, result = (
            validate_read_only_sql(query)
        )

        if not is_valid:

            return result

        cleaned_query = result

        # -------------------------------------------------
        # Execute
        # -------------------------------------------------

        columns, rows = execute_query(
            cleaned_query
        )

        # -------------------------------------------------
        # No results
        # -------------------------------------------------

        if not rows:

            return (
                "No results found."
            )

        # -------------------------------------------------
        # Format header
        # -------------------------------------------------

        result_lines = [
            " | ".join(
                str(column)
                for column in columns
            )
        ]

        # -------------------------------------------------
        # Format rows
        # -------------------------------------------------

        for row in rows:

            result_lines.append(
                " | ".join(
                    str(value)
                    for value in row
                )
            )

        return "\n".join(
            result_lines
        )

    except Exception as e:

        return (
            f"SQL Error: {str(e)}"
        )


# =========================================================
# CREATE LLM
# =========================================================

def create_agent(bind_tools=True):

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:

        raise ValueError(
            "GROQ_API_KEY is not set.\n"
            "Set GROQ_API_KEY before running the agent."
        )

    # -----------------------------------------------------
    # LLM
    # -----------------------------------------------------

    llm = ChatGroq(
        api_key=groq_api_key,
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # -----------------------------------------------------
    # Plain LLM
    # -----------------------------------------------------

    if not bind_tools:

        return llm

    # -----------------------------------------------------
    # Tools
    # -----------------------------------------------------

    tools = [
        rbi_rag_tool,
        calculator,
        sql_tool,
    ]

    # -----------------------------------------------------
    # Bind
    # -----------------------------------------------------

    return llm.bind_tools(
        tools
    )


# =========================================================
# SYSTEM INSTRUCTIONS
# =========================================================

SYSTEM_INSTRUCTION = f"""
You are a document-grounded AI assistant.

You have exactly three specialized tools:

1. rbi_rag_tool
2. calculator
3. sql_tool


=========================================================
RBI RAG TOOL
=========================================================

Use rbi_rag_tool for questions requiring information
from the RBI Financial Inclusion document.

Examples:

"What is financial inclusion?"

"What are the strategic objectives of financial inclusion?"

"What does the RBI document say about customer protection?"

"What recommendations are mentioned in the RBI document?"

You MUST use rbi_rag_tool for RBI/document questions.

Rules:

1. Use only information returned by the tool.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not assume facts.
5. Preserve source document names and page numbers.
6. If information is unavailable, answer exactly:

{UNKNOWN_DOCUMENT_ANSWER}


=========================================================
CALCULATOR
=========================================================

Use calculator for mathematical calculations.

Examples:

125 * 48

15 * 800 / 100

1000 / 4

2 ** 10

Rules:

1. Mathematical calculations MUST use calculator.
2. Do not perform arithmetic yourself.
3. Use the calculator result.
4. Convert natural language into a mathematical expression.

Example:

"15% of 800"

must become:

15 * 800 / 100


=========================================================
SQL TOOL
=========================================================

Use sql_tool for questions about the sales/business
database.

Examples:

"Which product has the highest revenue?"

"What is the total revenue?"

"Which category generated the most revenue?"

"How many laptops were sold?"

Rules:

1. Database questions MUST use sql_tool.
2. Generate only read-only SQL.
3. Use SELECT or read-only WITH queries.
4. Never modify the database.
5. Never invent database values.
6. Never assume database values.
7. Use only information returned by sql_tool.

Forbidden:

INSERT
UPDATE
DELETE
DROP
ALTER
TRUNCATE
CREATE
REPLACE
GRANT
REVOKE
MERGE

If requested database information is unavailable, answer:

{DATABASE_UNAVAILABLE_ANSWER}


=========================================================
MULTI-TOOL QUESTIONS
=========================================================

Some questions require multiple tools.

Example:

"According to the RBI document, what percentage is
mentioned and calculate its value for 500?"

Use:

rbi_rag_tool
calculator


Example:

"What was the total revenue for Electronics and what
percentage of total revenue did it represent?"

Use:

sql_tool
calculator


Example:

"Compare the financial inclusion objective with our
sales performance."

Use:

rbi_rag_tool
sql_tool


Rules:

1. Call every required tool.
2. Wait for the results.
3. Use only returned information.
4. Do not invent missing information.


=========================================================
UNRELATED QUESTIONS
=========================================================

If the question is unrelated to:

- RBI Financial Inclusion document
- mathematics
- sales/business database

Do not use outside knowledge.

Do not call any tool.

Answer exactly:

{UNKNOWN_DOCUMENT_ANSWER}


=========================================================
GENERAL RULES
=========================================================

1. Never invent facts.
2. Never invent RBI information.
3. Never invent database values.
4. Never assume database values.
5. Never use outside knowledge.
6. Use tools when required.
7. Keep answers concise and factual.
8. Preserve RBI source and page information.
9. Use calculator results for calculations.
10. Use SQL results for database questions.
11. Never modify the database.
12. SQL must be read-only.
"""


# =========================================================
# EXECUTE TOOL
# =========================================================

def execute_tool(
    tool_name,
    tool_args,
):

    if tool_name == "rbi_rag_tool":

        print(
            "\n[Agent] Using RBI RAG tool..."
        )

        return rbi_rag_tool.invoke(
            tool_args
        )

    if tool_name == "calculator":

        print(
            "\n[Agent] Using calculator..."
        )

        return calculator.invoke(
            tool_args
        )

    if tool_name == "sql_tool":

        print(
            "\n[Agent] Using SQL tool..."
        )

        return sql_tool.invoke(
            tool_args
        )

    return (
        f"Unknown tool requested: "
        f"{tool_name}"
    )


# =========================================================
# RUN AGENT
# =========================================================

def run_agent(
    question: str,
    return_metadata: bool = False,
):

    # -----------------------------------------------------
    # Tool-capable LLM
    # -----------------------------------------------------

    llm = create_agent(
        bind_tools=True
    )

    # -----------------------------------------------------
    # Messages
    # -----------------------------------------------------

    messages = [
        SystemMessage(
            content=SYSTEM_INSTRUCTION
        ),
        HumanMessage(
            content=question
        ),
    ]

    # -----------------------------------------------------
    # Track tools
    # -----------------------------------------------------

    tools_used = []

    tool_call_counts = {}

    tool_results = []

    # -----------------------------------------------------
    # Maximum number of rounds
    # -----------------------------------------------------

    max_rounds = 2

    # =====================================================
    # TOOL LOOP
    # =====================================================

    for _ in range(max_rounds):

        response = llm.invoke(
            messages
        )

        # -------------------------------------------------
        # No tool call
        # -------------------------------------------------

        if not response.tool_calls:

            final_answer = str(
                response.content
            ).strip()

            if return_metadata:

                return {
                    "answer": final_answer,
                    "tools_used": tools_used,
                }

            return final_answer

        # -------------------------------------------------
        # Add assistant message
        # -------------------------------------------------

        messages.append(
            response
        )

        # -------------------------------------------------
        # Execute tools
        # -------------------------------------------------

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            tool_args = tool_call.get(
                "args",
                {}
            )

            tool_call_id = tool_call["id"]

            # ---------------------------------------------
            # Track tool
            # ---------------------------------------------

            if tool_name not in tools_used:

                tools_used.append(
                    tool_name
                )

            # ---------------------------------------------
            # Execute (dedup: if the same tool has already
            # been used for this question, don't re-run it,
            # tell the model to reuse the earlier result)
            # ---------------------------------------------

            tool_call_counts[tool_name] = (
                tool_call_counts.get(tool_name, 0) + 1
            )

            if tool_call_counts[tool_name] > 1:

                tool_result = (
                    f"Tool '{tool_name}' has already been used "
                    "for this question. "
                    "Use the previous tool result to answer."
                )

            else:

                tool_result = execute_tool(
                    tool_name,
                    tool_args,
                )

            tool_result = str(
                tool_result
            )

            # ---------------------------------------------
            # Store
            # ---------------------------------------------

            tool_results.append(
                f"[{tool_name}]\n"
                f"{tool_result}"
            )

            # ---------------------------------------------
            # Add ToolMessage
            # ---------------------------------------------

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id,
                )
            )

    # =====================================================
    # SAFETY FALLBACK
    # =====================================================

    if tool_results:

        final_answer = (
            tool_results[-1]
        )

    else:

        final_answer = (
            UNKNOWN_DOCUMENT_ANSWER
        )

    if return_metadata:

        return {
            "answer": final_answer,
            "tools_used": tools_used,
        }

    return final_answer


# =========================================================
# TEST CASES
# =========================================================

TEST_CASES = [

    {
        "question": "What is financial inclusion?",
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "formal financial services",
            "economic growth",
        ],
    },

    {
        "question": (
            "What are the strategic objectives of "
            "financial inclusion in India?"
        ),
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "universal access",
            "basic bouquet",
        ],
    },

    {
        "question": (
            "What is 125 multiplied by 48?"
        ),
        "expected_tool": "calculator",
        "expected_keywords": [
            "6000",
        ],
    },

    {
        "question": "What is 15% of 800?",
        "expected_tool": "calculator",
        "expected_keywords": [
            "120",
        ],
    },

    {
        "question": (
            "Which product has the highest revenue?"
        ),
        "expected_tool": "sql_tool",
        "expected_keywords": [
            "laptop",
            "900000",
        ],
    },

    {
        "question": "What is the total revenue?",
        "expected_tool": "sql_tool",
        "expected_keywords": [
            "2215000",
        ],
    },

    {
        "question": (
            "Which category generated the most revenue?"
        ),
        "expected_tool": "sql_tool",
        "expected_keywords": [
            "electronics",
            "1950000",
        ],
    },

    {
        "question": "What is the capital of France?",
        "expected_tool": "none",
        "expected_keywords": [
            "i don't know based on the provided document",
        ],
    },

    {
        "question": (
            "What is the RBI repo rate mentioned for 2026?"
        ),
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document",
        ],
    },

    {
        "question": (
            "What was India's GDP growth rate in 2026 "
            "according to the RBI?"
        ),
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document",
        ],
    },

    {
        "question": (
            "What is the current RBI Governor's name?"
        ),
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document",
        ],
    },

    {
        "question": (
            "What is the population of India according "
            "to the RBI document?"
        ),
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document",
        ],
    },

    {
        "question": (
            "What is India's inflation rate in 2026?"
        ),
        "expected_tool": "rbi_rag_tool",
        "expected_keywords": [
            "i don't know based on the provided document",
        ],
    },
]


# =========================================================
# RUN ONE TEST
# =========================================================

def run_test(test_case):

    question = test_case["question"]

    expected_tool = test_case["expected_tool"]

    expected_keywords = (
        test_case["expected_keywords"]
    )

    print("\n" + "=" * 80)
    print("TEST CASE")
    print("=" * 80)

    print(
        f"Question: {question}"
    )

    print(
        f"Expected tool: {expected_tool}"
    )

    # -----------------------------------------------------
    # Run
    # -----------------------------------------------------

    result = run_agent(
        question,
        return_metadata=True,
    )

    answer = result["answer"]

    tools_used = result["tools_used"]

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    answer_lower = (
        str(answer)
        .lower()
        .strip()
    )

    # -----------------------------------------------------
    # Tool check
    # -----------------------------------------------------

    if expected_tool == "none":

        tool_passed = (
            len(tools_used) == 0
        )

    else:

        tool_passed = (
            expected_tool in tools_used
        )

    # -----------------------------------------------------
    # Keyword check
    # -----------------------------------------------------

    keyword_results = {}

    for keyword in expected_keywords:

        keyword_results[keyword] = (
            keyword.lower()
            in answer_lower
        )

    keywords_passed = all(
        keyword_results.values()
    )

    # -----------------------------------------------------
    # Overall
    # -----------------------------------------------------

    passed = (
        tool_passed
        and keywords_passed
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print("\nActual tools:")

    if tools_used:

        for tool_name in tools_used:

            print(
                f"  - {tool_name}"
            )

    else:

        print(
            "  - None"
        )

    print("\nTool selection:")

    print(
        "  PASS"
        if tool_passed
        else "  FAIL"
    )

    print("\nKeyword checks:")

    for keyword, found in (
        keyword_results.items()
    ):

        print(
            f"  {'PASS' if found else 'FAIL'}: "
            f"{keyword}"
        )

    print("\nAnswer:")

    print(
        answer
    )

    print("\nOverall:")

    print(
        "  PASS"
        if passed
        else "  FAIL"
    )

    return {
        "passed": passed,
        "question": question,
        "expected_tool": expected_tool,
        "actual_tool": (
            tools_used[0]
            if tools_used
            else "none"
        ),
        "actual_tools": tools_used,
        "tools_used": tools_used,
        "tool_passed": tool_passed,
        "keyword_results": keyword_results,
        "answer": answer,
    }


# =========================================================
# RUN ALL TESTS
# =========================================================

def run_all_tests():

    print("\n")
    print("=" * 80)
    print("RUNNING AGENT TEST SUITE")
    print("=" * 80)

    results = []

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print(
            f"\nTEST CASE "
            f"{index}/{len(TEST_CASES)}"
        )

        try:

            result = run_test(
                test_case
            )

            results.append(
                result
            )

        except Exception as e:

            print(
                "\nTEST ERROR:"
            )

            print(
                str(e)
            )

            results.append(
                {
                    "passed": False,
                    "question": test_case[
                        "question"
                    ],
                    "expected_tool": test_case[
                        "expected_tool"
                    ],
                    "actual_tool": "none",
                    "actual_tools": [],
                    "tools_used": [],
                    "tool_passed": False,
                    "keyword_results": {},
                    "answer": (
                        f"Test execution error: {e}"
                    ),
                }
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    passed_tests = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_tests = (
        len(results)
        - passed_tests
    )

    print("\n")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print(
        f"Total tests : {len(results)}"
    )

    print(
        f"Passed      : {passed_tests}"
    )

    print(
        f"Failed      : {failed_tests}"
    )

    if results:

        accuracy = (
            passed_tests
            / len(results)
            * 100
        )

        print(
            f"Accuracy    : {accuracy:.2f}%"
        )

    print("=" * 80)

    # -----------------------------------------------------
    # Individual summary
    # -----------------------------------------------------

    for index, result in enumerate(
        results,
        start=1,
    ):

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"{index:02d}. {status} - "
            f"{result['question']}"
        )

    # =====================================================
    # FAILED TOOL SELECTION CASES
    # =====================================================

    print("\n")
    print("=" * 80)
    print("FAILED TOOL SELECTION CASES")
    print("=" * 80)

    found_tool_failures = False

    for result in results:

        if not result["tool_passed"]:

            found_tool_failures = True

            print(
                "\nQuestion:",
                result["question"],
            )

            print(
                "Expected:",
                result["expected_tool"],
            )

            print(
                "Actual:",
                result["actual_tool"],
            )

            print(
                "All tools used:",
                result["tools_used"],
            )

    if not found_tool_failures:

        print(
            "\nNo tool-selection failures."
        )

    print("=" * 80)

    return results


# =========================================================
# CLI
# =========================================================

def run_cli():

    print("=" * 80)
    print("RBI FINANCIAL INCLUSION AI AGENT")
    print("=" * 80)

    print(
        "Available tools:"
    )

    print(
        "  1. RBI Financial Inclusion RAG"
    )

    print(
        "  2. Calculator"
    )

    print(
        "  3. Sales Database SQL"
    )

    print()

    print(
        "Commands:"
    )

    print(
        "  test  - Run automated tests"
    )

    print(
        "  exit  - Exit"
    )

    print("=" * 80)

    while True:

        try:

            question = input(
                "\nYou: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nGoodbye!"
            )

            break

        except EOFError:

            print(
                "\n\nGoodbye!"
            )

            break

        # -------------------------------------------------
        # Exit
        # -------------------------------------------------

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nGoodbye!"
            )

            break

        # -------------------------------------------------
        # Test
        # -------------------------------------------------

        if question.lower() == "test":

            run_all_tests()

            continue

        # -------------------------------------------------
        # Empty
        # -------------------------------------------------

        if not question:

            continue

        # -------------------------------------------------
        # Agent
        # -------------------------------------------------

        try:

            result = run_agent(
                question,
                return_metadata=True,
            )

            print(
                "\n" + "-" * 80
            )

            print(
                "Assistant:"
            )

            print(
                "-" * 80
            )

            print(
                result["answer"]
            )

            print(
                "\nTools used:"
            )

            if result["tools_used"]:

                for tool_name in (
                    result["tools_used"]
                ):

                    print(
                        f"  - {tool_name}"
                    )

            else:

                print(
                    "  - None"
                )

        except Exception as e:

            print(
                "\n" + "-" * 80
            )

            print(
                "ERROR"
            )

            print(
                "-" * 80
            )

            print(
                str(e)
            )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_cli()