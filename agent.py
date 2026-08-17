import os

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from query import ask_question


# =========================================================
# CALCULATOR TOOL
# =========================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Use this tool for arithmetic calculations such as
    addition, subtraction, multiplication, division,
    percentages, and powers.
    """

    try:
        allowed_chars = "0123456789+-*/().% "

        # Validate characters
        if not all(char in allowed_chars for char in expression):
            return "Error: Invalid mathematical expression."

        # Convert percentage
        expression = expression.replace("%", "/100")

        # Evaluate safely with builtins disabled
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return f"Result: {result}"

    except Exception as e:
        return f"Error: Unable to calculate the expression. {str(e)}"


# =========================================================
# RBI RAG TOOL
# =========================================================

@tool
def rbi_rag_tool(question: str) -> str:
    """
    Search the RBI Financial Inclusion document and answer
    questions using information from the document.

    Returns the answer along with source document and page.
    """

    try:

        answer, sources = ask_question(question)

        # -------------------------------------------------
        # Format sources
        # -------------------------------------------------

        if sources:

            source_text = "\n".join(
                f"- {source.get('source', 'Unknown')} | "
                f"Page {source.get('page', 'Unknown')}"
                for source in sources
            )

        else:

            source_text = "- No sources found"

        # -------------------------------------------------
        # Return RAG result
        # -------------------------------------------------

        return f"""
Answer:
{answer}

Sources:
{source_text}
"""

    except Exception as e:

        return (
            "Error while searching the RBI document: "
            f"{str(e)}"
        )


# =========================================================
# CREATE AGENT
# =========================================================

def create_agent():

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:

        raise ValueError(
            "GROQ_API_KEY is not set.\n"
            "Set it in your environment before running the agent."
        )

    # -----------------------------------------------------
    # Create LLM
    # -----------------------------------------------------

    llm = ChatGroq(
        api_key=groq_api_key,
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # -----------------------------------------------------
    # Register tools
    # -----------------------------------------------------

    tools = [
        rbi_rag_tool,
        calculator,
    ]

    # -----------------------------------------------------
    # Bind tools
    # -----------------------------------------------------

    return llm.bind_tools(tools)


# =========================================================
# RUN AGENT
# =========================================================

def run_agent(question: str) -> str:

    llm = create_agent()

    # =====================================================
    # SYSTEM INSTRUCTIONS
    # =====================================================

    system_instruction = """
You are a document-grounded AI assistant for the
RBI Financial Inclusion document.

You have access to exactly two tools.

---------------------------------------------------------
TOOL 1: rbi_rag_tool
---------------------------------------------------------

Use rbi_rag_tool when the user's question requires
information from the RBI Financial Inclusion document.

This includes questions about:

- RBI Financial Inclusion
- financial inclusion
- RBI recommendations
- RBI strategies
- strategic objectives
- financial services
- banking access
- customer protection
- financial literacy
- financial inclusion policies
- recommendations contained in the document
- facts, statistics, dates, objectives, or statements
  contained in the document

---------------------------------------------------------
TOOL 2: calculator
---------------------------------------------------------

Use calculator for mathematical calculations.

Examples:

- 100 + 200
- 500 * 20%
- 1000 / 4
- 2 ** 10

---------------------------------------------------------
IMPORTANT RULES
---------------------------------------------------------

1. If the question is about information contained in the
   RBI Financial Inclusion document, you MUST call
   rbi_rag_tool.

2. If the question requires mathematical calculation,
   you MUST call calculator.

3. If a question requires both RBI information and a
   calculation, you may call both tools.

4. Do NOT use outside knowledge to answer questions about
   the RBI document.

5. After using rbi_rag_tool, answer ONLY using information
   returned by that tool.

6. Do NOT invent or assume facts from the RBI document.

7. If the RBI tool does not provide enough information,
   say exactly:

I don't know based on the provided document.

8. For questions unrelated to the RBI document and not
   requiring mathematical calculation, respond exactly:

I don't know based on the provided document.

9. Preserve source document names and page numbers returned
   by rbi_rag_tool.

10. Keep answers concise and factual.

11. Do not use outside knowledge to fill missing information.

12. If the user asks a general-knowledge question that is
    unrelated to the RBI document, do not answer it.

13. Do not perform arithmetic yourself when calculator can
    perform the calculation. Use the calculator tool.
"""

    # =====================================================
    # INITIAL MESSAGES
    # =====================================================

    messages = [
        SystemMessage(
            content=system_instruction
        ),
        HumanMessage(
            content=question
        ),
    ]

    # =====================================================
    # AGENT / TOOL-CALLING LOOP
    # =====================================================

    while True:

        # -------------------------------------------------
        # Ask LLM
        # -------------------------------------------------

        response = llm.invoke(messages)

        # -------------------------------------------------
        # DEBUG
        # -------------------------------------------------

        print("\n" + "=" * 70)
        print("DEBUG TOOL CALLS")
        print("=" * 70)

        print(response.tool_calls)

        print("\n" + "=" * 70)
        print("DEBUG RESPONSE")
        print("=" * 70)

        print(response.content)

        # -------------------------------------------------
        # No tool requested
        # -------------------------------------------------

        if not response.tool_calls:

            return response.content

        # -------------------------------------------------
        # Add assistant message
        # -------------------------------------------------

        messages.append(response)

        # -------------------------------------------------
        # Execute every requested tool
        # -------------------------------------------------

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            # =============================================
            # RBI RAG TOOL
            # =============================================

            if tool_name == "rbi_rag_tool":

                print(
                    "\n[Agent] Using RBI RAG tool..."
                )

                tool_result = rbi_rag_tool.invoke(
                    tool_args
                )

            # =============================================
            # CALCULATOR TOOL
            # =============================================

            elif tool_name == "calculator":

                print(
                    "\n[Agent] Using calculator..."
                )

                tool_result = calculator.invoke(
                    tool_args
                )

            # =============================================
            # UNKNOWN TOOL
            # =============================================

            else:

                tool_result = (
                    f"Unknown tool requested: {tool_name}"
                )

            # -------------------------------------------------
            # Add tool result to conversation
            # -------------------------------------------------

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RBI Financial Inclusion AI Agent")
    print("=" * 70)

    print("Available tools:")
    print("  1. RBI Financial Inclusion RAG")
    print("  2. Calculator")

    print()
    print("Type 'exit' or 'quit' to stop.")

    print("=" * 70)

    while True:

        # -------------------------------------------------
        # Get user question
        # -------------------------------------------------

        question = input("\nYou: ").strip()

        # -------------------------------------------------
        # Exit
        # -------------------------------------------------

        if question.lower() in {"exit", "quit"}:

            print("\nGoodbye!")

            break

        # -------------------------------------------------
        # Ignore empty input
        # -------------------------------------------------

        if not question:

            continue

        # -------------------------------------------------
        # Run agent
        # -------------------------------------------------

        try:

            answer = run_agent(question)

            print("\n" + "-" * 70)
            print("Assistant:")
            print("-" * 70)

            print(answer)

        except Exception as e:

            print("\n" + "-" * 70)
            print("ERROR")
            print("-" * 70)

            print(str(e))