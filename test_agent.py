# from agent import run_agent


# questions = [
#     "What is financial inclusion?",
#     "What are the strategic objectives of financial inclusion in India?",
#     "What is the capital of France?"
# ]


# for question in questions:

#     print("\n" + "=" * 70)
#     print("QUESTION:", question)
#     print("=" * 70)

#     answer = run_agent(question)

#     print("\nANSWER:")
#     print(answer)

from agent import run_agent


questions = [
    "What is financial inclusion?",
    "What is 125 multiplied by 48?",
    "Which product has the highest revenue?",
    "What is the total revenue?",
    "Which category generated the most revenue?",
    "What is the capital of France?"
]

for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    answer = run_agent(question)

    print("\nANSWER:")
    print(answer)

   