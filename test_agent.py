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
    "What are the strategic objectives of financial inclusion in India?",
    "What is 125 multiplied by 48?",
    "What is 15% of 800?",
    "What is the capital of France?"
]


for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    answer = run_agent(question)

    print("\nANSWER:")
    print(answer)
