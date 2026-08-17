import re

from query import retrieve_context


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = text.lower()

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# SENTENCE SPLITTING
# =========================================================

def split_sentences(text):

    # Remove source section from the answer
    if "Sources:" in text:
        text = text.split("Sources:")[0]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# =========================================================
# GROUNDING CHECK
# =========================================================

def check_groundedness(answer, context):

    normalized_context = normalize_text(
        context
    )

    sentences = split_sentences(
        answer
    )

    if not sentences:

        return {
            "grounded": False,
            "score": 0.0,
            "supported_sentences": 0,
            "total_sentences": 0
        }

    supported = 0

    for sentence in sentences:

        normalized_sentence = normalize_text(
            sentence
        )

        words = normalized_sentence.split()

        # Ignore extremely short sentences
        if len(words) < 3:
            continue

        # Create meaningful word set
        meaningful_words = {
            word
            for word in words
            if len(word) >= 4
        }

        if not meaningful_words:
            continue

        matched_words = sum(
            word in normalized_context
            for word in meaningful_words
        )

        overlap_ratio = (
            matched_words
            / len(meaningful_words)
        )

        # Sentence considered supported
        # when enough meaningful words
        # appear in retrieved context.
        if overlap_ratio >= 0.40:

            supported += 1

    score = (
        supported
        / len(sentences)
        * 100
    )

    return {
        "grounded": score >= 70,
        "score": score,
        "supported_sentences": supported,
        "total_sentences": len(sentences)
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    question = "What is financial inclusion?"

    answer = """
    Financial inclusion is the access to formal financial
    services that supports economic growth and poverty
    alleviation.
    """

    context, _ = retrieve_context(
        question
    )

    result = check_groundedness(
        answer,
        context
    )

    print("=" * 60)
    print("GROUNDING TEST")
    print("=" * 60)

    print(
        "Question:",
        question
    )

    print(
        "Grounded:",
        result["grounded"]
    )

    print(
        "Grounding Score:",
        f"{result['score']:.2f}%"
    )

    print(
        "Supported Sentences:",
        result["supported_sentences"]
    )

    print(
        "Total Sentences:",
        result["total_sentences"]
    )