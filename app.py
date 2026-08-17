import streamlit as st

from query import ask_question


st.set_page_config(
    page_title="RBI Financial Inclusion RAG",
    page_icon="📚",
    layout="centered"
)


st.title("📚 RBI Financial Inclusion RAG")
st.caption("Ask questions from the RBI National Strategy for Financial Inclusion document.")


question = st.text_input(
    "Ask a question",
    placeholder="e.g. What are the strategic objectives of financial inclusion?"
)


if question:

    with st.spinner("Searching the document..."):

        answer, sources = ask_question(question)

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Retrieved Sources")

    seen = set()

    for source in sources:

        key = (source["source"], source["page"])

        if key in seen:
            continue

        seen.add(key)

        if source["page"] is not None:
            st.write(
                f"📄 {source['source']} — Page {source['page']}"
            )
        else:
            st.write(f"📄 {source['source']}")