"""Streamlit chat UI."""

import streamlit as st

from agent import ask_agent
from retrieve import retrieve
from store import EmptyIndexError

EXAMPLES = [
    "Compare how ABINet++ and KISS approach scene text recognition.",
    "Why is it hard to recognise text in dark, poorly lit photos?",
    "What error rate does the parameter-efficient fine-tuning approach reach on the IAM handwriting benchmark?",
]

st.set_page_config(page_title="OCR Paper Assistant", page_icon="📄")


@st.cache_resource(show_spinner="Loading search models (first start only, about 20 seconds)...")
def warm_up() -> bool:
    retrieve("optical character recognition", 1)
    return True


def render(result: dict) -> None:
    st.markdown(result["answer"])

    for number, sentence in result["unsupported_numbers"]:
        st.warning(f"**{number}** does not appear in the passages this sentence cites: \"{sentence[:120]}\"")
    if result["invalid_citations"]:
        st.warning(f"Cited passages that do not exist: {result['invalid_citations']}")

    if result["sources"]:
        with st.expander(f"Sources ({len(result['sources'])})"):
            for n, source in result["sources"]:
                aid = source["arxiv_id"]
                st.markdown(
                    f"**[{n}]** [{source['title']}](https://arxiv.org/abs/{aid}) "
                    f"— arXiv:{aid}, page {source['page']}"
                )
    count = len(result["steps"])
    with st.expander(f"How it searched ({count} step{'s' if count != 1 else ''})"):
        for i, step in enumerate(result["steps"], start=1):
            st.code(f"{i}. {step}", language=None)
    st.caption(f"{result['seconds']:.0f} s")


try:
    warm_up()
except EmptyIndexError as err:
    st.error(str(err))
    st.stop()

st.title("OCR Paper Assistant")
st.caption(
    "Answers questions about 40 OCR research papers from arXiv, citing paper "
    "and page, and says when the papers don't contain the answer. Runs "
    "entirely locally (Qwen3-4B via Ollama). Each question is answered on its "
    "own: it does not remember earlier questions."
)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render(turn)

clicked = None
if not st.session_state.history:
    st.markdown("**Try one:**")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            clicked = example

question = st.chat_input("Ask about the papers") or clicked
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the papers and reading... (about 30 seconds)"):
            result = ask_agent(question, verbose=False)
        render(result)
    st.session_state.history.append(result)
    # Redraw without the example buttons, which were drawn before this answer.
    st.rerun()
