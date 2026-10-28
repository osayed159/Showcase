"""Single-search pipeline: retrieve passages, ask the model, resolve citations."""

import re
import sys
import time

import ollama

from llm import MODEL, strip_thinking
from retrieve import retrieve

TOP_K = 5

# Ollama silently drops the start of a prompt longer than this, so set it explicitly.
CONTEXT_TOKENS = 4096

SYSTEM_PROMPT = """You answer questions about OCR research papers using ONLY the numbered passages provided.

Rules:
- Use only information stated in the passages. Do not add facts from memory.
- Copy every number exactly as it is written in the passage you cite. Never compute, convert or estimate a number.
- Do not compare results from different papers unless a passage itself makes that comparison.
- After each claim, cite the passage it came from in square brackets, like [2]. Cite several if needed, like [1][3].
- If the passages do not contain the answer, reply exactly: I don't know based on these papers.
- Be concise: at most four sentences."""

REFUSAL = "I don't know based on these papers."

CITATION = re.compile(r"\[(\d+)\]")

NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")

ANY_NUMBER = re.compile(r"\d+(?:\.\d+)?")

THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")

SENTENCE = re.compile(r"(?<=[.!?])\s+")


def unsupported_numbers(text: str, passages: list[dict]) -> list[tuple[str, str]]:
    """Numbers in the answer that its cited passages (or all, if none cited) don't contain."""
    problems = []
    for sentence in _sentences(text):
        cited = [int(n) for n in CITATION.findall(sentence) if 1 <= int(n) <= len(passages)]
        sources = [passages[n - 1] for n in cited] if cited else passages
        evidence = set()
        for passage in sources:
            evidence |= _numbers_in(passage["text"])
        for number in NUMBER.findall(_drop_thousands(CITATION.sub("", sentence))):
            if number not in evidence:
                problems.append((number, sentence.strip()))
    return problems


def _drop_thousands(text: str) -> str:
    return THOUSANDS.sub("", text)


def _numbers_in(text: str) -> set[str]:
    return set(ANY_NUMBER.findall(_drop_thousands(text)))


def _sentences(text: str) -> list[str]:
    # Attach a citation written after the full stop ("... on IAM. [1]") to its sentence.
    sentences: list[str] = []
    for part in SENTENCE.split(text):
        if sentences and not CITATION.sub("", part).strip(" .,;"):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences


def build_prompt(question: str, passages: list[dict]) -> str:
    blocks = [
        f"[{i}] {p['citation']} -- {p['title']}\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    ]
    return "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def answer(question: str, k: int = TOP_K) -> dict:
    started = time.perf_counter()
    passages = retrieve(question, k)
    retrieved_at = time.perf_counter()

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, passages)},
        ],
        options={"temperature": 0.0, "num_ctx": CONTEXT_TOKENS},
    )
    text = strip_thinking(response.message.content)
    finished = time.perf_counter()

    cited = sorted({int(n) for n in CITATION.findall(text)})
    sources = [passages[n - 1] for n in cited if 1 <= n <= len(passages)]
    invalid = [n for n in cited if not 1 <= n <= len(passages)]

    return {
        "question": question,
        "answer": text,
        "refused": REFUSAL.lower() in text.lower(),
        "sources": sources,
        "cited_numbers": cited,
        "invalid_citations": invalid,
        "unsupported_numbers": unsupported_numbers(text, passages),
        "passages": passages,
        "prompt_tokens": response.prompt_eval_count,
        "retrieval_seconds": retrieved_at - started,
        "generation_seconds": finished - retrieved_at,
    }


def show(result: dict) -> None:
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    if result["sources"]:
        print("Sources:")
        for n, source in zip(result["cited_numbers"], result["sources"]):
            print(f"  [{n}] {source['citation']}  {source['title'][:60]}")
    if result["invalid_citations"]:
        print(f"  !! cited passages that do not exist: {result['invalid_citations']}")
    for number, sentence in result["unsupported_numbers"]:
        print(f"  !! {number} is not in the passages this sentence cites: \"{sentence[:90]}\"")
    print(
        f"({result['prompt_tokens']} prompt tokens of {CONTEXT_TOKENS}; "
        f"retrieval {result['retrieval_seconds']:.1f}s, "
        f"generation {result['generation_seconds']:.1f}s)"
    )


DEMO_QUESTIONS = [
    "What error rate does the parameter-efficient fine-tuning approach reach on the IAM handwriting benchmark?",
    "Where does GPT-4V fall short when reading text in images?",
    "What is the capital of Egypt?",
]


def main() -> None:
    questions = sys.argv[1:] or DEMO_QUESTIONS
    for question in questions:
        show(answer(question))


if __name__ == "__main__":
    main()
