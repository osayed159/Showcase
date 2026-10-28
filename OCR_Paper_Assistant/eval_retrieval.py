"""Score retrieval (hit@k, MRR) against the evaluation questions."""

import json
from pathlib import Path
from typing import Callable

QUESTIONS_FILE = Path("data/eval_questions.json")

MAX_K = 10

REPORT_KS = (1, 3, 5, 10)

Retriever = Callable[[str, int], list[dict]]


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if include_unanswerable:
        return questions
    return [q for q in questions if q.get("answerable", True)]


def first_relevant_rank(hits: list[dict], relevant: list[dict]) -> int | None:
    targets = {(r["arxiv_id"], r["page"]) for r in relevant}
    for rank, hit in enumerate(hits, start=1):
        if (hit["arxiv_id"], hit["page"]) in targets:
            return rank
    return None


def evaluate(retrieve: Retriever, label: str = "", verbose: bool = True) -> dict:
    questions = load_questions()
    ranks: list[int | None] = []

    if verbose:
        print(f"\n{label or 'retriever'}")
        print(f"{'rank':>5}  question")

    for q in questions:
        hits = retrieve(q["question"], MAX_K)
        rank = first_relevant_rank(hits, q["relevant"])
        ranks.append(rank)
        if verbose:
            shown = str(rank) if rank else "MISS"
            print(f"{shown:>5}  [{q['type']:<10}] {q['question'][:70]}")

    n = len(questions)
    scores = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in REPORT_KS}
    scores["MRR"] = sum(1 / r for r in ranks if r) / n

    if verbose:
        print("\n" + "  ".join(f"{name} {value:.2f}" for name, value in scores.items()))
    return scores


def main() -> None:
    from store import search

    evaluate(search, label="pure vector search (baseline)")


if __name__ == "__main__":
    main()
