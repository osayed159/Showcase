"""Retrieval methods, all with the signature (question, k) -> ranked hits."""

import re

from store import search

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
VENUE = re.compile(r"(?i)arXiv preprint|In Proceedings|pages \d+|pp\. \d+")


def looks_like_bibliography(text: str) -> bool:
    return len(YEAR.findall(text)) >= 5 and len(VENUE.findall(text)) >= 3


OVERFETCH = 3


def vector(question: str, k: int) -> list[dict]:
    return search(question, n_results=k)


def vector_no_bibliography(question: str, k: int) -> list[dict]:
    hits = search(question, n_results=k * OVERFETCH)
    return [h for h in hits if not looks_like_bibliography(h["text"])][:k]


_bm25 = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _get_bm25():
    global _bm25
    if _bm25 is None:
        from rank_bm25 import BM25Okapi

        from store import EmptyIndexError, get_collection

        everything = get_collection().get(include=["documents", "metadatas"])
        if not everything["documents"]:
            raise EmptyIndexError()
        _bm25 = {
            "ids": everything["ids"],
            "documents": everything["documents"],
            "metadatas": everything["metadatas"],
            "index": BM25Okapi([_tokenize(d) for d in everything["documents"]]),
        }
    return _bm25


def bm25(question: str, k: int) -> list[dict]:
    b = _get_bm25()
    scores = b["index"].get_scores(_tokenize(question))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "id": b["ids"][i],
            "text": b["documents"][i],
            "arxiv_id": b["metadatas"][i]["arxiv_id"],
            "page": b["metadatas"][i]["page"],
            "citation": b["metadatas"][i]["citation"],
            "title": b["metadatas"][i]["title"],
            "similarity": float(scores[i]),
        }
        for i in best
    ]


FUSION_CANDIDATES = 50

RRF_K = 60


def hybrid(question: str, k: int) -> list[dict]:
    """Vector and BM25 results merged with reciprocal rank fusion."""
    fused: dict[str, float] = {}
    by_id: dict[str, dict] = {}

    for results in (
        vector(question, FUSION_CANDIDATES),
        bm25(question, FUSION_CANDIDATES),
    ):
        for rank, hit in enumerate(results, start=1):
            fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1 / (RRF_K + rank)
            by_id.setdefault(hit["id"], hit)

    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**by_id[i], "similarity": fused[i]} for i in best]


RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

# FlashRank caches models in /tmp by default.
RERANK_CACHE = ".cache/flashrank"

RERANK_POOL = 20

_ranker = None


def _get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker

        _ranker = Ranker(model_name=RERANK_MODEL, cache_dir=RERANK_CACHE)
    return _ranker


def rerank(question: str, candidates: list[dict], k: int) -> list[dict]:
    from flashrank import RerankRequest

    request = RerankRequest(
        query=question,
        passages=[{"id": c["id"], "text": c["text"]} for c in candidates],
    )
    by_id = {c["id"]: c for c in candidates}
    ranked = _get_ranker().rerank(request)
    return [{**by_id[r["id"]], "similarity": float(r["score"])} for r in ranked[:k]]


def hybrid_reranked(question: str, k: int) -> list[dict]:
    return rerank(question, hybrid(question, RERANK_POOL), k)


def vector_reranked(question: str, k: int) -> list[dict]:
    return rerank(question, vector(question, RERANK_POOL), k)


retrieve = hybrid_reranked


def retrieve_in_paper(arxiv_id: str, question: str, k: int) -> list[dict]:
    return rerank(question, search(question, n_results=RERANK_POOL, arxiv_id=arxiv_id), k)


STRATEGIES = {
    "vector (baseline)": vector,
    "vector, no bibliography": vector_no_bibliography,
    "bm25 alone": bm25,
    "hybrid (vector + bm25)": hybrid,
    "vector -> rerank": vector_reranked,
    "hybrid -> rerank": hybrid_reranked,
}


def main() -> None:
    from eval_retrieval import evaluate

    results = {name: evaluate(fn, label=name, verbose=False) for name, fn in STRATEGIES.items()}

    metrics = list(next(iter(results.values())))
    print(f"\n{'strategy':<28}" + "".join(f"{m:>9}" for m in metrics))
    for name, scores in results.items():
        print(f"{name:<28}" + "".join(f"{scores[m]:>9.2f}" for m in metrics))


if __name__ == "__main__":
    main()
