"""Text embeddings with all-MiniLM-L6-v2 (ONNX, no PyTorch)."""

import time

from chromadb.utils import embedding_functions

EMBEDDING_DIMENSIONS = 384
MAX_INPUT_TOKENS = 256

BATCH_SIZE = 64

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = embedding_functions.DefaultEmbeddingFunction()
    return _embedder


def embed_texts(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    embedder = get_embedder()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(embedder(batch))
    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    # The vectors are unit length, so the dot product is the cosine similarity.
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    examples = [
        "optical character recognition for printed documents",
        "OCR systems that read scanned text",
        "handwritten text recognition in historical archives",
        "off-road racing vehicle detection datasets",
        "the price of tomatoes in Cairo",
    ]
    vectors = embed_texts(examples)

    print(f"Dimensions: {len(vectors[0])}")
    print("\nSimilarity to: " + examples[0])
    for text, vector in zip(examples[1:], vectors[1:]):
        print(f"  {cosine_similarity(vectors[0], vector):.2f}  {text}")

    from chunk import chunk_corpus

    chunks = chunk_corpus()
    sample = [c.text for c in chunks[:BATCH_SIZE * 2]]

    started = time.perf_counter()
    embed_texts(sample)
    elapsed = time.perf_counter() - started

    rate = len(sample) / elapsed
    print(f"\nChunks in corpus: {len(chunks):,}")
    print(f"Rate: {rate:.0f} chunks/second")
    print(f"Estimated full corpus: {len(chunks) / rate:.0f} seconds")


if __name__ == "__main__":
    main()
