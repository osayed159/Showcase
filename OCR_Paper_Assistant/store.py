"""Chroma vector index of the chunks."""

import chromadb

from chunk import Chunk, chunk_corpus
from embed import get_embedder

DB_PATH = "chroma_db"
COLLECTION_NAME = "papers"

# Chroma defaults to squared L2.
DISTANCE_METRIC = "cosine"

WRITE_BATCH = 500


class EmptyIndexError(RuntimeError):
    """Raised instead of returning no results when the index hasn't been built."""

    def __init__(self) -> None:
        super().__init__(
            "The search index is empty. Build it with: python download.py && python store.py"
        )


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=DB_PATH)


def get_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedder(),
        metadata={"hnsw:space": DISTANCE_METRIC},
    )


def build_index(chunks: list[Chunk] | None = None, verbose: bool = True) -> int:
    if chunks is None:
        chunks = chunk_corpus()

    collection = get_collection()

    for start in range(0, len(chunks), WRITE_BATCH):
        batch = chunks[start:start + WRITE_BATCH]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "arxiv_id": c.arxiv_id,
                    "title": c.title,
                    "page": c.page,
                    "citation": c.citation,
                }
                for c in batch
            ],
        )
        if verbose:
            done = min(start + WRITE_BATCH, len(chunks))
            print(f"  indexed {done:,} / {len(chunks):,}")

    return collection.count()


def search(query: str, n_results: int = 5, arxiv_id: str | None = None) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        raise EmptyIndexError()
    where = {"arxiv_id": arxiv_id} if arxiv_id else None
    response = collection.query(query_texts=[query], n_results=n_results, where=where)

    return [
        {
            "id": chunk_id,
            "text": document,
            "arxiv_id": metadata["arxiv_id"],
            "page": metadata["page"],
            "citation": metadata["citation"],
            "title": metadata["title"],
            "similarity": 1 - distance,
        }
        for chunk_id, document, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        )
    ]


def main() -> None:
    chunks = chunk_corpus()
    print(f"Indexing {len(chunks):,} chunks into {DB_PATH}/ ...")
    total = build_index(chunks)
    print(f"Collection holds {total:,} vectors.\n")

    question = "which datasets are used to evaluate handwriting recognition?"
    print(f"Query: {question}\n")
    for i, hit in enumerate(search(question), start=1):
        print(f"{i}. [{hit['similarity']:.3f}] {hit['citation']}")
        print(f"   {hit['title'][:66]}")
        print(f"   {hit['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
