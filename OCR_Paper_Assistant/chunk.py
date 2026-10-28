"""Split pages into overlapping, sentence-aligned chunks."""

import re
from dataclasses import dataclass

from ingest import Page, load_corpus

# Characters; keeps chunks under the embedding model's 256-token limit.
CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

MIN_CHUNK_SIZE = 100

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    arxiv_id: str
    title: str
    page: int
    index: int

    @property
    def id(self) -> str:
        return f"{self.arxiv_id}_p{self.page}_c{self.index}"

    @property
    def citation(self) -> str:
        return f"arXiv:{self.arxiv_id} p.{self.page}"


def split_sentences(text: str) -> list[str]:
    # Leave room for the overlap carried into each chunk.
    longest = CHUNK_SIZE - CHUNK_OVERLAP

    units: list[str] = []
    for sentence in SENTENCE_END.split(text):
        sentence = " ".join(sentence.split())
        if not sentence:
            continue
        while len(sentence) > longest:
            cut = sentence.rfind(" ", 0, longest)
            if cut <= 0:
                cut = longest
            units.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            units.append(sentence)
    return units


def chunk_page(page: Page) -> list[Chunk]:
    sentences = split_sentences(page.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    length = 0

    def flush() -> None:
        nonlocal current, length
        text = " ".join(current).strip()
        if len(text) >= MIN_CHUNK_SIZE:
            chunks.append(
                Chunk(
                    text=text,
                    arxiv_id=page.arxiv_id,
                    title=page.title,
                    page=page.page,
                    index=len(chunks),
                )
            )

        carried: list[str] = []
        carried_length = 0
        for sentence in reversed(current):
            if carried_length + len(sentence) > CHUNK_OVERLAP:
                break
            carried.insert(0, sentence)
            carried_length += len(sentence) + 1
        current = carried
        length = carried_length

    for sentence in sentences:
        if length + len(sentence) + 1 > CHUNK_SIZE and current:
            flush()
        current.append(sentence)
        length += len(sentence) + 1

    if current:
        flush()

    return chunks


def chunk_corpus(pages: list[Page] | None = None) -> list[Chunk]:
    if pages is None:
        pages = load_corpus()
    return [chunk for page in pages for chunk in chunk_page(page)]


def main() -> None:
    pages = load_corpus()
    chunks = chunk_corpus(pages)

    lengths = sorted(len(c.text) for c in chunks)
    p95 = lengths[int(len(lengths) * 0.95)]

    print(f"\nPages:   {len(pages)}")
    print(f"Chunks:  {len(chunks)}   ({len(chunks) / len(pages):.1f} per page)")
    print(
        f"Size:    min {lengths[0]}  median {lengths[len(lengths) // 2]}  "
        f"p95 {p95}  max {lengths[-1]}"
    )

    oversized = sum(1 for n in lengths if n > 1100)
    print(f"Over 1,100 chars: {oversized}  (would risk truncation)")

    sample = next(c for c in chunks if c.index == 1)
    previous = next(
        c for c in chunks
        if c.arxiv_id == sample.arxiv_id and c.page == sample.page and c.index == 0
    )
    print(f"\n--- overlap check: {sample.citation} ---")
    print(f"chunk 0 ends:   ...{previous.text[-160:]}")
    print(f"chunk 1 begins: {sample.text[:160]}...")


if __name__ == "__main__":
    main()
