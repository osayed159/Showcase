"""Extract and clean text from the downloaded PDFs, one record per page."""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pypdf

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "papers"
METADATA_FILE = DATA_DIR / "papers.json"

MIN_CHARS_PER_PAGE = 100


@dataclass
class Page:
    text: str
    arxiv_id: str
    title: str
    authors: list[str]
    page: int

    @property
    def citation(self) -> str:
        return f"arXiv:{self.arxiv_id} p.{self.page}"


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = GLYPH_NAME.sub(r"\1\2", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# Some PDFs emit ligatures as glyph names, e.g. "handwri/t_ten".
GLYPH_NAME = re.compile(r"/([a-z])_([a-z])")


HYPHEN_BREAK = re.compile(r"([A-Za-z]+)-\n([A-Za-z0-9]+)")


def build_vocabulary(pages: list[Page]) -> set[str]:
    return {
        word.lower()
        for page in pages
        for word in re.findall(r"[A-Za-z]{2,}", page.text)
    }


def dehyphenate(text: str, vocabulary: set[str]) -> str:
    """Rejoin a line-break hyphen only if the joined word appears elsewhere in the corpus."""

    def decide(match: re.Match) -> str:
        head, tail = match.group(1), match.group(2)
        if (head + tail).lower() in vocabulary:
            return head + tail
        return f"{head}-{tail}"

    return HYPHEN_BREAK.sub(decide, text)


def load_metadata() -> dict[str, dict]:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"{METADATA_FILE} not found -- run `python download.py` first."
        )
    records = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return {record["arxiv_id"]: record for record in records}


def load_papers() -> Iterator[Page]:
    metadata = load_metadata()

    for arxiv_id, record in metadata.items():
        pdf_path = PDF_DIR / f"{arxiv_id.replace('/', '_')}.pdf"
        if not pdf_path.exists():
            continue

        try:
            reader = pypdf.PdfReader(pdf_path)
        except Exception as err:
            print(f"  skipped {arxiv_id}: cannot read ({err})")
            continue

        for index, pdf_page in enumerate(reader.pages, start=1):
            try:
                raw = pdf_page.extract_text() or ""
            except Exception as err:
                print(f"  skipped {arxiv_id} p.{index}: {err}")
                continue

            text = normalise(raw)
            if len(text) < MIN_CHARS_PER_PAGE:
                continue

            yield Page(
                text=text,
                arxiv_id=arxiv_id,
                title=record["title"],
                authors=record["authors"],
                page=index,
            )


def load_corpus() -> list[Page]:
    pages = list(load_papers())
    vocabulary = build_vocabulary(pages)
    for page in pages:
        page.text = dehyphenate(page.text, vocabulary)
    return pages


def main() -> None:
    pages = load_corpus()

    if not pages:
        print("No pages extracted. Has download.py run?")
        return

    papers = {p.arxiv_id for p in pages}
    lengths = sorted(len(p.text) for p in pages)
    total_chars = sum(lengths)

    print(f"\nPapers:      {len(papers)}")
    print(f"Pages:       {len(pages)}")
    print(f"Characters:  {total_chars:,}")
    print(
        f"Per page:    min {lengths[0]:,}  "
        f"median {lengths[len(lengths) // 2]:,}  max {lengths[-1]:,}"
    )
    print(f"Avg pages per paper: {len(pages) / len(papers):.1f}")

    ligatures = sum(len(re.findall(r"[ﬀ-ﬆ]", p.text)) for p in pages)
    breaks = sum(len(HYPHEN_BREAK.findall(p.text)) for p in pages)
    print(f"\nRemaining ligatures:        {ligatures}")
    print(f"Remaining hyphen breaks:    {breaks}")

    sample = pages[len(pages) // 2]
    print(f"\n--- sample: {sample.citation} ---")
    print(f"{sample.title[:70]}")
    print(f"{sample.text[:500]}")


if __name__ == "__main__":
    main()
