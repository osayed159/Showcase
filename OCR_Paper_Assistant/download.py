"""Download the paper PDFs. Use --refresh to search arXiv again."""

import argparse
import dataclasses
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from arxiv_search import MAX_RESULTS, QUERY, RATE_LIMIT_SECONDS, Paper, search

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "papers"
METADATA_FILE = DATA_DIR / "papers.json"

PDF_MAGIC = b"%PDF"


def download_pdf(paper: Paper, destination: Path) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return False

    request = urllib.request.Request(
        paper.pdf_url,
        headers={"User-Agent": "rag-agent/0.1 (student project)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()

    if not content.startswith(PDF_MAGIC):
        raise ValueError(
            f"{paper.arxiv_id}: response is not a PDF "
            f"(starts with {content[:20]!r})"
        )

    temporary = destination.with_suffix(".part")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the paper corpus from arXiv.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Run the arXiv search again and replace data/papers.json with its results.",
    )
    args = parser.parse_args()
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Search results change over time, so default to the committed paper list.
    refresh = args.refresh or not METADATA_FILE.exists()
    if refresh:
        print(f"Searching arXiv for up to {MAX_RESULTS} papers...")
        papers = search(QUERY, max_results=MAX_RESULTS)
        print(f"Found {len(papers)}.\n")
    else:
        records = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        papers = [Paper(**record) for record in records]
        print(f"Downloading the {len(papers)} papers listed in {METADATA_FILE} (--refresh to search again).\n")

    downloaded, skipped, failed = 0, 0, []

    for i, paper in enumerate(papers, start=1):
        destination = PDF_DIR / paper.filename
        label = f"[{i:3}/{len(papers)}] {paper.arxiv_id}"

        try:
            if download_pdf(paper, destination):
                size_mb = destination.stat().st_size / 1_048_576
                print(f"{label}  downloaded  {size_mb:5.1f} MB  {paper.title[:50]}")
                downloaded += 1
                time.sleep(RATE_LIMIT_SECONDS)
            else:
                print(f"{label}  have it")
                skipped += 1
        except (urllib.error.URLError, ValueError, OSError) as err:
            print(f"{label}  FAILED      {err}")
            failed.append((paper.arxiv_id, str(err)))

    on_disk = [p for p in papers if (PDF_DIR / p.filename).exists()]
    if refresh:
        METADATA_FILE.write_text(
            json.dumps([dataclasses.asdict(p) for p in on_disk], indent=2),
            encoding="utf-8",
        )

    print(f"\nDownloaded {downloaded}, already had {skipped}, failed {len(failed)}.")
    print(f"Corpus: {len(on_disk)} PDFs in {PDF_DIR}")
    if refresh:
        print(f"Metadata: {METADATA_FILE}")
    if failed:
        print("\nFailures:")
        for arxiv_id, err in failed:
            print(f"  {arxiv_id}: {err}")


if __name__ == "__main__":
    main()
