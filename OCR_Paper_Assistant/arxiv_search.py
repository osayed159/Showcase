"""Search arXiv using only the standard library."""

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

API = "https://export.arxiv.org/api/query"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

RATE_LIMIT_SECONDS = 3.0

PAGE_SIZE = 100


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    pdf_url: str = ""
    categories: list[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return f"{self.arxiv_id.replace('/', '_')}.pdf"


def _text(entry: ET.Element, path: str) -> str:
    node = entry.find(path, NS)
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def _parse_entry(entry: ET.Element) -> Paper:
    raw_id = _text(entry, "atom:id")
    arxiv_id = raw_id.rsplit("/abs/", 1)[-1]

    pdf_url = ""
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
            break

    return Paper(
        arxiv_id=arxiv_id,
        title=_text(entry, "atom:title"),
        authors=[
            " ".join(name.text.split())
            for name in entry.findall("atom:author/atom:name", NS)
            if name.text
        ],
        abstract=_text(entry, "atom:summary"),
        published=_text(entry, "atom:published")[:10],
        pdf_url=pdf_url,
        categories=[
            c.get("term", "") for c in entry.findall("atom:category", NS)
        ],
    )


def search(query: str, max_results: int = 50, sort_by: str = "relevance") -> list[Paper]:
    papers: list[Paper] = []

    while len(papers) < max_results:
        want = min(PAGE_SIZE, max_results - len(papers))
        params = urllib.parse.urlencode(
            {
                "search_query": query,
                "start": len(papers),
                "max_results": want,
                "sortBy": sort_by,
                "sortOrder": "descending",
            }
        )

        request = urllib.request.Request(
            f"{API}?{params}",
            headers={"User-Agent": "rag-agent/0.1 (student project)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            feed = ET.fromstring(response.read())

        entries = feed.findall("atom:entry", NS)
        if not entries:
            break

        papers.extend(_parse_entry(e) for e in entries)

        total = feed.find("opensearch:totalResults", NS)
        if total is not None and total.text and len(papers) >= int(total.text):
            break

        if len(papers) < max_results:
            time.sleep(RATE_LIMIT_SECONDS)

    return papers[:max_results]


QUERY = (
    'abs:"optical character recognition" '
    'OR abs:"scene text recognition" '
    'OR abs:"handwritten text recognition" '
    'OR abs:"text spotting" '
    'OR abs:"document understanding"'
)

MAX_RESULTS = 40


def main() -> None:
    print(f"Query: {QUERY}\n")
    papers = search(QUERY, max_results=MAX_RESULTS)
    print(f"Found {len(papers)} papers:\n")

    for i, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        print(f"{i:3}. [{paper.arxiv_id}] {paper.published}  {paper.title}")
        print(f"     {authors}")
        print(f"     {', '.join(paper.categories[:4])}")
        print()


if __name__ == "__main__":
    main()
