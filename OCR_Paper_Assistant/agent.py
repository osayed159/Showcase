"""Tool-calling agent that chooses its own searches."""

import json
import sys
import time

import ollama

from answer import CITATION, REFUSAL, unsupported_numbers
from llm import MODEL, strip_thinking
from retrieve import retrieve, retrieve_in_paper

PASSAGES_PER_SEARCH = 3

CONTEXT_TOKENS = 8192

MAX_TOOL_CALLS = 6

NUDGE = (
    "Before answering that you don't know, search further: use search_in_paper "
    "for each paper or method named in the question, or search_papers with "
    "different words."
)

SEED_PASSAGES = 5

PAPERS_FILE = "data/papers.json"

AGENT_PROMPT = f"""You answer questions about a collection of OCR research papers. You cannot see the papers directly: use the tools to find passages, then answer from them.

How to search:
- A first search for the question has already been run for you. Read its passages first.
- If they are not enough, search more. To ask about one specific paper or method, call search_in_paper with its name, e.g. paper="KISS".
- To compare several papers or methods, search for EACH one separately. Never use one search for the whole comparison.
- If results do not contain what you need, search again with different words.
- Stop searching once you have enough to answer.

Rules for your answer:
- Use only information stated in the passages the tools returned. Do not add facts from memory.
- Copy every number exactly as written in the passage you cite. Never compute, convert or estimate a number.
- Do not compare results from different papers unless a passage itself makes that comparison.
- After each claim, cite the passage number in square brackets, like [4].
- Only after searching: if the passages still do not contain the answer, reply exactly: {REFUSAL}
- Be concise: at most six sentences."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search ALL papers for passages relevant to a query. Use for general questions, or when you do not know which paper holds the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in a few words."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_paper",
            "description": "Search inside ONE paper only. Use when the question names a specific paper or method, e.g. paper='KISS' or paper='ABINet++'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper": {"type": "string", "description": "The paper's name as it appears in its title (e.g. 'KISS', 'DARE'), or its arxiv_id."},
                    "query": {"type": "string", "description": "What to look for inside that paper."},
                },
                "required": ["paper", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_papers",
            "description": "List every paper in the collection with its arxiv_id and title. Use when you are unsure what a paper is called.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class Run:
    def __init__(self) -> None:
        self.passages: list[dict] = []
        self.number_of: dict[str, int] = {}
        self.steps: list[str] = []

    def number(self, hits: list[dict]) -> str:
        blocks = []
        for hit in hits:
            if hit["id"] not in self.number_of:
                self.passages.append(hit)
                self.number_of[hit["id"]] = len(self.passages)
            n = self.number_of[hit["id"]]
            blocks.append(f"[{n}] {hit['citation']} -- {hit['title']}\n{hit['text']}")
        return "\n\n".join(blocks) if blocks else "No passages found."


def load_papers() -> dict[str, str]:
    with open(PAPERS_FILE, encoding="utf-8") as f:
        return {p["arxiv_id"]: p["title"] for p in json.load(f)}


def resolve_paper(name: str, papers: dict[str, str]) -> tuple[str | None, str]:
    """Map a paper name or arxiv_id to an arxiv_id, or return an error for the model."""
    name = name.strip()
    if name in papers:
        return name, ""
    matches = [aid for aid, title in papers.items() if name.lower() in title.lower()]
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        return None, f"Error: no paper title contains {name!r}. Call list_papers to see the titles."
    options = "\n".join(f"{aid} | {papers[aid]}" for aid in matches)
    return None, f"Error: {name!r} matches several papers. Use one of these arxiv_ids:\n{options}"


def run_tool(run: Run, name: str, args: dict) -> str:
    papers = load_papers()

    if name == "list_papers":
        return "\n".join(f"{aid} | {title}" for aid, title in papers.items())

    query = str(args.get("query", "")).strip()
    if not query:
        return "Error: 'query' is required. Say what to look for."

    if name == "search_papers":
        return run.number(retrieve(query, PASSAGES_PER_SEARCH))

    if name == "search_in_paper":
        arxiv_id, error = resolve_paper(str(args.get("paper", "")), papers)
        if error:
            return error
        header = f"Results from {arxiv_id}: {papers[arxiv_id]}\n\n"
        return header + run.number(retrieve_in_paper(arxiv_id, query, PASSAGES_PER_SEARCH))

    return f"Error: there is no tool called {name!r}."


def ask_agent(question: str, verbose: bool = True) -> dict:
    run = Run()
    started = time.perf_counter()
    messages = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": question},
    ]

    # Run the first search in code; left to the model, it often refused without searching.
    seed_args = {"query": question}
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search_papers", "arguments": seed_args}}],
    })
    messages.append({
        "role": "tool",
        "content": run.number(retrieve(question, SEED_PASSAGES)),
        "tool_name": "search_papers",
    })
    run.steps.append(f"search_papers(query={question!r}) [automatic]")
    if verbose:
        print(f"  step 1: search_papers(query={question!r}) [automatic]")

    calls = 1
    max_prompt = 0
    nudged = False

    while True:
        out_of_calls = calls >= MAX_TOOL_CALLS
        if out_of_calls:
            messages.append({"role": "user", "content": "Answer now, using only the passages found so far."})

        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=None if out_of_calls else TOOLS,
            options={"temperature": 0.0, "num_ctx": CONTEXT_TOKENS},
        )
        max_prompt = max(max_prompt, response.prompt_eval_count or 0)
        message = response.message

        if not message.tool_calls:
            # Accept a refusal only after the model has searched on its own.
            refusing = REFUSAL.lower() in (message.content or "").lower()
            if refusing and calls == 1 and not nudged and not out_of_calls:
                messages.append(message)
                messages.append({"role": "user", "content": NUDGE})
                nudged = True
                if verbose:
                    print("  (refused without searching -- nudged to search)")
                continue
            break

        messages.append(message)
        for call in message.tool_calls:
            calls += 1
            name, args = call.function.name, dict(call.function.arguments)
            result = run_tool(run, name, args)
            shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
            run.steps.append(f"{name}({shown})")
            if verbose:
                print(f"  step {calls}: {name}({shown})")
            messages.append({"role": "tool", "content": result, "tool_name": name})

    text = strip_thinking(message.content)
    cited = sorted({int(n) for n in CITATION.findall(text)})

    return {
        "question": question,
        "answer": text,
        "refused": REFUSAL.lower() in text.lower(),
        "steps": run.steps,
        "sources": [(n, run.passages[n - 1]) for n in cited if 1 <= n <= len(run.passages)],
        "invalid_citations": [n for n in cited if not 1 <= n <= len(run.passages)],
        "unsupported_numbers": unsupported_numbers(text, run.passages),
        "max_prompt_tokens": max_prompt,
        "seconds": time.perf_counter() - started,
    }


def show(result: dict) -> None:
    print(f"A: {result['answer']}")
    if result["sources"]:
        print("Sources:")
        for n, source in result["sources"]:
            print(f"  [{n}] {source['citation']}  {source['title'][:60]}")
    if result["invalid_citations"]:
        print(f"  !! cited passages that do not exist: {result['invalid_citations']}")
    for number, sentence in result["unsupported_numbers"]:
        print(f"  !! {number} is not in the passages this sentence cites: \"{sentence[:90]}\"")
    print(
        f"({len(result['steps'])} tool calls, largest prompt "
        f"{result['max_prompt_tokens']} of {CONTEXT_TOKENS} tokens, "
        f"{result['seconds']:.0f}s)"
    )


DEMO_QUESTIONS = [
    "Compare how ABINet++ and KISS approach scene text recognition.",
    "What datasets does the DARE paper use?",
]


def main() -> None:
    for question in sys.argv[1:] or DEMO_QUESTIONS:
        print(f"\nQ: {question}")
        show(ask_agent(question))


if __name__ == "__main__":
    main()
