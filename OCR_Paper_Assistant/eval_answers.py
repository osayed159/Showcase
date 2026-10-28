"""Score the pipeline's and agent's answers. Pass a system name to run only that one."""

import json
import sys
import time
from pathlib import Path

from eval_retrieval import load_questions

RUNS_DIR = Path("data/eval_runs")


def score(question: dict, result: dict) -> dict:
    text = result["answer"].lower()
    answerable = question.get("answerable", True)
    refused = result["refused"]

    sources = [s[1] if isinstance(s, tuple) else s for s in result["sources"]]
    relevant_papers = {r["arxiv_id"] for r in question["relevant"]}
    cites_relevant = any(s["arxiv_id"] in relevant_papers for s in sources)

    correct = answerable and not refused and any(
        term.lower() in text for term in question["must_include"]
    )
    return {
        "id": question["id"],
        "answerable": answerable,
        "refused": refused,
        "correct": correct,
        "grounded": correct and cites_relevant,
        "unsupported_numbers": len(result["unsupported_numbers"]),
        "invalid_citations": len(result["invalid_citations"]),
        "seconds": result.get("seconds")
        or result.get("retrieval_seconds", 0) + result.get("generation_seconds", 0),
        "answer": result["answer"],
    }


def summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    absent = [r for r in rows if not r["answerable"]]
    return {
        "correct": sum(r["correct"] for r in answerable),
        "grounded": sum(r["grounded"] for r in answerable),
        "false_refusals": sum(r["refused"] for r in answerable),
        "answerable": len(answerable),
        "correct_refusals": sum(r["refused"] for r in absent),
        "unanswerable": len(absent),
        "unsupported_numbers": sum(r["unsupported_numbers"] for r in rows),
        "invalid_citations": sum(r["invalid_citations"] for r in rows),
        "avg_seconds": sum(r["seconds"] for r in rows) / len(rows),
    }


def run_system(name: str, ask, questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        row = score(q, ask(q["question"]))
        mark = (
            "refused " if row["refused"] else "CORRECT " if row["correct"] else "wrong   "
        ) if row["answerable"] else ("refused " if row["refused"] else "ANSWERED")
        print(f"  {name:<8} {mark} {row['seconds']:5.0f}s  {q['id']}", flush=True)
        rows.append(row)
    return rows


def main() -> None:
    from agent import ask_agent
    from answer import answer

    questions = load_questions(include_unanswerable=True)
    systems = {
        "pipeline": answer,
        "agent": lambda q: ask_agent(q, verbose=False),
    }
    chosen = sys.argv[1:] or list(systems)

    started = time.perf_counter()
    all_rows = {name: run_system(name, systems[name], questions) for name in chosen}

    print(f"\n{'':<10}{'correct':>9}{'grounded':>10}{'false refusal':>15}{'refused absent':>16}{'bad numbers':>13}{'avg s':>7}")
    summaries = {}
    for name, rows in all_rows.items():
        s = summaries[name] = summarise(rows)
        print(
            f"{name:<10}"
            f"{s['correct']:>5}/{s['answerable']:<3}"
            f"{s['grounded']:>6}/{s['answerable']:<3}"
            f"{s['false_refusals']:>11}/{s['answerable']:<3}"
            f"{s['correct_refusals']:>12}/{s['unanswerable']:<3}"
            f"{s['unsupported_numbers']:>13}"
            f"{s['avg_seconds']:>7.0f}"
        )
    print(f"\n({(time.perf_counter() - started) / 60:.0f} minutes)")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_file = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    run_file.write_text(
        json.dumps({"summary": summaries, "rows": all_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Answers saved to {run_file}")


if __name__ == "__main__":
    main()
