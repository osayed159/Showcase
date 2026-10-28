# Evaluation

Detailed results behind the summary in the [README](README.md).

## Test set

[data/eval_questions.json](data/eval_questions.json) holds 16 questions:

- **13 answerable questions.** For each, the answering page and a key fact were
  found by reading the papers.
- **3 unanswerable questions** about Bentham, Kraken and FPGAs. Each key term
  appears 0 times in the corpus, so the correct response is a refusal.

Every answer from the runs reported here is saved in
[data/eval_results.json](data/eval_results.json).

## Retrieval

The share of questions where the answering page appears in the top 1 or top 5
results, and the mean reciprocal rank (MRR).

| Method | Top 1 | Top 5 | MRR |
| --- | --- | --- | --- |
| Vector search | 0.15 | 0.54 | 0.31 |
| Vector search, reference lists removed | 0.15 | 0.54 | 0.31 |
| Keyword search (BM25) | 0.15 | 0.54 | 0.29 |
| Vector + keyword search, merged | 0.08 | 0.54 | 0.26 |
| Vector search, then re-ranked | 0.23 | 0.46 | 0.32 |
| **Vector + keyword search, then re-ranked** (used) | **0.31** | **0.77** | **0.45** |

The merged search found the answering page in its top 20 for 11 of 13
questions, compared with 9 for vector search. Re-ranking that larger pool
produced the best result.

Run with `python retrieve.py`.

## Answers

| | Fixed pipeline | Agent |
| --- | --- | --- |
| Correct | 10/13 | 12/13 |
| Correct and cites a paper that answers the question | 9/13 | 10/13 |
| Refused an answerable question | 1 | 0 |
| Refused an unanswerable question | 3/3 | 3/3 |
| Answers citing a passage that doesn't exist | 3/16 | 2/16 |
| Numbers flagged by the number check | 0 | 0 |
| Median time per question | 15 s | 23 s |

An answer counts as correct when it contains the key fact for its question.
This is a minimum bar: it does not check the rest of the answer.

These runs used an earlier version of the number check, which compared text
rather than whole numbers and missed some cases (for example, "3" inside
"2013"). The check has since been fixed; the answers were not re-run with it.

Times were measured on a laptop with an Intel i7-8750H, 16 GB of RAM and a
GTX 1050 Ti (4 GB), with the model partly on the CPU.

Run with `python eval_answers.py` (about 10 minutes). Each run is saved to
`data/eval_runs/`.

## Changes made during development

- **Text cleaning.** PDF extraction broke 1,520 words with ligature characters
  and split 4,623 words across lines. Hyphenated line breaks are rejoined only
  when the joined word appears elsewhere in the corpus, which keeps names such
  as GPT-4V intact (3,900 joined, 703 kept, measured before a later fix to
  glyph-name artifacts).
- **Reference-list filter.** Removing reference-list chunks from results changed
  no rankings, so it was not adopted.
- **Model choice.** For one test question, the base Qwen3-4B model generated
  between 695 and 5,818 hidden reasoning tokens depending on how thinking was
  disabled, and took over 70 seconds. The instruct version generates only the
  answer and takes 8 to 22 seconds.
- **Number check.** The model once reported a figure that its cited passage did
  not contain. Answers are now checked for numbers missing from their passages.
- **Agent refusals.** The first version of the agent often refused before
  searching. It now always runs a first search, and a refusal is accepted only
  after it has searched again on its own.

## Known issues

- The retrieval method and the agent's refusal handling were both chosen after
  seeing results on this test set, and one answer page was added to the key
  (DARE, page 3) after a run showed it also answered its question. The scores
  are therefore not from a blind test.
- One answer marked wrong (generating handwriting with handwriting-style fonts)
  is arguably a valid answer that the key fact did not cover.
- In answers that combine several papers, facts are sometimes cited to a
  passage from a different paper.
- Results can vary slightly between runs even at temperature 0, likely because
  the model runs partly on the CPU and partly on the GPU.
