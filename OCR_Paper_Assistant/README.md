# OCR Paper Assistant

A question-answering agent for 40 OCR research papers from arXiv. It answers
from the papers, cites its sources by paper and page, and says it doesn't know
when the papers don't contain the answer. It runs locally with no paid API.

```
Q: What error rate does the parameter-efficient fine-tuning approach reach
   on the IAM handwriting benchmark?

A: The parameter-efficient fine-tuning approach reaches a Character Error Rate
   (CER) of 4.02 on the IAM dataset [2].

   [2] arXiv:2404.12734v4 p.1  Mixed Text Recognition with Efficient
       Parameter Fine-Tuning and Transformer
```

## How it works

```
arXiv papers ──▶ text, page by page ──▶ 2,777 chunks ──▶ embeddings in Chroma

question ──▶ vector + keyword search ──▶ re-rank ──▶ top passages
                                                          │
                          Qwen3-4B (Ollama) answers, citing its sources
```

1. **Indexing.** The papers are downloaded from arXiv, converted to text page by
   page, split into chunks of about 1,000 characters, and embedded into a Chroma
   vector database.
2. **Retrieval.** A question is matched against the chunks with both vector
   search and keyword search (BM25). A cross-encoder re-ranks the top 20, and
   the best passages are passed to the model.
3. **Answering.** An agent built on Qwen3-4B decides what to search for. It can
   search all papers or inside one named paper, which helps with questions
   that compare papers. Numbers in an answer are checked against the passages
   it cites, and unsupported ones are flagged.

## Evaluation

Tested on 13 questions with known answers and 3 questions whose answers are not
in the papers:

- Retrieval found the right page in the top 5 for **77%** of questions, up from
  54% with vector search alone.
- The agent answered **12 of 13** answerable questions correctly.
- It refused all **3** questions whose answers are not in the papers.
- The automatic number check flagged **no** unsupported numbers.

Full results, the methods compared, and notes on what changed along the way are
in [EVALUATION.md](EVALUATION.md).

## Running it

Tested on Python 3.13 and Ollama 0.40.11, on a laptop with 16 GB of RAM and a
4 GB NVIDIA GPU. The GPU is optional but makes answers faster. Allow about 4 GB
of disk: 2.5 GB for the model, 0.7 GB for Python packages, 230 MB for the
papers and about 230 MB for the index and search models.

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M

python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python download.py                # download the 40 papers (~3 min)
python store.py                   # build the search index (~2 min)
```

Then start one of:

```bash
streamlit run app.py              # chat UI at http://localhost:8501
uvicorn api:app --port 8000       # REST API, docs at http://localhost:8000/docs
python agent.py "your question"   # command line
```

Tests: `python -m unittest discover tests`

With Docker, build the index first, then run `docker compose up --build`. This
starts Ollama, the API on port 8000 and the UI on port 8501. The image builds;
the full stack has not been run end to end.

## Limitations

- The test set is small (16 questions), and both the retrieval method and the
  agent's refusal handling were chosen after seeing results on it, so the
  scores are not from a blind test.
- Citations are less precise in answers that combine several papers, and the
  model occasionally cites a passage number that doesn't exist. Both the UI
  and the API flag the latter.
- The number check does not cover numbers written as words.
- Answers can vary slightly between runs.
- It reads PDFs with a text layer. Scanned documents would need OCR first.
- Each question is answered on its own, without memory of earlier questions.
