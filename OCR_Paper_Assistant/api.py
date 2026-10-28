"""REST API. Run with: uvicorn api:app --port 8000"""

from contextlib import asynccontextmanager

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import ask_agent
from llm import MODEL
from retrieve import retrieve
from store import EmptyIndexError, get_collection


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["Why is low-light text recognition hard?"])


class Source(BaseModel):
    number: int = Field(description="The [n] used in the answer text.")
    arxiv_id: str
    page: int
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    refused: bool = Field(description="True when the papers do not contain the answer.")
    sources: list[Source]
    steps: list[str] = Field(description="The searches the agent ran, in order.")
    unsupported_numbers: list[str] = Field(
        description="Numbers in the answer that its cited passages never state. Treat the answer with suspicion if non-empty."
    )
    seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start even without an index, so /health can report it.
    try:
        retrieve("optical character recognition", 1)
    except EmptyIndexError as err:
        print(f"WARNING: {err}")
    yield


app = FastAPI(
    title="OCR Paper Assistant",
    description="Answers questions about 40 OCR research papers from arXiv, with citations.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Check that Ollama is reachable, the model is pulled and the index is built."""
    try:
        models = [m.model for m in ollama.list().models]
    except Exception as err:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {err}")
    if MODEL not in models:
        raise HTTPException(status_code=503, detail=f"Model {MODEL} not pulled in Ollama")

    chunks = get_collection().count()
    if chunks == 0:
        raise HTTPException(status_code=503, detail=str(EmptyIndexError()))
    return {"status": "ok", "model": MODEL, "chunks": chunks}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question from the papers, with sources."""
    try:
        result = ask_agent(request.question, verbose=False)
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama unreachable")
    except EmptyIndexError as err:
        raise HTTPException(status_code=503, detail=str(err))

    return AskResponse(
        answer=result["answer"],
        refused=result["refused"],
        sources=[
            Source(
                number=n,
                arxiv_id=s["arxiv_id"],
                page=s["page"],
                title=s["title"],
                url=f"https://arxiv.org/abs/{s['arxiv_id']}",
            )
            for n, s in result["sources"]
        ],
        steps=result["steps"],
        unsupported_numbers=[number for number, _ in result["unsupported_numbers"]],
        seconds=round(result["seconds"], 1),
    )
