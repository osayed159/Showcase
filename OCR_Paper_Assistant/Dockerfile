# Build the index before the image: python download.py && python store.py

FROM python:3.13-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
COPY data/papers.json data/eval_questions.json ./data/
COPY chroma_db ./chroma_db

RUN python -c "from retrieve import retrieve; retrieve('optical character recognition', 1)"

EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
