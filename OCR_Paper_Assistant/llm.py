"""The local LLM used for answering."""

# Instruct variant: the base qwen3:4b's reasoning can't be disabled on Ollama 0.40.11.
MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()
