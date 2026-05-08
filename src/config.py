import os
from langchain_openai import ChatOpenAI


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
    """LLM instance'ı döndürür. OPENAI_API_KEY ortam değişkeni gerekli."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY ortam değişkeni ayarlanmamış! "
            "Lütfen .env dosyası oluşturun veya export OPENAI_API_KEY=sk-... şeklinde ayarlayın."
        )
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
