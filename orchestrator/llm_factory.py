from __future__ import annotations
from config import settings


def get_llm(temperature: float = 0.3, max_tokens: int = 512):
    """
    Return an instantiated LangChain chat model (Groq / Llama 3.3).

    Usage:
        from orchestrator.llm_factory import get_llm
        llm = get_llm()
        response = llm.invoke("Your prompt here")
        print(response.content)
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )