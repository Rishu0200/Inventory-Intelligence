from __future__ import annotations
from config import settings


def get_llm(temperature: float = 0.3, max_tokens: int = 512):
    """
    Return an instantiated LangChain chat model for the active provider.

    Args:
        temperature: 0.0 = deterministic, 1.0 = creative
        max_tokens:  Max tokens in the response

    Usage:
        from orchestrator.llm_factory import get_llm
        llm = get_llm()
        response = llm.invoke("Your prompt here")
        print(response.content)
    """
    provider = settings.llm_provider.lower()

    # ── Groq (free tier, recommended) ────────────────────────────────────────
    if provider == "groq":
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

    # ── Ollama (fully local, no API key) ─────────────────────────────────────
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("Run: pip install langchain-ollama")
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )

    # ── OpenAI (paid) ────────────────────────────────────────────────────────
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── Google Gemini (free tier) ─────────────────────────────────────────────
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Run: pip install langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{settings.llm_provider}'. "
            f"Choose: groq | ollama | openai | gemini"
        )