"""
Central configuration — all settings loaded from .env
Every module imports `settings` and `Paths` from here.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).parent


class Paths:
    DATA_RAW          = ROOT / "data" / "raw"
    DATA_SYNTHETIC    = ROOT / "data" / "synthetic"
    DATA_PROCESSED    = ROOT / "data" / "processed"
    MODELS            = ROOT / "data" / "processed" / "models"
    CHROMA_STORE      = ROOT / "data" / "processed" / "chroma_store"
    PO_PDFS           = ROOT / "data" / "synthetic" / "purchase_orders"
    CATALOG_PDFS      = ROOT / "data" / "synthetic" / "supplier_catalogs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"

    # ChromaDB
    chroma_collection_pos: str = "purchase_orders"
    chroma_collection_catalogs: str = "supplier_catalogs"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "inventory-intelligence"

    # App
    demo_mode: bool = True       # True → rule-based agents (no API key needed)
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def use_llm(self) -> bool:
        return bool(self.openai_api_key) and not self.demo_mode


settings = Settings()
