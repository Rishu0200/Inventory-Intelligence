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
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # ChromaDB
    chroma_collection_pos: str = "purchase_orders"
    chroma_collection_catalogs: str = "supplier_catalogs"
    chroma_cloud_api_key: str = ""
    chroma_cloud_tenant: str = ""
    chroma_cloud_database: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "inventory-intelligence"

    # App
    demo_mode: bool = True       
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000\

    database_url: str = ""     # Neon/Supabase connection string
    redis_url: str = ""        # Upstash REST URL
    redis_token: str = ""      # Upstash REST token
    chroma_cloud_api_key: str = ""
    chroma_cloud_tenant: str = ""
    chroma_cloud_database: str = ""
    r2_access_key: str = ""
    r2_secret_key: str = ""
    r2_bucket: str = ""
    r2_endpoint: str = ""

    @property
    def use_llm(self) -> bool:
        return bool(self.groq_api_key) and not self.demo_mode


settings = Settings()
