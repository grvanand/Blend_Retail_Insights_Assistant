# config/settings.py
# Centralized configuration using Pydantic Settings.
# All modules import from here — no direct os.getenv() calls elsewhere.
 
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
 
# Resolve .env from project root regardless of where the app is launched from
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
 
 
class Settings(BaseSettings):
    """
    Loads and validates all environment variables from .env file.
    Fail-fast: missing required keys raise ValidationError at startup.
    """
 
    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",          # silently ignore unknown env vars
    )
 
    # --- LangSmith Observability ---
    langchain_tracing_v2: bool = Field(default=True, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(..., alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="BlendDemo", alias="LANGCHAIN_PROJECT")
 
    # --- HuggingFace ---
    hf_token: str = Field(..., alias="HF_TOKEN")
 
    # --- Groq LLM ---
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model_name: str = Field(default="llama3-70b-8192", alias="GROQ_MODEL_NAME")
 
    # --- App Settings ---
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    data_dir: str = Field(default="data/raw", alias="DATA_DIR")
    faiss_index_path: str = Field(default="vectorstore/faiss_index", alias="FAISS_INDEX_PATH")
 
    # --- LLM Inference Settings ---
    llm_temperature: float = 0.0        # deterministic for data Q&A
    llm_max_tokens: int = 1024
    llm_max_retries: int = 3            # used by retry decorator in utils/retry.py
 
    # --- Embedding Settings ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
 
    # --- FAISS Settings ---
    faiss_top_k: int = 5                # number of chunks retrieved per query
 
 
# Singleton instance — import this across all modules
settings = Settings()