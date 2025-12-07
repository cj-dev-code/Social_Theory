"""
Central configuration for the Feminism RAG Discord Bot.

Principles:
- Secrets and environment-specific paths are loaded from `.env` at the project root.
- Algorithmic / modeling choices live in code so they are version-tracked and reviewable.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# --- Locate project root ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Load .env ---
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# EXTERNAL CREDENTIALS + INFRA (must come from environment)
# ============================================================

# Cohere (used for reranking retrieved docs)
COHERE_API_KEY: str | None = os.getenv("COHERE_API_KEY")
COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

# Discord bot token (for running the bot)
DISCORD_BOT_TOKEN: str | None = os.getenv("DISCORD_BOT_TOKEN")

# Local Ollama endpoint (where your Qwen / feminism model is served)
OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_0")#"qwen2.5:7b-instruct")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Qdrant vector DB (location + collection name for your experiment-9 index)
QDRANT_PATH: str = os.getenv("QDRANT_PATH", str(PROJECT_ROOT / "qdrant_index"))
QDRANT_COLLECTION: str = os.getenv(
    "QDRANT_COLLECTION",
    "experiment_9_1024_128_bge-large-en-v1.5_3_docs_take_2",
)


HF_CACHE: str | None = os.getenv("HF_CACHE")



# ============================================================
# ALGORITHMIC / MODELING CHOICES (kept in code, not in .env)
# ============================================================

# Embeddings: design choice, must match your built index
EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
QDRANT_VECTOR_NAME: str | None = "fast-bge-large-en-v1.5"

# Local LLM hyperparameters — deliberate design choices
OLLAMA_TEMPERATURE: float = 0.2
OLLAMA_NUM_CTX: int = 8192

# Retrieval parameters — these should be reviewable in PRs
RETRIEVAL_MMR_K: int = 15
RERANK_FINAL_K: int = 5
