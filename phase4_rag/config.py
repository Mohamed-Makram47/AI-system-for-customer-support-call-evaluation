"""
config.py — Loads environment variables and exposes them as typed constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Groq API
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"

# FAISS / Retrieval
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
TOP_K: int = int(os.getenv("TOP_K", "3"))

# Embedding model (local, no API needed)
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Paths
MANUALS_DIR: str = "manuals/"
INDEXES_DIR: str = "data/indexes/"
MAPS_DIR: str = "data/maps/"
CLASSIFIER_MODEL_DIR: str = "model/"
