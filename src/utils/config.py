"""
Configuration file for Legal Document Analysis PoC
"""
import os
from pathlib import Path

# Databricks Paths
DATABRICKS_WORKSPACE_URL = os.getenv("DATABRICKS_WORKSPACE_URL", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

# Data Paths (DBFS or local)
BASE_DATA_PATH = "/dbfs/mnt/legal-docs" if os.getenv("ENV") == "databricks" else "./data"
RAW_DATA_PATH = f"{BASE_DATA_PATH}/raw"
PROCESSED_DATA_PATH = f"{BASE_DATA_PATH}/processed"
EMBEDDINGS_PATH = f"{BASE_DATA_PATH}/embeddings"

# Delta Table Names
BRONZE_DOCS_TABLE = "bronze_legal_docs"
BRONZE_METADATA_TABLE = "bronze_doc_metadata"
SILVER_TEXTS_TABLE = "silver_legal_texts"
GOLD_EXTRACTIONS_TABLE = "gold_extraction_results"
AUDIT_LOG_TABLE = "audit_log"

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens

# LLM Configuration
LLM_MODEL_1 = "meta-llama/Llama-3-8b-Instruct"  # Primary model
LLM_MODEL_2 = "mistralai/Mistral-7B-Instruct-v0.2"  # Ensemble validation
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1000

# RAG Configuration
TOP_K_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.7

# Ensemble Validation
ENSEMBLE_AGREEMENT_THRESHOLD = 2  # Number of models that must agree

# OCR Configuration
OCR_LANGUAGE = "eng"  # English default
SUPPORTED_LANGUAGES = ["eng", "fra", "deu"]  # English, French, German

# Translation Configuration
TRANSLATION_MODEL_EN_FR = "Helsinki-NLP/opus-mt-en-fr"
TRANSLATION_MODEL_EN_DE = "Helsinki-NLP/opus-mt-en-de"
TRANSLATION_MODEL_FR_EN = "Helsinki-NLP/opus-mt-fr-en"
TRANSLATION_MODEL_DE_EN = "Helsinki-NLP/opus-mt-de-en"

# SEC EDGAR Configuration
SEC_TICKER = os.getenv("SEC_TICKER", "AAPL")  # Default: Apple
SEC_FILING_TYPES = ["10-K", "10-Q", "8-K"]
SEC_FILING_DATE_RANGE = ("2023-01-01", "2024-12-31")

# Processing Configuration
BATCH_SIZE = 100
MAX_DOCUMENT_SIZE_MB = 50

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE_PATH = "./logs/legal_doc_analysis.log"

# Streamlit Configuration
STREAMLIT_PORT = 8501
STREAMLIT_HOST = "0.0.0.0"

# Path to FAISS index
FAISS_INDEX_PATH = f"{BASE_DATA_PATH}/faiss_index/legal_docs.index"

