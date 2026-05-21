import os

# BASE PROJECT PATHS
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# DATA PATH
RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Sanskrit_doc.pdf"
)

FAISS_INDEX_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "faiss_index"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "artifacts",
    "logs"
)

# EMBEDDING MODEL CONFIG
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# LLM CONFIGURATION
LLM_MODEL_NAME = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7

# CHUNKING CONFIGURATION
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

# RETRIEVAL CONFIGURATION
TOP_K_RESULTS = 3

# STREAMLIT CONFIG
APP_TITLE = "Sanskrit RAG System"
APP_ICON = "📚"


# CPU OPTIMIZATION
DEVICE = -1  # CPU


# LOGGING CONFIG
LOG_LEVEL = "INFO"