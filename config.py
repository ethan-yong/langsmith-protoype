"""
Configuration file for RAG Sentiment Analysis System
=====================================================
This file contains all configurable parameters for the system.
Modify these values to customize the behavior without changing the main script.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """
    Configuration class containing all system parameters.
    """
    
    # ========================================================================
    # Directory Paths
    # ========================================================================
    PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./documents")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./faiss_index")
    
    # ========================================================================
    # Model Configuration
    # ========================================================================
    # LLM Model (for answer generation)
    LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    
    # Remote LLM Configuration (for VM/API endpoints)
    USE_REMOTE_LLM = os.getenv("USE_REMOTE_LLM", "false").lower() == "true"
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
    
    # Alternative models you can try:
    # - "meta-llama/Llama-2-7b-chat-hf"
    # - "meta-llama/Llama-2-13b-chat-hf"
    # - "mistralai/Mistral-7B-Instruct-v0.2"
    # - "tiiuae/falcon-7b-instruct"
    
    # Embedding Model (for creating vector embeddings)
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Alternative embedding models:
    # - "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
    # - "sentence-transformers/paraphrase-MiniLM-L6-v2"
    # - "BAAI/bge-small-en-v1.5" (good for English)
    
    # Use 4-bit quantization (reduces memory usage by ~75%)
    USE_4BIT_QUANTIZATION = os.getenv("USE_4BIT_QUANTIZATION", "true").lower() == "true"
    
    # ========================================================================
    # Text Processing Configuration
    # ========================================================================
    # Chunk size for splitting documents
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    
    # Overlap between chunks (maintains context)
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Enable OCR for scanned PDFs
    OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
    
    # ========================================================================
    # Retrieval Configuration
    # ========================================================================
    # Number of document chunks to retrieve for each query
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
    
    # Retrieval search type: "similarity" or "mmr" (maximal marginal relevance)
    RETRIEVAL_TYPE = os.getenv("RETRIEVAL_TYPE", "similarity")
    
    # ========================================================================
    # LLM Generation Configuration
    # ========================================================================
    # Maximum number of tokens to generate
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
    
    # Temperature (0.0 = deterministic, 1.0 = creative)
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Top-p sampling (nucleus sampling)
    TOP_P = float(os.getenv("TOP_P", "0.95"))
    
    # Repetition penalty (higher = less repetition)
    REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
    
    # ========================================================================
    # System Configuration
    # ========================================================================
    # Path to Tesseract executable (Windows only, if not in PATH)
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)
    
    # Path to Poppler bin directory (Windows only, if not in PATH)
    POPPLER_PATH = os.getenv("POPPLER_PATH", None)
    
    # ========================================================================
    # Opik Configuration (Optional - for experiment tracking and tracing)
    # ========================================================================
    OPIK_API_KEY = os.getenv("OPIK_API_KEY", None)
    OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE", "default")
    OPIK_PROJECT = os.getenv("OPIK_PROJECT", "sentiment-analysis-rag")
    
    # ========================================================================
    # Hugging Face Configuration
    # ========================================================================
    HF_TOKEN = os.getenv("HF_TOKEN", None)
    
    # ========================================================================
    # Custom Prompt Template
    # ========================================================================
    QA_PROMPT_TEMPLATE = """You are an expert assistant specialized in sentiment analysis research.
Use the following pieces of context from research papers to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite the source document and page number when providing information.

Context:
{context}

Question: {question}

Helpful Answer:"""
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        Path(cls.PDF_DIRECTORY).mkdir(parents=True, exist_ok=True)
        Path(cls.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and print warnings."""
        issues = []
        
        # Check if PDF directory exists and has files
        if not Path(cls.PDF_DIRECTORY).exists():
            issues.append(f"⚠️  PDF directory not found: {cls.PDF_DIRECTORY}")
        elif not list(Path(cls.PDF_DIRECTORY).glob("*.pdf")):
            issues.append(f"⚠️  No PDF files found in: {cls.PDF_DIRECTORY}")
        
        # Check if HuggingFace token is set for LLaMA models
        if "llama" in cls.LLM_MODEL.lower() and not cls.HF_TOKEN:
            issues.append("⚠️  HF_TOKEN not set. You may need to login: huggingface-cli login")
        
        # Check if Tesseract is configured (Windows)
        if cls.OCR_ENABLED and cls.TESSERACT_CMD:
            if not Path(cls.TESSERACT_CMD).exists():
                issues.append(f"⚠️  Tesseract not found at: {cls.TESSERACT_CMD}")
        
        return issues
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*60)
        print("⚙️  CONFIGURATION")
        print("="*60)
        print(f"PDF Directory: {cls.PDF_DIRECTORY}")
        print(f"Vector Store: {cls.VECTOR_STORE_PATH}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"4-bit Quantization: {cls.USE_4BIT_QUANTIZATION}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"OCR Enabled: {cls.OCR_ENABLED}")
        print(f"Retrieval K: {cls.RETRIEVAL_K}")
        print(f"Max New Tokens: {cls.MAX_NEW_TOKENS}")
        print(f"Temperature: {cls.TEMPERATURE}")
        print(f"LangSmith Tracing: {cls.LANGCHAIN_TRACING_V2}")
        print("="*60)


# Create a singleton instance
config = Config()

# Example usage in main script:
# from config import config
# processor = PDFProcessor(ocr_enabled=config.OCR_ENABLED)
