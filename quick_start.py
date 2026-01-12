"""
Quick Start Script for RAG Sentiment Analysis System
=====================================================
This is a simplified version for quick testing and demonstration.
Use this to verify your setup before using the full system.
"""

import os
from pathlib import Path

print("🚀 Quick Start - RAG System Setup Verification")
print("="*60)

# ============================================================================
# Step 1: Check Python Version
# ============================================================================
import sys
print(f"\n✓ Python Version: {sys.version}")
if sys.version_info < (3, 9):
    print("⚠️  Warning: Python 3.9+ recommended")

# ============================================================================
# Step 2: Check Core Dependencies
# ============================================================================
print("\n📦 Checking Dependencies...")

dependencies = {
    "torch": "PyTorch",
    "transformers": "Hugging Face Transformers",
    "langchain": "LangChain",
    "faiss": "FAISS (CPU)",
    "PyPDF2": "PDF Reader",
    "pytesseract": "Tesseract OCR",
    "pdf2image": "PDF to Image",
    "sentence_transformers": "Sentence Transformers",
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} - MISSING")
        missing.append(module)

if missing:
    print(f"\n❌ Missing dependencies: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# ============================================================================
# Step 3: Check CUDA Availability
# ============================================================================
import torch
print(f"\n🔧 Hardware Configuration:")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("  ⚠️  Running on CPU (will be slower)")

# ============================================================================
# Step 4: Check External Tools
# ============================================================================
print("\n🔨 Checking External Tools...")

# Check Tesseract
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f"  ✓ Tesseract OCR v{version}")
except Exception as e:
    print(f"  ✗ Tesseract OCR - NOT FOUND")
    print(f"    Install from: https://github.com/UB-Mannheim/tesseract/wiki")

# Check Poppler (indirectly via pdf2image)
try:
    from pdf2image import pdfinfo_from_path
    print(f"  ✓ Poppler (pdf2image)")
except Exception as e:
    print(f"  ⚠️  Poppler might not be installed correctly")
    print(f"    Install from: https://github.com/oschwartz10612/poppler-windows/releases/")

# ============================================================================
# Step 5: Check Directory Structure
# ============================================================================
print("\n📁 Checking Directory Structure...")

PDF_DIR = "./documents"
VECTOR_DIR = "./faiss_index"

# Create directories if they don't exist
Path(PDF_DIR).mkdir(exist_ok=True)
Path(VECTOR_DIR).mkdir(exist_ok=True)

print(f"  ✓ PDF Directory: {PDF_DIR}")
pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
print(f"    Found {len(pdf_files)} PDF files")

if pdf_files:
    for pdf in pdf_files:
        print(f"      - {pdf.name}")
else:
    print("    ⚠️  No PDF files found. Add PDFs to ./documents/")

# ============================================================================
# Step 6: Test Embedding Model (Small Test)
# ============================================================================
print("\n🧪 Testing Embedding Model...")
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # Test embedding
    test_text = "This is a test sentence for sentiment analysis."
    embedding = embeddings.embed_query(test_text)
    
    print(f"  ✓ Embedding Model Working")
    print(f"    Embedding dimension: {len(embedding)}")
    
except Exception as e:
    print(f"  ✗ Embedding Model Error: {e}")

# ============================================================================
# Step 7: Check Hugging Face Authentication
# ============================================================================
print("\n🤗 Checking Hugging Face Authentication...")
try:
    from huggingface_hub import HfApi
    api = HfApi()
    
    # Try to get user info
    try:
        user = api.whoami()
        print(f"  ✓ Logged in as: {user['name']}")
    except Exception:
        print(f"  ⚠️  Not logged in to Hugging Face")
        print(f"    Run: huggingface-cli login")
        print(f"    (Required for LLaMA models)")
        
except ImportError:
    print(f"  ⚠️  huggingface-hub not installed")
    print(f"    Run: pip install huggingface-hub")

# ============================================================================
# Step 8: Check Opik (Optional)
# ============================================================================
print("\n📊 Checking Opik (Optional)...")
try:
    from opik import Opik
    api_key = os.getenv("OPIK_API_KEY")
    
    if api_key:
        print(f"  ✓ Opik API key found")
        print(f"    View traces at: https://www.comet.com/opik")
    else:
        print(f"  ⚠️  Opik API key not set (optional)")
        print(f"    Set OPIK_API_KEY to enable tracing")
        print(f"    Get your key from: https://www.comet.com/opik")
        
except ImportError:
    print(f"  ⚠️  Opik not installed (optional)")
    print(f"    Run: pip install opik")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("📋 SETUP SUMMARY")
print("="*60)

can_run = True
issues = []

if missing:
    can_run = False
    issues.append("Missing Python packages")

if not torch.cuda.is_available():
    issues.append("No CUDA GPU (will be slow)")

if not pdf_files:
    issues.append("No PDF files to process")

if can_run:
    print("✅ System is ready to run!")
    print("\nNext steps:")
    print("  1. Add PDF files to ./research_papers/")
    print("  2. Run: python rag_sentiment_analysis.py")
    print("  3. Ask questions about your research papers!")
else:
    print("⚠️  Setup incomplete:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nPlease fix the issues above before running the main script.")

print("="*60)

# ============================================================================
# Mini Test (If PDFs exist)
# ============================================================================
if pdf_files and not missing:
    print("\n🧪 Would you like to run a quick PDF extraction test? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        print("\n" + "="*60)
        print("📄 Testing PDF Extraction...")
        print("="*60)
        
        from PyPDF2 import PdfReader
        
        test_pdf = pdf_files[0]
        print(f"\nTesting with: {test_pdf.name}")
        
        try:
            reader = PdfReader(str(test_pdf))
            num_pages = len(reader.pages)
            print(f"  ✓ Number of pages: {num_pages}")
            
            # Extract first page
            first_page_text = reader.pages[0].extract_text()
            print(f"  ✓ First page text length: {len(first_page_text)} characters")
            
            if len(first_page_text) < 50:
                print(f"  ⚠️  Very little text extracted - might be a scanned PDF")
                print(f"      OCR will be used automatically in the main script")
            else:
                print(f"  ✓ Text extraction working well")
                print(f"\n  Preview (first 200 chars):")
                print(f"  {first_page_text[:200]}...")
            
            print("\n✅ PDF extraction test completed successfully!")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")

print("\n👋 Quick start check complete!")
