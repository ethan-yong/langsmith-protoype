# 🚀 Complete Setup Guide

This guide will walk you through setting up the RAG Sentiment Analysis system step-by-step.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Configuration](#advanced-configuration)

---

## System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.9 or higher
- **RAM**: 16GB
- **Disk Space**: 20GB free
- **Internet**: Required for initial model downloads

### Recommended Requirements

- **RAM**: 32GB or more
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060, RTX 4070, A4000)
- **CUDA**: 11.8 or higher
- **SSD**: For faster model loading

---

## Installation Steps

### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ **Check "Add Python to PATH"**
4. Verify: `python --version`

**Mac:**
```bash
# Using Homebrew
brew install python@3.11
python3 --version
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3 --version
```

---

### Step 2: Install System Dependencies

#### Windows

**Tesseract OCR:**
1. Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run installer (use default path: `C:\Program Files\Tesseract-OCR`)
3. Add to PATH:
   - Search "Environment Variables" in Start Menu
   - Edit "Path" variable
   - Add: `C:\Program Files\Tesseract-OCR`
   - Click OK

**Poppler:**
1. Download from [here](https://github.com/oschwartz10612/poppler-windows/releases/)
2. Extract to `C:\poppler\`
3. Add `C:\poppler\bin` to PATH (same process as Tesseract)

**Verify Installation:**
```powershell
tesseract --version
```

### Step 3: Clone/Download Project

```bash
# If you have the files, navigate to the directory
cd path/to/langsmith-prototype

# Or create a new directory
mkdir sentiment-rag
cd sentiment-rag
# Copy all project files here
```

---

### Step 4: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 5: Install Python Packages

```bash
# Upgrade pip first
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

This will take 5-15 minutes depending on your internet speed.

**For GPU Support (NVIDIA only):**
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA 11.8 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 6: Hugging Face Setup

1. **Create Account:**
   - Go to [huggingface.co](https://huggingface.co/)
   - Sign up (free)

2. **Request LLaMA Access:**
   - Visit [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   - Click "Agree and access repository"
   - Wait for approval (usually instant)

3. **Generate Access Token:**
   - Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Create new token (read access is enough)
   - Copy the token

4. **Login via CLI:**
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```
   Paste your token when prompted.

---

### Step 7: Verify Setup

Run the quick start verification script:

```bash
python quick_start.py
```

This will check:
- ✅ Python version
- ✅ All dependencies
- ✅ CUDA availability
- ✅ Tesseract OCR
- ✅ Poppler
- ✅ Hugging Face authentication
- ✅ Embedding model

---

## Configuration

### Basic Configuration

1. **Add Your PDFs:**
   - Copy your sentiment analysis research papers to `./documents/`
   - Supported: Both text-based and scanned PDFs
   - The `documents/` folder should already exist in your project

3. **(Optional) Create .env file:**
   ```bash
   # Copy example
   cp .env.example .env
   
   # Edit with your settings
   # notepad .env  # Windows
   # nano .env     # Mac/Linux
   ```

### Environment Variables (Optional)

Create a `.env` file in the project root:

```env
# LangSmith (optional)
LANGSMITH_API_KEY=your_key_here

# Custom paths (Windows)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\poppler\bin

# Model settings
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
CHUNK_SIZE=1000
RETRIEVAL_K=4
```

---

## Running the System

### First Run

```bash
python rag_sentiment_analysis.py
```

**What happens:**
1. System checks for PDFs in `./documents/`
2. Extracts text (with OCR if needed)
3. Downloads embedding model (~90MB)
4. Creates document chunks
5. Generates embeddings
6. Saves to FAISS vector store
7. Downloads LLaMA model (~16GB, first time only)
8. Starts interactive Q&A

**First run takes 15-30 minutes** due to model downloads.

### Subsequent Runs

Much faster! Uses cached models and vector store:
- Vector store loading: ~5 seconds
- Model loading: ~2 minutes
- Ready to ask questions!

### Example Session

```
🚀 RAG System for Sentiment Analysis Research Papers
============================================================

📁 Found 3 PDF files:
   - sentiment_analysis_survey_2023.pdf
   - deep_learning_sentiment.pdf
   - transformer_nlp.pdf

[...processing steps...]

💬 INTERACTIVE QA MODE
============================================================

👤 Your question: What are the main approaches to sentiment analysis?

🤔 Thinking...

============================================================
📝 ANSWER:
============================================================
The main approaches to sentiment analysis include:

1. Lexicon-based methods using sentiment dictionaries
2. Machine learning approaches (SVM, Naive Bayes)
3. Deep learning methods (CNN, LSTM, Transformers)
4. Transfer learning with pre-trained models like BERT

============================================================
📚 SOURCES:
============================================================

[1] Source: sentiment_analysis_survey_2023.pdf
    Page: 5
    Excerpt: Traditional approaches to sentiment analysis...

[2] Source: deep_learning_sentiment.pdf
    Page: 12
    Excerpt: Modern deep learning architectures...

👤 Your question: Tell me more about BERT's performance

[Continues with context from previous question...]
```

---

## Troubleshooting

### Common Issues

#### 1. "Tesseract not found"

**Error:**
```
pytesseract.pytesseract.TesseractNotFoundError
```

**Solution (Windows):**
Add this to the top of `rag_sentiment_analysis.py`:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Solution (Mac/Linux):**
```bash
brew install tesseract  # Mac
sudo apt install tesseract-ocr  # Linux
```

---

#### 2. "Unable to convert PDF to image"

**Error:**
```
pdf2image.exceptions.PDFInfoNotInstalledError
```

**Solution:**
Install Poppler (see Step 2 above).

**Windows Alternative:**
In code, specify path:
```python
pdf2image.convert_from_path(
    pdf_path,
    poppler_path=r'C:\poppler\bin'
)
```

---

#### 3. "CUDA out of memory"

**Error:**
```
torch.cuda.OutOfMemoryError
```

**Solutions:**

**Option A: Use CPU (slower)**
```python
# In config.py or code
device = "cpu"
```

**Option B: Enable 4-bit quantization**
```python
# Already enabled by default
USE_4BIT_QUANTIZATION = True
```

**Option C: Use smaller model**
```python
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # Smaller
```

**Option D: Reduce batch size**
```python
MAX_NEW_TOKENS = 256  # Instead of 512
```

---

#### 4. "401 Unauthorized" (Hugging Face)

**Error:**
```
requests.exceptions.HTTPError: 401 Client Error
```

**Solution:**
1. Request access to LLaMA: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Wait for approval (usually instant)
3. Login again:
   ```bash
   huggingface-cli login
   ```

---

#### 5. Model Download is Slow

**Issue:** LLaMA download takes too long

**Solutions:**

**Option A: Use mirror (China/Asia)**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Option B: Resume interrupted download**
Downloads resume automatically, just restart the script.

**Option C: Download manually**
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

---

#### 6. "No module named 'bitsandbytes'"

**Error on Windows:**
```
ImportError: No module named 'bitsandbytes'
```

**Solution:**
```bash
# bitsandbytes requires CUDA on Windows
# If no GPU, disable 4-bit quantization:
USE_4BIT_QUANTIZATION = False

# Or use CPU-only version:
pip install bitsandbytes-cpu
```

---

## Advanced Configuration

### Using Different Models

#### Smaller Model (Less Memory)

```python
# In config.py
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
```

Models by size:
- **7B models**: ~14GB VRAM (4-bit) or ~28GB RAM (CPU)
- **13B models**: ~26GB VRAM (4-bit) or ~52GB RAM (CPU)

#### Better Embeddings

```python
# In config.py
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

Trade-off: Better quality but slower.

---

### Using Chroma Instead of FAISS

Replace FAISS with Chroma for persistent storage:

```python
# Install
pip install chromadb

# In code
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

---

### Adding Opik Tracing

For professional debugging, monitoring, and experiment tracking:

1. **Sign up:** https://www.comet.com/opik
2. **Get API key** from workspace settings
3. **Set environment variable:**
   ```bash
   # Windows
   $env:OPIK_API_KEY="your-key"
   
   # Mac/Linux
   export OPIK_API_KEY="your-key"
   ```
4. **View traces:** https://www.comet.com/opik

See **OPIK_INTEGRATION.md** for detailed usage guide.

---

### Optimizing Performance

#### For Speed:
- Use GPU
- Enable 4-bit quantization
- Use smaller models
- Reduce `CHUNK_SIZE`
- Reduce `RETRIEVAL_K`

#### For Quality:
- Use larger models (13B)
- Better embedding models
- Increase `RETRIEVAL_K`
- Larger `CHUNK_SIZE`
- Add reranking step

---

## Next Steps

Once setup is complete:

1. ✅ Add your PDF research papers
2. ✅ Run the system
3. ✅ Ask questions
4. ✅ Customize configuration
5. ✅ Extend with new features

---

## Getting Help

If you encounter issues:

1. **Check this guide** for your specific error
2. **Run diagnostics:** `python quick_start.py`
3. **Check logs** for detailed error messages
4. **Verify PDFs** are readable and not corrupted
5. **Test with a single small PDF** first

---

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | First Load | Subsequent | Query Time |
|----------|------------|------------|------------|
| RTX 4090 + SSD | 2 min | 30 sec | 2-5 sec |
| RTX 3060 + SSD | 5 min | 1 min | 5-10 sec |
| CPU only (32GB) | 10 min | 3 min | 20-40 sec |

---

**🎉 You're ready to start using the system!**

For questions or issues, refer to the main README.md or documentation.
