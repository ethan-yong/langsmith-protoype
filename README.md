# 🤖 RAG System for Sentiment Analysis Research Papers

A comprehensive Retrieval-Augmented Generation (RAG) system that processes research papers on sentiment analysis and provides an intelligent Q&A interface using local LLaMA models.

## ✨ Features

- 📄 **PDF Processing**: Automatically loads and processes multiple PDF files
- 🔍 **Smart OCR**: Detects scanned PDFs and applies OCR extraction
- 🧩 **Intelligent Chunking**: Splits documents into optimal chunks for embeddings
- 🎯 **Local Embeddings**: Uses Hugging Face models (no OpenAI API needed)
- 🗄️ **FAISS Vector Store**: Fast similarity search with local storage
- 🦙 **LLaMA 3.1 Integration**: Powered by Meta's LLaMA-3.1-8B-Instruct model
- 💬 **Conversation Memory**: Multi-turn conversations with context awareness
- 📚 **Source Citations**: Answers include PDF name and page numbers
- 📊 **Opik Tracing**: Optional observability, logging, and experiment tracking
- 🔧 **Modular Design**: Easy to extend and customize

## 🏗️ Architecture

```
┌─────────────────┐
│   PDF Files     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PDF Processor  │ ◄── OCR Detection
│   (PyPDF2 +     │     (pytesseract)
│   pdf2image)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Chunking   │
│ (LangChain)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │ ◄── HuggingFace
│ (sentence-      │     Embeddings
│  transformers)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Vector   │
│     Store       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │
│   (Top-K)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLaMA 3.1 QA   │ ◄── 4-bit Quantization
│     Agent       │     (memory efficient)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Answer + Sources│
└─────────────────┘
```

## 📋 Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **RAM**: 16GB minimum (32GB recommended for LLaMA)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but highly recommended)
- **Disk Space**: ~20GB for models and dependencies

### External Dependencies

1. **Tesseract OCR** (for scanned PDFs):
   - Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)
   - Mac: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

2. **Poppler** (for PDF to image conversion):
   - Windows: [Download binaries](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Mac: `brew install poppler`
   - Linux: `sudo apt-get install poppler-utils`

## 🚀 Installation

### Step 1: Clone or Download

```bash
# If you have git
git clone <your-repo-url>
cd langsmith-prototype

# Or simply extract the files to your directory
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install System Dependencies

**Windows:**
```powershell
# Download and install Tesseract from:
# https://github.com/UB-Mannheim/tesseract/wiki

# Add Tesseract to PATH or update the script with:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Download Poppler from:
# https://github.com/oschwartz10612/poppler-windows/releases/
# Extract and add to PATH
```

**Mac:**
```bash
brew install tesseract
brew install poppler
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

### Step 5: Hugging Face Setup

1. Create account at [huggingface.co](https://huggingface.co/)
2. Request access to [LLaMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
3. Login via CLI:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```
4. Enter your access token when prompted

### Step 6: (Optional) Opik Setup

For tracing, logging, and experiment tracking:

1. Create account at [comet.com/opik](https://www.comet.com/opik)
2. Get API key from workspace settings
3. Set environment variable:
   ```bash
   # Windows (PowerShell)
   $env:OPIK_API_KEY="your-api-key"
   
   # Mac/Linux
   export OPIK_API_KEY="your-api-key"
   ```
4. View traces at [comet.com/opik](https://www.comet.com/opik)

## 📖 Usage

### Basic Usage

1. **Add your PDF files:**
   - Place your sentiment analysis research papers in `./documents/`
   - The system handles both text-based and scanned PDFs automatically
   - The `documents/` folder should already exist in your project

3. **Run the script:**
   ```bash
   python rag_sentiment_analysis.py
   ```

4. **Ask questions:**
   ```
   👤 Your question: What are the main approaches to sentiment analysis?
   
   👤 Your question: How do neural networks improve sentiment classification?
   
   👤 Your question: What datasets are commonly used for training?
   ```

### Example Session

```
🚀 RAG System for Sentiment Analysis Research Papers
============================================================

📁 Found 3 PDF files:
   - sentiment_analysis_survey_2023.pdf
   - deep_learning_for_sentiment.pdf
   - transformer_models_nlp.pdf

🔧 PDF Processor initialized
============================================================
📄 Extracting text from PDF: sentiment_analysis_survey_2023.pdf
  ✓ Page 1 processed (2847 chars)
  ✓ Page 2 processed (3124 chars)
  ...

✅ Total documents extracted: 45

🤖 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
📝 Chunking 45 documents...
✅ Created 178 chunks from 45 documents

🗄️  Creating FAISS vector store...
✅ Vector store created and saved to ./faiss_index

🦙 Loading LLaMA model: meta-llama/Llama-3.1-8B-Instruct
✅ LLaMA model loaded on cuda

💬 INTERACTIVE QA MODE
============================================================

👤 Your question: What are the latest techniques in sentiment analysis?

🤔 Thinking...

============================================================
📝 ANSWER:
============================================================
Based on the research papers, the latest techniques in sentiment analysis include:

1. **Transformer-based models**: BERT, RoBERTa, and GPT variants have shown 
   significant improvements over traditional approaches, achieving 90%+ accuracy
   on standard benchmarks.

2. **Aspect-based sentiment analysis**: Fine-grained analysis that identifies
   sentiment toward specific aspects or features of products/services.

3. **Multimodal sentiment analysis**: Combining text with images, audio, or
   video for more comprehensive understanding.

4. **Few-shot and zero-shot learning**: Enabling sentiment analysis with
   minimal training data using large language models.

============================================================
📚 SOURCES:
============================================================

[1] Source: sentiment_analysis_survey_2023.pdf
    Page: 12
    Excerpt: Recent advances in transformer architectures have revolutionized
    sentiment analysis tasks. Models like BERT and its variants...

[2] Source: deep_learning_for_sentiment.pdf
    Page: 5
    Excerpt: Aspect-based sentiment analysis (ABSA) has gained significant
    attention in recent years, allowing for more nuanced...

👤 Your question: Can you tell me more about BERT's performance?
[Conversation continues with context from previous question...]
```

## 🔧 Configuration

### Customizing the System

**Change embedding model:**
```python
embedding_manager = EmbeddingManager(
    embedding_model_name="sentence-transformers/all-mpnet-base-v2"
)
```

**Adjust chunk size:**
```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Larger chunks
    chunk_overlap=300,
    # ...
)
```

**Change retrieval settings:**
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}  # Retrieve more chunks
)
```

**Use different LLaMA variant:**
```python
qa_agent = LLaMAQAAgent(
    model_name="meta-llama/Llama-2-13B-chat",  # Larger model
    use_4bit=True
)
```

## 🎯 Adding More PDFs

The system is designed to be modular and extensible:

### Method 1: Add to Directory
```bash
# Simply add new PDFs to the research_papers folder
cp new_paper.pdf ./research_papers/

# Delete existing index to rebuild
rm -rf ./faiss_index

# Run the script again
python rag_sentiment_analysis.py
```

### Method 2: Programmatic Addition
```python
# In the main() function
pdf_paths = [
    "./research_papers/paper1.pdf",
    "./research_papers/paper2.pdf",
    "./new_papers/additional_paper.pdf",  # Add new paths
]
```

### Method 3: Incremental Updates
To add documents to existing vector store:
```python
# Load existing store
vectorstore = embedding_manager.load_vector_store()

# Process new PDFs
new_docs = processor.process_pdf("new_paper.pdf")
new_chunks = embedding_manager.chunk_documents(new_docs)

# Add to existing store
vectorstore.add_documents(new_chunks)
vectorstore.save_local("./faiss_index")
```

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```
Solution: Enable 4-bit quantization and reduce batch size
- Make sure use_4bit=True in LLaMAQAAgent
- Close other applications
- Use smaller model variant
```

**2. Tesseract Not Found**
```
Error: pytesseract.pytesseract.TesseractNotFoundError

Solution (Windows):
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

Solution (Mac/Linux):
brew install tesseract  # or apt-get install tesseract-ocr
```

**3. Poppler Not Found**
```
Error: Unable to convert PDF to image

Solution (Windows):
- Download Poppler from https://github.com/oschwartz10612/poppler-windows/releases/
- Add to PATH or use: pdf2image.convert_from_path(pdf_path, poppler_path=r'C:\poppler\bin')
```

**4. Model Download Fails**
```
Error: 401 Unauthorized

Solution:
- Request access to LLaMA models at https://huggingface.co/meta-llama
- Login: huggingface-cli login
- Wait for access approval (usually instant)
```

**5. CUDA Out of Memory**
```
Solution:
- Reduce max_new_tokens in pipeline
- Use 4-bit quantization
- Process fewer documents at once
- Use CPU instead (slower): device="cpu"
```

## 📊 Performance Tips

### For Faster Processing

1. **Use GPU**: CUDA-enabled GPU significantly speeds up inference
2. **Quantization**: 4-bit quantization reduces memory by ~75%
3. **Smaller models**: Consider using LLaMA-7B instead of 13B
4. **Cache vector store**: Reuse existing FAISS index
5. **Batch processing**: Process multiple PDFs in one session

### For Better Answers

1. **Increase retrieval k**: Retrieve more relevant chunks
2. **Adjust chunk size**: Balance between context and precision
3. **Fine-tune prompts**: Customize QA_PROMPT for your domain
4. **Add metadata filtering**: Filter by specific papers or date ranges
5. **Use reranking**: Add cross-encoder reranking step

## 🔒 Privacy & Security

- ✅ **100% Local**: All processing happens on your machine
- ✅ **No API calls**: No data sent to external services (except optional LangSmith)
- ✅ **Offline capable**: Works without internet (after model download)
- ✅ **Data control**: Your research papers never leave your computer

## 📝 Code Structure

```
.
├── rag_sentiment_analysis.py    # Main script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── research_papers/              # PDF storage (created automatically)
├── faiss_index/                  # Vector store (created automatically)
└── venv/                         # Virtual environment (you create this)
```

## 🤝 Contributing

To extend this system:

1. **Add new PDF sources**: Modify `PDFProcessor` class
2. **Change vector store**: Replace FAISS with Chroma or Pinecone
3. **Add preprocessing**: Extend `extract_text_normal` method
4. **Custom embeddings**: Modify `EmbeddingManager`
5. **Different LLM**: Replace LLaMA with other models

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LLaMA Models](https://huggingface.co/meta-llama)
- [FAISS Guide](https://faiss.ai/)
- [Opik Documentation](https://www.comet.com/docs/opik)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## ⚖️ License

This project is for educational and research purposes. Please ensure you comply with:
- LLaMA's license agreement
- Your institution's research policies
- Copyright laws for research papers

## 🙏 Acknowledgments

- Meta AI for LLaMA models
- LangChain team for the RAG framework
- Hugging Face for model hosting
- The open-source community

---

**Built with ❤️ for researchers by an expert Python engineer**

*Questions? Issues? Feel free to open an issue or contribute!*
