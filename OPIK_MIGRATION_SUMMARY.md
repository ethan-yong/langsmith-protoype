# 🔄 Opik Migration Summary

This document summarizes the changes made to migrate from LangSmith to Opik for logging and tracing.

## 📋 What Changed

### Files Modified

1. **`rag_sentiment_analysis.py`** - Main RAG script
   - ✅ Replaced LangSmith imports with Opik
   - ✅ Updated `setup_langsmith_tracing()` → `setup_opik_tracing()`
   - ✅ Added Opik client to `LLaMAQAAgent.__init__()`
   - ✅ Integrated automatic logging in `ask_question()` method

2. **`run_with_existing_pdfs.py`** - Quick runner script
   - ✅ Updated imports
   - ✅ Pass Opik client to QA agent

3. **`requirements.txt`** - Dependencies
   - ✅ Replaced `langsmith` with `opik`

4. **`config.py`** - Configuration
   - ✅ Replaced LangSmith config with Opik config
   - ✅ Updated environment variable names

5. **`README.md`** & **`SETUP_GUIDE.md`** - Documentation
   - ✅ Updated references from LangSmith to Opik
   - ✅ Added Opik setup instructions

### Files Added

1. **`OPIK_INTEGRATION.md`** - Complete Opik integration guide
2. **`opik_example.py`** - Practical examples and usage patterns
3. **`OPIK_MIGRATION_SUMMARY.md`** - This file

---

## 🔑 Key Differences: LangSmith → Opik

### Initialization

**Before (LangSmith):**
```python
from langsmith import Client

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-project"
```

**After (Opik):**
```python
from opik import Opik

# Initialize client
opik_client = Opik(
    api_key=os.getenv("OPIK_API_KEY"),
    workspace="default",
    project_name="sentiment-analysis-rag"
)
```

### Logging Traces

**Before (LangSmith):**
```python
# LangSmith uses automatic tracing via LangChain integration
# No explicit logging code needed, but less control
```

**After (Opik):**
```python
# Explicit logging with full control
opik_client.log_traces(
    traces=[{
        "name": "qa_request",
        "input": question,
        "output": answer,
        "metadata": {
            "model": "llama-3.1-8b",
            "duration": 2.5,
            "sources": sources
        },
        "tags": ["rag", "qa"]
    }]
)
```

### Environment Variables

**Before (LangSmith):**
```env
LANGSMITH_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-project
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

**After (Opik):**
```env
OPIK_API_KEY=your_key
OPIK_WORKSPACE=default
OPIK_PROJECT=sentiment-analysis-rag
```

---

## 🚀 Quick Start with Opik

### 1. Install Opik

```bash
pip install opik
```

### 2. Get API Key

1. Go to https://www.comet.com/opik
2. Sign up for free
3. Generate API key from workspace settings

### 3. Configure

Add to `.env`:
```env
OPIK_API_KEY=your_api_key_here
OPIK_WORKSPACE=default
```

### 4. Run System

```bash
python run_with_existing_pdfs.py
```

That's it! All Q&A interactions are automatically logged to Opik.

---

## 📊 What Gets Logged

Every time you ask a question, Opik automatically logs:

```json
{
  "name": "sentiment_analysis_qa",
  "input": "What are the main approaches to sentiment analysis?",
  "output": "The main approaches include: 1. Lexicon-based...",
  "metadata": {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "num_sources": 4,
    "sources": [
      {
        "source": "sentiment_survey.pdf",
        "page": 3,
        "content_preview": "Traditional approaches..."
      }
    ],
    "duration_seconds": 3.24,
    "device": "cuda"
  },
  "tags": ["rag", "sentiment-analysis", "llama"]
}
```

---

## 🔍 Viewing Traces

1. Go to https://www.comet.com/opik
2. Login to your account
3. Select your workspace
4. Navigate to "sentiment-analysis-rag" project
5. Browse all logged traces with full details

---

## 💡 Benefits of Opik

### vs LangSmith

| Feature | Opik | LangSmith |
|---------|------|-----------|
| **Cost** | Free tier generous | Limited free |
| **Open Source** | ✅ Yes | ❌ No |
| **Self-Hosting** | ✅ Possible | ❌ No |
| **Explicit Control** | ✅ Full | Automatic only |
| **UI/UX** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Integration** | Simple API | LangChain-focused |

### Additional Benefits

- 📊 **Better Analytics**: Advanced dashboards and metrics
- 🔍 **Detailed Tracing**: See every step of your RAG pipeline
- 🧪 **Experiment Tracking**: Compare different configurations
- 📈 **Performance Monitoring**: Track latency and costs
- 🆓 **Generous Free Tier**: Perfect for development
- 🌐 **Community Support**: Active Discord community

---

## 📚 Usage Examples

### Example 1: Basic Logging (Automatic)

```python
# Just use the system normally - logging happens automatically!
qa_agent = LLaMAQAAgent(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    opik_client=opik_client  # Pass the client
)

# Ask questions - automatically logged
answer, sources = qa_agent.ask_question(qa_chain, "Your question here")
```

### Example 2: Custom Logging

```python
from opik import Opik
import time

opik_client = Opik(api_key="your_key")

start = time.time()
# Your processing
result = process_data(input_data)
duration = time.time() - start

# Log custom trace
opik_client.log_traces(
    traces=[{
        "name": "custom_processing",
        "input": input_data,
        "output": result,
        "metadata": {
            "duration": duration,
            "status": "success"
        }
    }]
)
```

### Example 3: Using Decorators

```python
from opik.decorators import track

@track(name="pdf_extraction", project_name="sentiment-analysis-rag")
def extract_pdf(pdf_path: str) -> str:
    # Your extraction logic
    text = extract(pdf_path)
    return text

# Automatically logged when called
text = extract_pdf("paper.pdf")
```

---

## 🔧 Configuration Options

### Opik Client Options

```python
opik_client = Opik(
    api_key="your_key",           # Required
    workspace="default",           # Optional
    project_name="my-project",     # Optional
    api_url="https://...",         # Optional (for self-hosted)
    timeout=30                     # Optional (request timeout)
)
```

### Trace Options

```python
opik_client.log_traces(
    traces=[{
        "name": "trace_name",          # Required
        "input": "input_data",         # Required
        "output": "output_data",       # Required
        "metadata": {                  # Optional
            "model": "llama-3.1",
            "custom_key": "custom_value"
        },
        "tags": ["tag1", "tag2"],      # Optional
        "start_time": timestamp,       # Optional
        "end_time": timestamp          # Optional
    }]
)
```

---

## 🎯 Next Steps

1. ✅ **Install Opik**: `pip install opik`
2. ✅ **Get API Key**: https://www.comet.com/opik
3. ✅ **Update .env**: Add `OPIK_API_KEY`
4. ✅ **Run System**: All logging automatic!
5. ✅ **View Dashboard**: https://www.comet.com/opik

### Learn More

- Read **OPIK_INTEGRATION.md** for detailed guide
- Run **opik_example.py** for hands-on examples
- Check [Opik Docs](https://www.comet.com/docs/opik) for advanced features

---

## ❓ FAQ

### Q: Do I need to change my existing code?

**A:** No! If you were using the system without LangSmith, nothing changes. Just add your Opik API key to enable logging.

### Q: Can I use both Opik and LangSmith?

**A:** Technically yes, but not recommended. Opik is more flexible and easier to use.

### Q: Is Opik free?

**A:** Yes! Generous free tier perfect for development and small projects.

### Q: Can I self-host Opik?

**A:** Yes! Opik is open-source and can be self-hosted.

### Q: What if I don't want logging?

**A:** Simply don't set `OPIK_API_KEY`. The system works perfectly without it.

---

## 📞 Support

- **Documentation**: [comet.com/docs/opik](https://www.comet.com/docs/opik)
- **GitHub**: [github.com/comet-ml/opik](https://github.com/comet-ml/opik)
- **Discord**: Join the community
- **Issues**: GitHub Issues

---

**🎉 Migration Complete!** Your RAG system now uses Opik for superior logging and observability.
