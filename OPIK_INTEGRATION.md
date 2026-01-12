# 🔍 Opik Integration Guide

This document explains how to use **Opik** for logging, tracing, and evaluating your RAG system's AI requests.

## 📋 Table of Contents

1. [What is Opik?](#what-is-opik)
2. [Setup](#setup)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Viewing Traces](#viewing-traces)
6. [Advanced Features](#advanced-features)

---

## 🎯 What is Opik?

**Opik** is an open-source LLM observability and evaluation platform by Comet ML. It provides:

- 📊 **Trace Logging**: Track every prompt, output, and intermediate step
- ⏱️ **Performance Monitoring**: Duration, token usage, costs
- 🔍 **Debugging**: Inspect model inputs/outputs in detail
- 📈 **Analytics**: Aggregate metrics across all requests
- 🧪 **Evaluation**: Compare different model configurations
- 🆓 **Free Tier**: Generous free usage for development

---

## 🚀 Setup

### Step 1: Install Opik

Already included in `requirements.txt`:

```bash
pip install opik
```

### Step 2: Get Your API Key

1. Go to [https://www.comet.com/opik](https://www.comet.com/opik)
2. Sign up for a free account
3. Navigate to your workspace settings
4. Generate an API key

### Step 3: Configure Environment

Add to your `.env` file:

```env
# Opik Configuration
OPIK_API_KEY=your_opik_api_key_here
OPIK_WORKSPACE=default
OPIK_PROJECT=sentiment-analysis-rag
```

Or set as environment variable:

```powershell
# Windows
$env:OPIK_API_KEY="your_key_here"

# Mac/Linux
export OPIK_API_KEY="your_key_here"
```

---

## ⚙️ Configuration

The system automatically initializes Opik when the API key is present.

### Automatic Initialization

In `rag_sentiment_analysis.py`:

```python
from opik import Opik

def setup_opik_tracing():
    """Configure Opik tracing for observability."""
    api_key = os.getenv("OPIK_API_KEY")
    workspace = os.getenv("OPIK_WORKSPACE", "default")
    
    if api_key:
        opik_client = Opik(
            api_key=api_key,
            workspace=workspace,
            project_name="sentiment-analysis-rag"
        )
        return opik_client
    return None
```

### Manual Initialization

```python
from opik import Opik

# Initialize Opik client
opik_client = Opik(
    api_key="your_api_key",
    workspace="default",
    project_name="my-rag-project"
)
```

---

## 📝 Usage

### Automatic Logging (Built-in)

The system automatically logs all Q&A interactions:

```python
# This happens automatically when you ask questions
qa_agent = LLaMAQAAgent(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    opik_client=opik_client  # Pass the client
)

# Ask a question - automatically logged to Opik
answer, sources = qa_agent.ask_question(qa_chain, "What is sentiment analysis?")
```

**What gets logged:**
- ✅ User question (input)
- ✅ Model answer (output)
- ✅ Retrieved source documents
- ✅ Model name and configuration
- ✅ Execution duration
- ✅ Device (CPU/GPU)
- ✅ Timestamp

---

### Manual Logging Example

For custom logging:

```python
import time
from opik import Opik

opik_client = Opik(api_key="your_key")

# Log a custom trace
start_time = time.time()

# Your AI operation
prompt = "Analyze sentiment: The movie was amazing!"
response = model.generate(prompt)

end_time = time.time()

# Log to Opik
opik_client.log_traces(
    traces=[{
        "name": "sentiment_classification",
        "input": prompt,
        "output": response,
        "metadata": {
            "model": "llama-3.1-8b",
            "duration_seconds": end_time - start_time,
            "classification": "positive",
            "confidence": 0.95
        },
        "tags": ["sentiment", "classification"]
    }]
)
```

---

### Logging with Decorators

Use Opik's `@track` decorator for automatic function tracking:

```python
from opik.decorators import track

@track(
    name="pdf_extraction",
    project_name="sentiment-analysis-rag"
)
def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF - automatically logged."""
    # Your extraction logic
    text = extract(pdf_path)
    return text

# Call normally - automatically logged to Opik
text = extract_pdf_text("paper.pdf")
```

---

## 🔍 Viewing Traces

### Access Your Dashboard

1. Go to [https://www.comet.com/opik](https://www.comet.com/opik)
2. Log in to your account
3. Select your workspace
4. Navigate to the **"sentiment-analysis-rag"** project

### What You'll See

**Traces Dashboard:**
- 📊 List of all logged requests
- ⏱️ Duration and timestamp
- ✅ Success/failure status
- 🔍 Detailed view for each trace

**Individual Trace View:**
```
┌─────────────────────────────────────────────────┐
│ Trace: sentiment_analysis_qa                    │
│ Timestamp: 2026-01-12 10:30:45                  │
│ Duration: 3.24s                                 │
├─────────────────────────────────────────────────┤
│ INPUT:                                          │
│ "What are the main approaches to sentiment      │
│  analysis?"                                     │
├─────────────────────────────────────────────────┤
│ OUTPUT:                                         │
│ "The main approaches include: 1. Lexicon-based  │
│  methods, 2. Machine learning..."              │
├─────────────────────────────────────────────────┤
│ METADATA:                                       │
│ - Model: meta-llama/Llama-3.1-8B-Instruct      │
│ - Sources: 4 documents retrieved                │
│ - Device: cuda                                  │
│ - Tags: rag, sentiment-analysis, llama          │
└─────────────────────────────────────────────────┘
```

---

## 🎓 Advanced Features

### 1. Comparing Different Configurations

Log experiments with different models:

```python
# Experiment 1: LLaMA 8B
opik_client.log_traces(traces=[{
    "name": "qa_llama_8b",
    "input": question,
    "output": answer_8b,
    "metadata": {"model": "llama-3.1-8b", "experiment": "baseline"}
}])

# Experiment 2: LLaMA 13B
opik_client.log_traces(traces=[{
    "name": "qa_llama_13b",
    "input": question,
    "output": answer_13b,
    "metadata": {"model": "llama-3.1-13b", "experiment": "larger_model"}
}])

# Compare in Opik dashboard
```

### 2. Tracking Retrieval Quality

Log retrieval metrics:

```python
opik_client.log_traces(traces=[{
    "name": "document_retrieval",
    "input": query,
    "output": {"retrieved_docs": len(docs)},
    "metadata": {
        "retrieval_k": 4,
        "avg_similarity": 0.87,
        "sources": [doc.metadata for doc in docs]
    }
}])
```

### 3. Error Tracking

Log failures for debugging:

```python
try:
    answer = qa_agent.ask_question(qa_chain, question)
except Exception as e:
    opik_client.log_traces(traces=[{
        "name": "qa_error",
        "input": question,
        "output": str(e),
        "metadata": {
            "error_type": type(e).__name__,
            "status": "failed"
        },
        "tags": ["error"]
    }])
    raise
```

### 4. Custom Metrics

Add custom evaluation metrics:

```python
from opik.evaluation import evaluate

# Define custom evaluator
def relevance_score(output, expected):
    # Your scoring logic
    return {"relevance": 0.9, "accuracy": 0.85}

# Evaluate logged traces
results = evaluate(
    project_name="sentiment-analysis-rag",
    evaluator=relevance_score
)
```

---

## 📊 Example: Complete Integration

```python
from opik import Opik
import time

# Initialize
opik_client = Opik(
    api_key=os.getenv("OPIK_API_KEY"),
    project_name="sentiment-analysis-rag"
)

# Setup RAG system
qa_agent = LLaMAQAAgent(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    opik_client=opik_client
)

# Ask questions - automatically logged
questions = [
    "What are the challenges in sentiment analysis?",
    "How do neural networks help?",
    "What datasets are commonly used?"
]

for question in questions:
    start = time.time()
    
    # Get answer (automatically logged via opik_client)
    answer, sources = qa_agent.ask_question(qa_chain, question)
    
    duration = time.time() - start
    
    print(f"Question: {question}")
    print(f"Answer: {answer[:100]}...")
    print(f"Duration: {duration:.2f}s")
    print(f"✅ Logged to Opik\n")

print("View all traces at: https://www.comet.com/opik")
```

---

## 🔧 Troubleshooting

### Issue: "Opik not available"

**Solution:**
```bash
pip install opik
```

### Issue: "API key not found"

**Solution:**
Check your `.env` file:
```env
OPIK_API_KEY=your_actual_key_here
```

### Issue: "Failed to log to Opik"

**Solution:**
- Verify internet connection
- Check API key is valid
- Ensure workspace exists
- Check Opik service status

### Issue: Traces not appearing

**Solution:**
- Wait 10-30 seconds for processing
- Refresh the dashboard
- Check project name matches
- Verify logs in terminal for errors

---

## 🆚 Opik vs LangSmith

| Feature | Opik | LangSmith |
|---------|------|-----------|
| **Open Source** | ✅ Yes | ❌ No |
| **Free Tier** | Generous | Limited |
| **Self-Hosting** | ✅ Possible | ❌ No |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Integration** | Simple API | LangChain-focused |
| **Analytics** | ✅ Advanced | ✅ Advanced |
| **Community** | Growing | Mature |

---

## 📚 Resources

- **Official Docs**: [https://www.comet.com/docs/opik](https://www.comet.com/docs/opik)
- **GitHub**: [https://github.com/comet-ml/opik](https://github.com/comet-ml/opik)
- **Examples**: [https://www.comet.com/docs/opik/examples](https://www.comet.com/docs/opik/examples)
- **Discord**: Community support channel

---

## ✅ Quick Setup Checklist

- [ ] Install Opik: `pip install opik`
- [ ] Sign up at [https://www.comet.com/opik](https://www.comet.com/opik)
- [ ] Get API key from workspace settings
- [ ] Add `OPIK_API_KEY` to `.env` file
- [ ] Run your RAG system
- [ ] View traces in dashboard

---

**🎉 You're all set!** Your RAG system will now automatically log all requests to Opik for monitoring and analysis.

For questions or issues, refer to the [official documentation](https://www.comet.com/docs/opik) or our main README.md.
