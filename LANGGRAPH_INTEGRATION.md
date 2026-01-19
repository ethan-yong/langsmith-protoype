# LangGraph RAG Integration Guide

## Overview

The RAG system has been enhanced with LangGraph to provide adaptive, self-improving question answering. Instead of a simple linear chain, the system now uses a stateful graph that can:

- **Analyze question complexity** and adapt retrieval strategy
- **Self-reflect** on answer quality using your existing LangSmith evaluator
- **Automatically refine** questions when quality is low
- **Iterate** until a quality threshold is met (or max attempts reached)

## Architecture

```
User Question
     ↓
[Analyze Query] → Determine complexity (simple/complex/comparative/synthesis)
     ↓
[Retrieve] → Adaptive k (3/6/8 docs based on complexity)
     ↓
[Generate] → Create answer using LLM
     ↓
[Reflect] → Evaluate quality using LangSmith evaluator
     ↓
[Decision] → Score >= 0.7 OR refinements >= 2?
     ↓                              ↓
   YES: Return Answer          NO: Refine Query
                                    ↓
                              [Refine] → Create improved query
                                    ↓
                              Back to [Retrieve] with more docs
```

## Key Components

### 1. State Schema (`RAGState`)

The graph maintains state across all nodes:

```python
{
    "question": str,              # Current question (may be refined)
    "question_type": str,         # simple/complex/comparative/synthesis
    "retrieval_k": int,           # How many docs to retrieve (adaptive)
    "retrieved_docs": List,       # Retrieved documents
    "context": str,               # Concatenated document text
    "answer": str,                # Generated answer
    "reflection_score": float,    # Quality score from evaluator (0-1)
    "reflection_comment": str,    # Detailed feedback from evaluator
    "refinement_count": int,      # How many times refined
    "previous_issues": str,       # What was wrong before
    "chat_history": List          # Conversation history
}
```

### 2. Graph Nodes

#### Analyze Query Node
- Classifies question type using LLM
- Sets retrieval strategy (k value)
- **Simple questions** → k=3 (faster, fewer docs)
- **Complex/synthesis** → k=6 (more context)
- **Comparative** → k=8 (comprehensive)

#### Retrieve Node
- Uses adaptive k from analyze step
- Retrieves documents from FAISS
- Builds context string

#### Generate Node
- Creates answer using main LLM
- Uses comprehensive prompt template
- Handles different LLM types (ChatOpenAI, HuggingFacePipeline)

#### Reflect Node
**This is where your existing LangSmith evaluator is used!**

- Creates a mock run object
- Calls `create_relevance_evaluator()` from `langsmith_integration.py`
- Gets detailed score (0-1) and comment
- Stores both in state

#### Refine Node
**Uses evaluator feedback to improve!**

- Takes the evaluator's specific criticism
- Uses LLM to create a **targeted** refinement
- Increases retrieval k by 2
- Tracks refinement count

### 3. Conditional Logic

The graph decides whether to refine or finish:

```python
if score < 0.7 and refinements < 2:
    # Try again with refined query
    return "refine"
else:
    # Good enough, or tried too many times
    return "end"
```

## Usage

### Basic Usage

```python
from langgraph_rag_agent import build_rag_graph

# Build the graph once
graph_agent = build_rag_graph(vectorstore, llm)

# Run with a question
result = graph_agent.invoke("What are the main challenges in sentiment analysis?")

# Access results
answer = result["answer"]
sources = result["retrieved_docs"]
quality_score = result["reflection_score"]
refinements = result["refinement_count"]
```

### Running the Enhanced System

Simply use the existing runner:

```bash
python run_with_existing_pdfs.py
```

The system automatically uses the LangGraph agent now!

## Example Execution Flow

### Example 1: Simple Question (No Refinement Needed)

**Question:** "What is sentiment analysis?"

```
1. Analyze: Type = "simple", k = 3
2. Retrieve: 3 documents
3. Generate: Creates answer
4. Reflect: Score = 0.85
5. Decision: 0.85 >= 0.7 → Return answer ✓
```

**Result:** Fast response with high quality

### Example 2: Complex Question (Needs Refinement)

**Question:** "What are the main challenges?"

```
1. Analyze: Type = "complex", k = 6
2. Retrieve: 6 documents
3. Generate: Creates vague answer
4. Reflect: Score = 0.55, Comment = "lacks specificity, doesn't mention sarcasm..."
5. Decision: 0.55 < 0.7 → Refine
6. Refine: New question = "What are the main challenges in sentiment analysis, 
            specifically regarding sarcasm, cultural nuances..."
7. Retrieve: 8 documents (k increased)
8. Generate: Creates better answer
9. Reflect: Score = 0.82, Comment = "comprehensive coverage..."
10. Decision: 0.82 >= 0.7 → Return answer ✓
```

**Result:** System self-corrected to provide better answer

## Benefits

### 1. Adaptive Resource Usage
- Simple questions use fewer documents (faster)
- Complex questions get more context (better quality)

### 2. Quality Assurance
- Every answer is evaluated using your sophisticated LLM-as-judge
- Poor answers trigger automatic refinement
- System tries up to 2 times to improve

### 3. Intelligent Refinement
- Not generic "be more comprehensive"
- Uses **specific feedback** from evaluator
- Example: "lacks detail on sarcasm" → "explain sarcasm handling in detail"

### 4. Transparency
- See question type classification
- View quality scores
- Track how many refinements were needed
- Understand why system refined (evaluator feedback)

### 5. Maintains Existing Features
- Still uses your LangSmith evaluator
- Still logs to LangSmith
- Still provides source citations
- Still has conversation memory (can be extended)

## Configuration

### Environment Variables

All existing env vars still work:

```bash
# LLM Configuration
LLM_MODEL=meta-llama/Llama-2-7b-chat-hf
USE_REMOTE_LLM=true
OPENAI_API_BASE=http://localhost:11434/v1

# Judge/Evaluator Model
JUDGE_MODEL=openai/llama3:8b
OPENAI_API_KEY=your-key

# LangSmith
LANGSMITH_API_KEY=your-key
LANGSMITH_TRACING=true
```

### Quality Threshold

To change when refinement happens, edit `langgraph_rag_agent.py`:

```python
def should_refine(self, state: RAGState) -> str:
    score = state.get("reflection_score", 0)
    refinement_count = state.get("refinement_count", 0)
    
    # Change this threshold (default: 0.7)
    if score < 0.7 and refinement_count < 2:
        return "refine"
    return "end"
```

### Max Refinements

To allow more refinement attempts:

```python
# Change refinement_count < 2 to higher number
if score < 0.7 and refinement_count < 3:  # Allow 3 refinements
    return "refine"
```

## Testing

### Quick Component Test

```bash
python test_langgraph_integration.py
```

This will:
1. Check all imports
2. Verify environment variables
3. Test evaluator creation
4. Optionally run a full integration test

### Manual Testing

Best questions to test refinement:
- **Vague:** "What are the challenges?" (should refine)
- **Specific:** "What is sentiment analysis?" (should not refine)
- **Complex:** "Compare linguistic and ML approaches" (may refine)

## Troubleshooting

### Graph takes too long

- Check if LLM is responding slowly
- Try reducing max refinements to 1
- Use a faster judge model

### Quality scores always low

- Check evaluator prompt in `langsmith_integration.py`
- Verify judge model is working
- Try adjusting quality threshold

### Import errors

```bash
pip install langgraph>=0.2.0
```

### Graph doesn't refine when it should

- Check evaluator is returning scores correctly
- Verify threshold (default: 0.7)
- Check `should_refine()` logic

## Extending the Graph

### Add Web Search Node

```python
def web_search_node(self, state: RAGState) -> RAGState:
    """Search web if retrieval insufficient."""
    # Your web search logic
    pass

# Add to graph
workflow.add_node("web_search", self.web_search_node)

# Route to it conditionally
def should_web_search(state):
    if len(state["retrieved_docs"]) < 2:
        return "web_search"
    return "generate"
```

### Add Human Approval Node

```python
def human_approval_node(self, state: RAGState) -> RAGState:
    """Pause for human review."""
    print(f"Answer: {state['answer']}")
    approved = input("Approve? (yes/no): ")
    return {**state, "approved": approved == "yes"}
```

### Add Memory/Conversation Context

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Use with thread_id
result = graph.invoke(
    {"question": question},
    config={"configurable": {"thread_id": "session_123"}}
)
```

## Next Steps

1. **Run the system:** `python run_with_existing_pdfs.py`
2. **Test with various questions** to see refinement in action
3. **Monitor LangSmith** to see detailed traces of graph execution
4. **Adjust thresholds** based on your quality requirements
5. **Extend the graph** with custom nodes for your use case

## Files Modified/Created

- ✅ **NEW:** `langgraph_rag_agent.py` - Graph implementation
- ✅ **NEW:** `test_langgraph_integration.py` - Test suite
- ✅ **NEW:** `LANGGRAPH_INTEGRATION.md` - This guide
- ✅ **MODIFIED:** `run_with_existing_pdfs.py` - Uses graph instead of simple chain
- ✅ **MODIFIED:** `requirements.txt` - Added langgraph dependency

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Evaluation Guide](https://docs.smith.langchain.com/)
- [Your existing evaluator code](langsmith_integration.py)

---

**Questions or issues?** Check the test output or run with verbose logging enabled.
