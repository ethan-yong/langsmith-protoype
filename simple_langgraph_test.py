"""
Simple LangGraph Test
=====================
Minimal example demonstrating LangGraph concepts:
- Custom state management
- Multiple nodes (retrieve, reflect, refine)
- Conditional routing (refinement loop)
- Visual execution trace

This is a simplified version of the full RAG graph to understand the basics.

Usage:
    python simple_langgraph_test.py
"""

import sys
import io

# Fix Windows encoding issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from typing import TypedDict
from langgraph.graph import StateGraph, END


# ============================================================================
# 1. Define State Schema
# ============================================================================

class SimpleRAGState(TypedDict):
    """State that flows through the graph."""
    question: str
    answer: str
    score: float
    attempts: int
    feedback: str


# ============================================================================
# 2. Define Node Functions
# ============================================================================

def retrieve_node(state: SimpleRAGState) -> SimpleRAGState:
    """Simulate retrieving documents and generating an answer."""
    attempt = state.get('attempts', 0)
    question = state['question']
    
    print(f"\n📚 [RETRIEVE] Attempt #{attempt + 1}")
    print(f"   Question: {question}")
    
    # Simulate different quality answers based on question detail
    if "detailed" in question.lower() or "specific" in question.lower():
        answer = """Sentiment analysis is a natural language processing technique used to 
determine the emotional tone behind text. It analyzes opinions, sentiments, and emotions 
expressed in text data. Key applications include social media monitoring, customer feedback 
analysis, and brand reputation management. Common challenges include handling sarcasm, 
context-dependent meanings, and cultural nuances."""
    else:
        answer = "Sentiment analysis is a technique to determine emotions in text."
    
    print(f"   Generated answer: {len(answer)} chars")
    
    return {
        **state,
        "answer": answer,
        "attempts": attempt + 1
    }


def reflect_node(state: SimpleRAGState) -> SimpleRAGState:
    """Evaluate answer quality (simulates LLM-as-judge evaluator)."""
    answer = state['answer']
    question = state['question']
    
    print(f"\n🤔 [REFLECT] Evaluating answer quality...")
    
    # Simulate scoring based on answer length and detail
    score = 0.0
    feedback = ""
    
    if len(answer) < 100:
        score = 0.4
        feedback = "Answer is too brief. Lacks detail and specific examples."
    elif len(answer) < 200:
        score = 0.6
        feedback = "Answer provides basic information but lacks depth on key challenges."
    else:
        score = 0.85
        feedback = "Answer is comprehensive and covers key aspects well."
    
    print(f"   Score: {score:.2f}/1.0")
    print(f"   Feedback: {feedback}")
    
    return {
        **state,
        "score": score,
        "feedback": feedback
    }


def refine_node(state: SimpleRAGState) -> SimpleRAGState:
    """Refine the question based on evaluator feedback."""
    original_question = state['question']
    feedback = state.get('feedback', '')
    score = state.get('score', 0)
    
    print(f"\n🔄 [REFINE] Score {score:.2f} below threshold (0.7)")
    print(f"   Feedback: {feedback}")
    print(f"   Creating more specific question...")
    
    # Refine question based on what was missing
    if "brief" in feedback.lower() or "lacks detail" in feedback.lower():
        refined_question = f"{original_question} Provide detailed explanation with specific examples."
    elif "depth" in feedback.lower():
        refined_question = f"{original_question} Include key challenges and applications in detail."
    else:
        refined_question = f"{original_question} (more comprehensive)"
    
    print(f"   Refined: {refined_question}")
    
    return {
        **state,
        "question": refined_question
    }


# ============================================================================
# 3. Define Routing Logic
# ============================================================================

def should_refine(state: SimpleRAGState) -> str:
    """Decide whether to refine or finish."""
    score = state.get('score', 0)
    attempts = state.get('attempts', 0)
    
    # Refine if score is low and we haven't tried too many times
    if score < 0.7 and attempts < 3:
        return "refine"
    else:
        if attempts >= 3:
            print(f"\n⚠️  Max attempts reached ({attempts}), returning best answer")
        else:
            print(f"\n✅ Score {score:.2f} meets threshold (0.7)")
        return "end"


# ============================================================================
# 4. Build the Graph
# ============================================================================

def build_simple_graph() -> StateGraph:
    """Build and compile the LangGraph."""
    workflow = StateGraph(SimpleRAGState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("refine", refine_node)
    
    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "reflect")
    
    # Conditional edge: refine or finish?
    workflow.add_conditional_edges(
        "reflect",
        should_refine,
        {
            "refine": "refine",
            "end": END
        }
    )
    
    # After refining, go back to retrieve
    workflow.add_edge("refine", "retrieve")
    
    return workflow.compile()


# ============================================================================
# 5. Test Function
# ============================================================================

def test_simple_question():
    """Test with a simple question that should trigger refinement."""
    print("\n" + "="*70)
    print("Test 1: Simple Question (Should Trigger Refinement)")
    print("="*70)
    
    graph = build_simple_graph()
    
    result = graph.invoke({
        "question": "What is sentiment analysis?",
        "answer": "",
        "score": 0.0,
        "attempts": 0,
        "feedback": ""
    })
    
    print("\n" + "="*70)
    print("📊 Final Result")
    print("="*70)
    print(f"Question (final): {result['question']}")
    print(f"Answer: {result['answer'][:150]}...")
    print(f"Score: {result['score']:.2f}/1.0")
    print(f"Attempts: {result['attempts']}")
    print(f"Feedback: {result['feedback']}")
    
    return result


def test_detailed_question():
    """Test with a detailed question that should pass first try."""
    print("\n" + "="*70)
    print("Test 2: Detailed Question (Should Pass First Try)")
    print("="*70)
    
    graph = build_simple_graph()
    
    result = graph.invoke({
        "question": "What is sentiment analysis? Explain in detail with applications.",
        "answer": "",
        "score": 0.0,
        "attempts": 0,
        "feedback": ""
    })
    
    print("\n" + "="*70)
    print("📊 Final Result")
    print("="*70)
    print(f"Question (final): {result['question']}")
    print(f"Answer: {result['answer'][:150]}...")
    print(f"Score: {result['score']:.2f}/1.0")
    print(f"Attempts: {result['attempts']}")
    print(f"Feedback: {result['feedback']}")
    
    return result


def visualize_graph():
    """Display the graph structure as Mermaid."""
    print("\n" + "="*70)
    print("Graph Structure (Mermaid Diagram)")
    print("="*70)
    
    graph = build_simple_graph()
    
    try:
        mermaid = graph.get_graph().draw_mermaid()
        print(mermaid)
        print("\n💡 Copy to https://mermaid.live/ to visualize")
    except Exception as e:
        print(f"Could not generate Mermaid: {e}")
        print("\nText representation:")
        print("""
        START
          ↓
        [retrieve] - Get documents, generate answer
          ↓
        [reflect] - Evaluate quality, assign score
          ↓
        DECISION (score >= 0.7 or attempts >= 3?)
          ↓                          ↓
         YES                        NO
          ↓                          ↓
         END                    [refine] - Improve question
                                     ↓
                               (loop back to retrieve)
        """)


# ============================================================================
# 6. Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🧪 Simple LangGraph Test Suite")
    print("="*70)
    print("\nThis demonstrates core LangGraph concepts:")
    print("  • Custom state (TypedDict)")
    print("  • Multiple nodes (retrieve, reflect, refine)")
    print("  • Conditional routing (quality-based)")
    print("  • Refinement loops (up to 3 attempts)")
    
    # Show graph structure
    visualize_graph()
    
    # Run tests automatically
    print("\n" + "="*70)
    result1 = test_simple_question()
    
    print("\n" + "="*70)
    result2 = test_detailed_question()
    
    # Summary
    print("\n" + "="*70)
    print("📈 Summary")
    print("="*70)
    print(f"Test 1 (simple):   {result1['attempts']} attempts, final score: {result1['score']:.2f}")
    print(f"Test 2 (detailed): {result2['attempts']} attempts, final score: {result2['score']:.2f}")
    print("\nKey takeaway:")
    print("  - Vague questions trigger refinement loops")
    print("  - Detailed questions pass on first attempt")
    print("  - The graph adapts based on quality feedback")
    print("\n✅ Tests complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest cancelled")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
