"""
LangGraph Visualization Script
===============================
Quick script to visualize the LangGraph RAG agent structure.

This script:
1. Loads the vector store
2. Initializes the LLM
3. Builds the LangGraph
4. Outputs a Mermaid diagram

Usage:
    python visualize_graph.py

Output:
    - Prints Mermaid diagram code to terminal
    - Copy to https://mermaid.live/ to view
    - Or view in VS Code with Mermaid extension
"""

import os
import sys

# Fix Windows encoding issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("\n" + "="*70)
    print("LangGraph RAG Agent - Structure Visualization")
    print("="*70)
    
    # Import components
    print("\n[1] Loading components...")
    try:
        from rag_sentiment_analysis import EmbeddingManager, LLaMAQAAgent
        from langgraph_rag_agent import build_rag_graph
        print("   [OK] Imports successful")
    except ImportError as e:
        print(f"   [ERROR] Import failed: {e}")
        print("\n   Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return
    
    # Load vector store
    print("\n[2] Loading vector store...")
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_vector_store("./faiss_index")
    
    if not vectorstore:
        print("   [ERROR] No vector store found at ./faiss_index")
        print("\n   Please run the following first to create the vector store:")
        print("   python run_with_existing_pdfs.py")
        return
    
    print("   [OK] Vector store loaded")
    
    # Initialize LLM
    print("\n[3] Initializing LLM...")
    use_remote = os.getenv("USE_REMOTE_LLM", "false").lower() == "true"
    api_base = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("LLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    
    print(f"   Model: {model_name}")
    print(f"   Remote: {use_remote}")
    print(f"   API Base: {api_base or 'default'}")
    
    try:
        qa_agent = LLaMAQAAgent(
            model_name=model_name,
            use_4bit=True,
            opik_client=None,
            use_remote=use_remote,
            api_base=api_base
        )
        print("   [OK] LLM initialized")
    except Exception as e:
        print(f"   [ERROR] LLM initialization failed: {e}")
        print("\n   This is okay - we can still show the graph structure")
        print("   Creating a minimal graph for visualization...")
        qa_agent = None
    
    # Build graph
    print("\n[4] Building LangGraph...")
    try:
        if qa_agent:
            graph_agent = build_rag_graph(vectorstore, qa_agent.llm)
        else:
            # Create a dummy LLM for structure visualization
            from langchain_openai import ChatOpenAI
            dummy_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key="dummy",
                base_url="http://dummy"
            )
            graph_agent = build_rag_graph(vectorstore, dummy_llm)
        
        graph = graph_agent.build_graph()
        print("   [OK] Graph built successfully")
    except Exception as e:
        print(f"   [ERROR] Graph building failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display Mermaid diagram
    print("\n" + "="*70)
    print("LangGraph Structure (Mermaid Diagram)")
    print("="*70)
    print()
    
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        print(mermaid_code)
    except Exception as e:
        print(f"[ERROR] Failed to generate Mermaid diagram: {e}")
        print("\nFalling back to text description...")
        print_graph_description()
        return
    
    print()
    print("="*70)
    print("How to View This Diagram:")
    print("="*70)
    print()
    print("Option 1: Online Viewer (Easiest)")
    print("   1. Copy the Mermaid code above")
    print("   2. Go to https://mermaid.live/")
    print("   3. Paste the code")
    print("   4. See the interactive diagram!")
    print()
    print("Option 2: VS Code")
    print("   1. Install 'Markdown Preview Mermaid Support' extension")
    print("   2. Create a .md file with the code in a ```mermaid block")
    print("   3. Preview the markdown file")
    print()
    print("Option 3: GitHub/GitLab")
    print("   - Paste in a markdown file")
    print("   - GitHub/GitLab renders Mermaid automatically")
    print()
    print("="*70)
    print("[SUCCESS] Visualization complete!")
    print("="*70)


def print_graph_description():
    """Print a text description of the graph as fallback."""
    print("""
    Graph Structure:
    
    START
      ↓
    [Analyze Query Node]
      ↓
    [Retrieve Node]
      ↓
    [Generate Node]
      ↓
    [Reflect Node]
      ↓
    DECISION (Score >= 0.7 or Refinements >= 2?)
      ↓                          ↓
     YES                        NO
      ↓                          ↓
     END                    [Refine Node]
                                 ↓
                           (Back to Retrieve)
    
    Node Details:
    - Analyze Query: Classifies question type, sets retrieval k
    - Retrieve: Gets documents from vector store
    - Generate: Creates answer using LLM
    - Reflect: Evaluates quality with LangSmith evaluator
    - Refine: Improves query based on feedback
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVisualization cancelled")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
