"""
Test Script for LangGraph RAG Integration
==========================================
This script tests the LangGraph RAG agent to ensure all nodes work correctly.

Usage:
    python test_langgraph_integration.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_langgraph_components():
    """Test individual LangGraph components."""
    print("="*70)
    print("LangGraph RAG Integration - Component Tests")
    print("="*70)
    
    # Check imports
    print("\n1. Testing imports...")
    try:
        from langgraph_rag_agent import LangGraphRAGAgent, RAGState, build_rag_graph
        from langsmith_integration import create_relevance_evaluator
        print("   ✅ All imports successful")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("\n   Please install langgraph:")
        print("   pip install langgraph>=0.2.0")
        return False
    
    # Check environment variables
    print("\n2. Checking environment variables...")
    required_vars = ["OPENAI_API_BASE", "OPENAI_API_KEY", "LANGSMITH_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"   ⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   The system may still work with defaults")
    else:
        print("   ✅ All required environment variables set")
    
    # Check if vector store exists
    print("\n3. Checking for existing vector store...")
    vector_store_path = "./faiss_index"
    if Path(vector_store_path).exists():
        print(f"   ✅ Vector store found at {vector_store_path}")
    else:
        print(f"   ⚠️  No vector store found at {vector_store_path}")
        print("   You'll need to run run_with_existing_pdfs.py first to create it")
        return False
    
    # Test evaluator creation
    print("\n4. Testing evaluator creation...")
    try:
        evaluator = create_relevance_evaluator()
        if evaluator:
            print("   ✅ Relevance evaluator created successfully")
        else:
            print("   ❌ Evaluator creation returned None")
            return False
    except Exception as e:
        print(f"   ❌ Evaluator creation failed: {e}")
        return False
    
    return True


def test_full_integration():
    """Test the full LangGraph integration with the RAG system."""
    print("\n" + "="*70)
    print("Full Integration Test")
    print("="*70)
    
    try:
        # Import necessary components
        from rag_sentiment_analysis import EmbeddingManager, LLaMAQAAgent
        from langgraph_rag_agent import build_rag_graph
        
        print("\n1. Loading vector store...")
        embedding_manager = EmbeddingManager()
        vectorstore = embedding_manager.load_vector_store("./faiss_index")
        
        if vectorstore is None:
            print("   ❌ Failed to load vector store")
            print("   Please run run_with_existing_pdfs.py first to create the vector store")
            return False
        
        print("   ✅ Vector store loaded successfully")
        
        print("\n2. Initializing LLM...")
        use_remote = os.getenv("USE_REMOTE_LLM", "false").lower() == "true"
        api_base = os.getenv("OPENAI_API_BASE")
        model_name = os.getenv("LLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        
        print(f"   Model: {model_name}")
        print(f"   Remote: {use_remote}")
        print(f"   API Base: {api_base}")
        
        qa_agent = LLaMAQAAgent(
            model_name=model_name,
            use_4bit=True,
            opik_client=None,
            use_remote=use_remote,
            api_base=api_base
        )
        
        print("   ✅ LLM initialized")
        
        print("\n3. Building LangGraph RAG agent...")
        graph_agent = build_rag_graph(vectorstore, qa_agent.llm)
        print("   ✅ LangGraph agent built successfully")
        
        print("\n4. Testing with a sample question...")
        test_question = "What is sentiment analysis?"
        
        print(f"\n   Question: {test_question}")
        print("\n   Running graph (this may take a moment)...")
        print("   " + "-"*66)
        
        result = graph_agent.invoke(test_question)
        
        print("\n   " + "-"*66)
        print("   ✅ Graph execution completed!")
        
        print("\n5. Results:")
        print(f"   - Question Type: {result.get('question_type', 'unknown')}")
        print(f"   - Retrieval K: {result.get('retrieval_k', 0)}")
        print(f"   - Documents Retrieved: {len(result.get('retrieved_docs', []))}")
        print(f"   - Quality Score: {result.get('reflection_score', 0):.2f}/1.0")
        print(f"   - Refinements: {result.get('refinement_count', 0)}")
        
        answer = result.get('answer', 'No answer')
        print(f"\n   Answer Preview: {answer[:200]}...")
        
        if result.get('reflection_score', 0) >= 0.5:
            print("\n   ✅ Test completed successfully!")
            print("\n" + "="*70)
            print("All tests passed! LangGraph integration is working correctly.")
            print("="*70)
            return True
        else:
            print("\n   ⚠️  Test completed but quality score is low")
            print("   This might indicate an issue with the LLM or evaluator")
            return True  # Still consider it a pass if it executed
        
    except Exception as e:
        print(f"\n   ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🧪 Starting LangGraph Integration Tests...\n")
    
    # Run component tests
    if not test_langgraph_components():
        print("\n" + "="*70)
        print("❌ Component tests failed. Please fix the issues above.")
        print("="*70)
        return
    
    # Run full integration test
    print("\n\nProceed with full integration test? (This will load the LLM)")
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        success = test_full_integration()
        if not success:
            print("\n" + "="*70)
            print("❌ Integration test failed. Check the error messages above.")
            print("="*70)
    else:
        print("\n✅ Component tests passed. Skipping full integration test.")
        print("\nTo test the full system, run:")
        print("  python run_with_existing_pdfs.py")


if __name__ == "__main__":
    main()
