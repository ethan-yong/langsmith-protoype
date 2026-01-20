"""
Quick Runner Script for Existing PDF Setup
===========================================
This script is configured to work with your existing PDF structure in ./documents/
Simply run this to start using the RAG system immediately!
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows encoding for emojis (must be first!)
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Load environment variables from .env file
load_dotenv()

# Update the main script to use the documents folder
PDF_DIRECTORY = "./documents"

# Check if documents folder exists
if not Path(PDF_DIRECTORY).exists():
    print(f"❌ Directory not found: {PDF_DIRECTORY}")
    print("   Please make sure your PDFs are in the ./documents/ folder")
    sys.exit(1)

# Get PDF files
pdf_files = list(Path(PDF_DIRECTORY).glob("*.pdf"))

if not pdf_files:
    print(f"❌ No PDF files found in {PDF_DIRECTORY}")
    sys.exit(1)

print("="*60)
print("🚀 RAG System - Quick Start with Existing PDFs")
print("="*60)
print(f"\n📁 Found {len(pdf_files)} PDF files in ./documents/:")
for pdf in pdf_files:
    print(f"   ✓ {pdf.name}")

print("\n" + "="*60)
print("Starting RAG system...")
print("="*60)

# Import and run the main system
from rag_sentiment_analysis import (
    PDFProcessor,
    EmbeddingManager,
    LLaMAQAAgent,
    setup_opik_tracing
)
from langsmith_integration import log_and_evaluate_rag_response
from pdf_source_extractor import PDFSourceIndex
from langgraph_rag_agent import build_rag_graph

def main():
    """
    Main execution function adapted for existing PDF structure.
    """

    # Setup Opik tracing (optional)
    opik_client = setup_opik_tracing()
    
    # #region agent log
    import json, time
    with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location":"run_with_existing_pdfs.py:66","message":"Opik setup complete, starting vectorstore load","data":{},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"H4"}) + '\n')
    # #endregion

    # ========================================================================
    # STEP 1: Check if vector store already exists
    # ========================================================================
    VECTOR_STORE_PATH = "./faiss_index"
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_vector_store(VECTOR_STORE_PATH)
    
    # #region agent log
    with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location":"run_with_existing_pdfs.py:73","message":"Vectorstore loaded","data":{"vectorstore_exists":vectorstore is not None},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"H4"}) + '\n')
    # #endregion

    if vectorstore is not None:
        # Vector store exists - skip PDF processing!
        print("\n✅ Using existing vector store (skipping PDF processing)")
        print("   To rebuild from PDFs, delete the ./faiss_index folder")
    else:
        # Vector store doesn't exist - process PDFs
        print("\n📄 No existing vector store found - processing PDFs...")

        # ====================================================================
        # STEP 2: Extract text from PDFs (only if needed)
        # ====================================================================
        processor = PDFProcessor(ocr_enabled=True)
        pdf_paths = [str(p) for p in pdf_files]
        documents = processor.process_multiple_pdfs(pdf_paths)

        if not documents:
            print("\n❌ No documents extracted. Exiting.")
            return

        # ====================================================================
        # STEP 3: Create embeddings and vector store
        # ====================================================================
        chunks = embedding_manager.chunk_documents(documents)
        vectorstore = embedding_manager.create_vector_store(chunks, VECTOR_STORE_PATH)

    # Build or load verbatim PDF index for efficient lookup
    # #region agent log
    with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location":"run_with_existing_pdfs.py:101","message":"Starting PDF index build","data":{},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"H1"}) + '\n')
    # #endregion
    
    pdf_index = PDFSourceIndex(PDF_DIRECTORY, cache_path="./pdf_page_index.json")
    if not pdf_index.load_cache():
        pdf_index.build()
        pdf_index.save_cache()
    
    # #region agent log
    with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location":"run_with_existing_pdfs.py:105","message":"PDF index complete","data":{},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"H1"}) + '\n')
    # #endregion

    # ========================================================================
    # STEP 4: Initialize LLaMA QA Agent
    # ========================================================================
    print("\n⚠️  Loading LLaMA model - This may take 2-5 minutes...")
    print("   (First time: ~16GB download + loading time)")
    print("   (Subsequent times: ~2 minutes)")

    # Check if using remote API
    use_remote = os.getenv("USE_REMOTE_LLM", "false").lower() == "true"
    api_base = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("LLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")

    # Debug output
    print(f"\n🔍 DEBUG: Environment Variables:")
    print(f"   USE_REMOTE_LLM: {os.getenv('USE_REMOTE_LLM')} -> use_remote={use_remote}")
    print(f"   OPENAI_API_BASE: {api_base}")
    print(f"   LLM_MODEL: {model_name}")

    # #region agent log
    with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location":"run_with_existing_pdfs.py:124","message":"Starting LLaMA agent initialization","data":{"model_name":model_name,"use_remote":use_remote},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"H2"}) + '\n')
    # #endregion
    
    qa_agent = LLaMAQAAgent(
        model_name=model_name,
        use_4bit=True,  # Use 4-bit quantization (for local models only)
        opik_client=opik_client,  # Pass Opik client for logging
        use_remote=use_remote,
        api_base=api_base
    )
    
    # #region agent log
    with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location":"run_with_existing_pdfs.py:131","message":"LLaMA agent initialized","data":{},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"H2"}) + '\n')
    # #endregion

    # Build LangGraph RAG agent instead of simple QA chain
    print("\n🔗 Building LangGraph RAG agent with adaptive retrieval...")
    
    # #region agent log
    with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location":"run_with_existing_pdfs.py:134","message":"Starting LangGraph build","data":{},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"H3"}) + '\n')
    # #endregion
    
    graph_agent = build_rag_graph(vectorstore, qa_agent.llm)
    
    # #region agent log
    with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location":"run_with_existing_pdfs.py:135","message":"LangGraph built successfully","data":{},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"H3"}) + '\n')
    # #endregion

    # PDF index already built above (lines 101-105) with caching - no need to rebuild
    # (Removed duplicate pdf_index.build() that was causing 30+ minute hang)

    # ========================================================================
    # STEP 5: Interactive QA Loop with LangGraph
    # ========================================================================
    print("\n" + "="*60)
    print("💬 INTERACTIVE QA MODE (LangGraph-Enhanced)")
    print("="*60)
    print("Ask questions about sentiment analysis from your research papers.")
    print("Type 'quit' or 'exit' to stop.")
    print("\n💡 New Features:")
    print("   - Adaptive retrieval based on question complexity")
    print("   - Self-reflection and automatic refinement")
    print("   - Quality-driven answer generation")
    print("="*60)

    # Suggested questions based on the PDFs
    print("\n📝 Suggested questions for your papers:")
    print("   • What are the main challenges in sentiment analysis?")
    print("   • How do linguistic approaches differ from machine learning?")
    print("   • What algorithms are commonly used for sentiment analysis?")
    print("   • What are the applications of sentiment analysis?")
    print("   • How is subjectivity handled in sentiment analysis?")
    print()
    
    # #region agent log
    with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location":"run_with_existing_pdfs.py:171","message":"Initialization complete, entering interactive loop","data":{},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"ALL"}) + '\n')
    # #endregion

    while True:
        try:
            # Get user input
            question = input("\n👤 Your question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            print("\n" + "="*60)
            print("🤖 LangGraph RAG Agent Processing...")
            print("="*60)

            # Run the graph - it will automatically:
            # 1. Analyze question complexity
            # 2. Retrieve with adaptive k
            # 3. Generate answer
            # 4. Reflect on quality
            # 5. Refine if needed (up to 2 times)
            result = graph_agent.invoke(question)

            # Extract answer and sources from the graph result
            answer = result.get("answer", "No answer generated")
            sources = result.get("retrieved_docs", [])
            
            # Display graph execution summary
            print("\n" + "="*60)
            print("GRAPH EXECUTION SUMMARY")
            print("="*60)
            print(f"Question Type: {result.get('question_type', 'unknown')}")
            print(f"Documents Retrieved: {len(sources)}")
            
            # Display score progression
            score_history = result.get('score_history', [])
            if score_history:
                print(f"\nSCORE PROGRESSION:")
                for entry in score_history:
                    iteration = entry['iteration']
                    score = entry['score']
                    print(f"   Iteration {iteration}: {score:.2f}/1.0")
                
                # Show improvement
                if len(score_history) > 1:
                    first_score = score_history[0]['score']
                    final_score = score_history[-1]['score']
                    improvement = final_score - first_score
                    print(f"   Total Improvement: {improvement:+.2f}")
            
            print(f"\nFinal Quality Score: {result.get('reflection_score', 0):.2f}/1.0")
            print(f"Refinements Made: {result.get('refinement_count', 0)}")
            if result.get('refinement_count', 0) > 0:
                print(f"Evaluator Feedback: {result.get('reflection_comment', '')[:150]}...")
            print("="*60)

            # Display answer with sources
            formatted_output = qa_agent.format_answer_with_sources(answer, sources)
            print(formatted_output)

            # Build PDF context for additional logging
            pdf_context = pdf_index.build_context_from_sources(sources)

            # Log to LangSmith for tracking (external to graph evaluation)
            log_result = log_and_evaluate_rag_response(
                question=question,
                context=pdf_context,
                answer=answer,
            )

            if log_result.get("evaluation"):
                print("\n📝 External LangSmith Evaluation:")
                print(log_result["evaluation"])

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies first
    try:
        import torch
        import transformers
        import langchain
        import faiss
        print("✅ All dependencies found\n")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Run main
    main()
