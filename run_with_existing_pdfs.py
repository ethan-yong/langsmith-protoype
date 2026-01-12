"""
Quick Runner Script for Existing PDF Setup
===========================================
This script is configured to work with your existing PDF structure in ./documents/
Simply run this to start using the RAG system immediately!
"""

import os
import sys
from pathlib import Path

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

def main():
    """
    Main execution function adapted for existing PDF structure.
    """
    
    # Setup Opik tracing (optional)
    opik_client = setup_opik_tracing()
    
    # ========================================================================
    # STEP 1: Extract text from PDFs
    # ========================================================================
    processor = PDFProcessor(ocr_enabled=True)
    pdf_paths = [str(p) for p in pdf_files]
    documents = processor.process_multiple_pdfs(pdf_paths)
    
    if not documents:
        print("\n❌ No documents extracted. Exiting.")
        return
    
    # ========================================================================
    # STEP 2: Create embeddings and vector store
    # ========================================================================
    embedding_manager = EmbeddingManager()
    
    # Check if vector store already exists
    VECTOR_STORE_PATH = "./faiss_index"
    vectorstore = embedding_manager.load_vector_store(VECTOR_STORE_PATH)
    
    if vectorstore is None:
        # Create new vector store
        chunks = embedding_manager.chunk_documents(documents)
        vectorstore = embedding_manager.create_vector_store(chunks, VECTOR_STORE_PATH)
    else:
        print("   (Using existing vector store. Delete ./faiss_index to rebuild)")
    
    # ========================================================================
    # STEP 3: Initialize LLaMA QA Agent
    # ========================================================================
    print("\n⚠️  Loading LLaMA model - This may take 2-5 minutes...")
    print("   (First time: ~16GB download + loading time)")
    print("   (Subsequent times: ~2 minutes)")
    
    qa_agent = LLaMAQAAgent(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        use_4bit=True,  # Use 4-bit quantization to reduce memory usage
        opik_client=opik_client  # Pass Opik client for logging
    )
    
    # Create QA chain
    qa_chain = qa_agent.create_qa_chain(vectorstore)
    
    # ========================================================================
    # STEP 4: Interactive QA Loop
    # ========================================================================
    print("\n" + "="*60)
    print("💬 INTERACTIVE QA MODE")
    print("="*60)
    print("Ask questions about sentiment analysis from your research papers.")
    print("Type 'quit' or 'exit' to stop.")
    print("\n💡 Tips:")
    print("   - The system has conversation memory")
    print("   - You can ask follow-up questions")
    print("   - Answers include source citations")
    print("="*60)
    
    # Suggested questions based on the PDFs
    print("\n📝 Suggested questions for your papers:")
    print("   • What are the main challenges in sentiment analysis?")
    print("   • How do linguistic approaches differ from machine learning?")
    print("   • What algorithms are commonly used for sentiment analysis?")
    print("   • What are the applications of sentiment analysis?")
    print("   • How is subjectivity handled in sentiment analysis?")
    print()
    
    while True:
        try:
            # Get user input
            question = input("\n👤 Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Get answer
            answer, sources = qa_agent.ask_question(qa_chain, question)
            
            # Display formatted answer with sources
            formatted_output = qa_agent.format_answer_with_sources(answer, sources)
            print(formatted_output)
        
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
