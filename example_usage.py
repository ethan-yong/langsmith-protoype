"""
Example Usage Script
====================
This script demonstrates how to use the RAG system programmatically
and provides examples of common use cases.
"""

from rag_sentiment_analysis import (
    PDFProcessor,
    EmbeddingManager,
    LLaMAQAAgent
)
from config import config
from pathlib import Path

# ============================================================================
# Example 1: Basic Usage - Process PDFs and Ask Questions
# ============================================================================

def example_basic_usage():
    """
    Basic example: Load PDFs, create vector store, and ask questions.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Step 1: Process PDFs
    processor = PDFProcessor(ocr_enabled=True)
    pdf_paths = ["./documents/paper1.pdf"]  # Add your PDF paths
    documents = processor.process_multiple_pdfs(pdf_paths)
    
    # Step 2: Create embeddings and vector store
    embedding_manager = EmbeddingManager()
    chunks = embedding_manager.chunk_documents(documents)
    vectorstore = embedding_manager.create_vector_store(chunks)
    
    # Step 3: Initialize QA agent
    qa_agent = LLaMAQAAgent(use_4bit=True)
    qa_chain = qa_agent.create_qa_chain(vectorstore)
    
    # Step 4: Ask questions
    questions = [
        "What are the main approaches to sentiment analysis?",
        "How do neural networks help with sentiment classification?",
        "What datasets are commonly used?"
    ]
    
    for question in questions:
        answer, sources = qa_agent.ask_question(qa_chain, question)
        print(qa_agent.format_answer_with_sources(answer, sources))


# ============================================================================
# Example 2: Using Existing Vector Store (Fast)
# ============================================================================

def example_use_existing_vectorstore():
    """
    Faster example: Load existing vector store instead of recreating.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Using Existing Vector Store")
    print("="*60)
    
    # Load existing vector store (much faster!)
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_vector_store("./faiss_index")
    
    if vectorstore is None:
        print("No existing vector store found. Run example 1 first.")
        return
    
    # Initialize QA agent and chain
    qa_agent = LLaMAQAAgent(use_4bit=True)
    qa_chain = qa_agent.create_qa_chain(vectorstore)
    
    # Ask a question
    answer, sources = qa_agent.ask_question(
        qa_chain,
        "What are the latest techniques in sentiment analysis?"
    )
    print(qa_agent.format_answer_with_sources(answer, sources))


# ============================================================================
# Example 3: Batch Processing Multiple Questions
# ============================================================================

def example_batch_questions():
    """
    Process multiple questions in batch and save results.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    # Load vector store
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_vector_store("./faiss_index")
    
    if vectorstore is None:
        print("No vector store found. Run example 1 first.")
        return
    
    # Initialize QA
    qa_agent = LLaMAQAAgent(use_4bit=True)
    qa_chain = qa_agent.create_qa_chain(vectorstore)
    
    # List of questions to process
    questions = [
        "What are the main challenges in sentiment analysis?",
        "How do transformers improve sentiment analysis?",
        "What evaluation metrics are used?",
        "What are the applications of sentiment analysis?",
        "What are the limitations of current approaches?"
    ]
    
    # Process all questions
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Processing: {question}")
        answer, sources = qa_agent.ask_question(qa_chain, question)
        
        results.append({
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'file': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown')
                }
                for doc in sources
            ]
        })
    
    # Save results to file
    import json
    with open('qa_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to qa_results.json")


# ============================================================================
# Example 4: Custom Retrieval with Filtering
# ============================================================================

def example_custom_retrieval():
    """
    Advanced: Custom retrieval with metadata filtering.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Retrieval")
    print("="*60)
    
    # Load vector store
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_vector_store("./faiss_index")
    
    if vectorstore is None:
        print("No vector store found. Run example 1 first.")
        return
    
    # Direct similarity search
    query = "sentiment analysis with neural networks"
    docs = vectorstore.similarity_search(query, k=5)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(docs)} relevant documents:\n")
    
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"    Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"    Content: {doc.page_content[:200]}...")
        print()


# ============================================================================
# Example 5: Adding New Documents to Existing Store
# ============================================================================

def example_add_documents():
    """
    Add new PDFs to existing vector store without rebuilding.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Adding New Documents")
    print("="*60)
    
    # Load existing vector store
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_vector_store("./faiss_index")
    
    if vectorstore is None:
        print("No vector store found. Create one first with example 1.")
        return
    
    # Process new PDFs
    processor = PDFProcessor(ocr_enabled=True)
    new_pdf = "./documents/new_paper.pdf"  # Your new PDF
    
    if not Path(new_pdf).exists():
        print(f"PDF not found: {new_pdf}")
        return
    
    print(f"Processing new PDF: {new_pdf}")
    new_documents = processor.process_pdf(new_pdf)
    
    # Chunk new documents
    new_chunks = embedding_manager.chunk_documents(new_documents)
    
    # Add to existing vector store
    vectorstore.add_documents(new_chunks)
    
    # Save updated store
    vectorstore.save_local("./faiss_index")
    
    print(f"✅ Added {len(new_chunks)} new chunks to vector store")


# ============================================================================
# Example 6: Conversation with Memory
# ============================================================================

def example_conversation_memory():
    """
    Demonstrate conversation memory with follow-up questions.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Conversation Memory")
    print("="*60)
    
    # Load vector store
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_vector_store("./faiss_index")
    
    if vectorstore is None:
        print("No vector store found. Run example 1 first.")
        return
    
    # Initialize QA
    qa_agent = LLaMAQAAgent(use_4bit=True)
    qa_chain = qa_agent.create_qa_chain(vectorstore)
    
    # Conversation with context
    conversation = [
        "What is sentiment analysis?",
        "What are its main applications?",  # "its" refers to sentiment analysis
        "Can you give me more details about the last application you mentioned?"
    ]
    
    for question in conversation:
        print(f"\n👤 Question: {question}")
        answer, sources = qa_agent.ask_question(qa_chain, question)
        print(f"🤖 Answer: {answer}\n")
        print("-" * 60)


# ============================================================================
# Example 7: Export Vector Store Statistics
# ============================================================================

def example_vectorstore_stats():
    """
    Get statistics about the vector store.
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Vector Store Statistics")
    print("="*60)
    
    # Load vector store
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_vector_store("./faiss_index")
    
    if vectorstore is None:
        print("No vector store found.")
        return
    
    # Get all documents
    try:
        # FAISS doesn't have a direct method to get all docs
        # We can search with a generic query
        all_docs = vectorstore.similarity_search("sentiment analysis", k=1000)
        
        print(f"Total documents in store: {len(all_docs)}")
        
        # Count by source
        sources = {}
        for doc in all_docs:
            source = doc.metadata.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print("\nDocuments by source:")
        for source, count in sorted(sources.items()):
            print(f"  {source}: {count} chunks")
        
        # Average chunk length
        avg_length = sum(len(doc.page_content) for doc in all_docs) / len(all_docs)
        print(f"\nAverage chunk length: {avg_length:.0f} characters")
        
    except Exception as e:
        print(f"Error getting stats: {e}")


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """
    Main function with example menu.
    """
    print("\n" + "="*60)
    print("🎯 RAG System - Usage Examples")
    print("="*60)
    print("\nAvailable examples:")
    print("1. Basic Usage (full pipeline)")
    print("2. Use Existing Vector Store (fast)")
    print("3. Batch Process Questions")
    print("4. Custom Retrieval")
    print("5. Add New Documents")
    print("6. Conversation with Memory")
    print("7. Vector Store Statistics")
    print("8. Run All Examples")
    print("0. Exit")
    
    choice = input("\nSelect example (0-8): ").strip()
    
    examples = {
        '1': example_basic_usage,
        '2': example_use_existing_vectorstore,
        '3': example_batch_questions,
        '4': example_custom_retrieval,
        '5': example_add_documents,
        '6': example_conversation_memory,
        '7': example_vectorstore_stats,
    }
    
    if choice == '0':
        print("👋 Goodbye!")
        return
    elif choice == '8':
        # Run all examples
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"❌ Error: {e}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
