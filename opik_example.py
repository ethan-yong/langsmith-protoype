"""
Opik Integration Example
========================
This script demonstrates how to use Opik for logging and tracing
in the RAG sentiment analysis system.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Opik
try:
    from opik import Opik
    from opik.decorators import track
except ImportError:
    print("❌ Opik not installed. Run: pip install opik")
    exit(1)

# Import RAG components
from rag_sentiment_analysis import (
    PDFProcessor,
    EmbeddingManager,
    LLaMAQAAgent
)


# ============================================================================
# Example 1: Basic Opik Initialization
# ============================================================================

def example_basic_initialization():
    """
    Basic example of initializing Opik client.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Opik Initialization")
    print("="*60)
    
    # Initialize Opik
    opik_client = Opik(
        api_key=os.getenv("OPIK_API_KEY"),
        workspace=os.getenv("OPIK_WORKSPACE", "default"),
        project_name="sentiment-analysis-rag"
    )
    
    print("✅ Opik client initialized")
    print(f"   Workspace: {os.getenv('OPIK_WORKSPACE', 'default')}")
    print(f"   Project: sentiment-analysis-rag")
    
    return opik_client


# ============================================================================
# Example 2: Manual Logging
# ============================================================================

def example_manual_logging():
    """
    Manually log a prompt and response to Opik.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Manual Logging")
    print("="*60)
    
    opik_client = Opik(
        api_key=os.getenv("OPIK_API_KEY"),
        project_name="sentiment-analysis-rag"
    )
    
    # Simulate a Q&A interaction
    question = "What are the main approaches to sentiment analysis?"
    answer = """The main approaches to sentiment analysis include:
    1. Lexicon-based methods using sentiment dictionaries
    2. Machine learning approaches (SVM, Naive Bayes, Random Forests)
    3. Deep learning methods (CNN, LSTM, Transformers)
    4. Transfer learning with pre-trained models like BERT"""
    
    start_time = time.time()
    time.sleep(1)  # Simulate processing
    end_time = time.time()
    
    # Log to Opik
    opik_client.log_traces(
        traces=[{
            "name": "sentiment_analysis_qa",
            "input": question,
            "output": answer,
            "metadata": {
                "model": "llama-3.1-8b",
                "duration_seconds": end_time - start_time,
                "num_sources": 3,
                "device": "cuda"
            },
            "tags": ["rag", "sentiment-analysis", "manual-example"]
        }]
    )
    
    print("✅ Logged to Opik")
    print(f"   Question: {question[:50]}...")
    print(f"   Answer: {answer[:50]}...")
    print(f"   Duration: {end_time - start_time:.2f}s")
    print("\n   View at: https://www.comet.com/opik")


# ============================================================================
# Example 3: Using Decorators
# ============================================================================

@track(
    name="pdf_text_extraction",
    project_name="sentiment-analysis-rag"
)
def extract_pdf_text(pdf_path: str) -> dict:
    """
    Extract text from PDF - automatically tracked by Opik.
    """
    processor = PDFProcessor(ocr_enabled=False)
    
    # Simulate extraction
    print(f"   Processing: {Path(pdf_path).name}")
    time.sleep(0.5)
    
    return {
        "pdf": Path(pdf_path).name,
        "pages": 10,
        "characters": 5432,
        "status": "success"
    }


def example_decorator_tracking():
    """
    Use Opik's @track decorator for automatic logging.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Decorator-based Tracking")
    print("="*60)
    
    # Just call the function - automatically logged!
    result = extract_pdf_text("./documents/sample_paper.pdf")
    
    print("✅ Function execution automatically logged to Opik")
    print(f"   Result: {result}")
    print("\n   View trace at: https://www.comet.com/opik")


# ============================================================================
# Example 4: Full RAG Pipeline with Opik
# ============================================================================

def example_full_rag_with_opik():
    """
    Complete RAG pipeline with Opik logging.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Full RAG Pipeline with Opik")
    print("="*60)
    
    # Check if PDFs exist
    pdf_dir = Path("./documents")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in ./documents/")
        print("   Add PDFs to run this example")
        return
    
    # Initialize Opik
    opik_client = Opik(
        api_key=os.getenv("OPIK_API_KEY"),
        project_name="sentiment-analysis-rag"
    )
    
    print("✅ Opik initialized")
    
    # Process PDFs (simplified for example)
    processor = PDFProcessor(ocr_enabled=False)
    print(f"\n📄 Processing {len(pdf_files)} PDFs...")
    
    # Log PDF processing
    for pdf in pdf_files[:1]:  # Just first PDF for demo
        start = time.time()
        
        # Process
        docs = processor.process_pdf(str(pdf))
        
        duration = time.time() - start
        
        # Log to Opik
        opik_client.log_traces(
            traces=[{
                "name": "pdf_processing",
                "input": str(pdf.name),
                "output": {"num_pages": len(docs)},
                "metadata": {
                    "pdf_name": pdf.name,
                    "pages_extracted": len(docs),
                    "duration_seconds": duration,
                    "ocr_used": False
                },
                "tags": ["pdf", "extraction"]
            }]
        )
        
        print(f"   ✅ Processed: {pdf.name} ({len(docs)} pages)")
        print(f"      Logged to Opik")
    
    print("\n✅ Full pipeline logged to Opik")
    print("   View dashboard: https://www.comet.com/opik")


# ============================================================================
# Example 5: Batch Logging Multiple Traces
# ============================================================================

def example_batch_logging():
    """
    Log multiple traces in a single batch.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Logging")
    print("="*60)
    
    opik_client = Opik(
        api_key=os.getenv("OPIK_API_KEY"),
        project_name="sentiment-analysis-rag"
    )
    
    # Multiple Q&A pairs
    qa_pairs = [
        {
            "question": "What is sentiment analysis?",
            "answer": "Sentiment analysis is the computational study of opinions...",
            "duration": 2.3
        },
        {
            "question": "How do transformers help?",
            "answer": "Transformers provide context-aware representations...",
            "duration": 1.8
        },
        {
            "question": "What are the challenges?",
            "answer": "Key challenges include sarcasm detection...",
            "duration": 2.1
        }
    ]
    
    # Prepare traces
    traces = []
    for i, qa in enumerate(qa_pairs):
        traces.append({
            "name": f"qa_batch_{i+1}",
            "input": qa["question"],
            "output": qa["answer"],
            "metadata": {
                "model": "llama-3.1-8b",
                "duration_seconds": qa["duration"],
                "batch_index": i
            },
            "tags": ["batch", "qa"]
        })
    
    # Log all at once
    opik_client.log_traces(traces=traces)
    
    print(f"✅ Logged {len(traces)} traces in batch")
    print("   View at: https://www.comet.com/opik")


# ============================================================================
# Example 6: Error Tracking
# ============================================================================

def example_error_tracking():
    """
    Track errors and failures with Opik.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Error Tracking")
    print("="*60)
    
    opik_client = Opik(
        api_key=os.getenv("OPIK_API_KEY"),
        project_name="sentiment-analysis-rag"
    )
    
    try:
        # Simulate an error
        raise ValueError("Invalid input format: expected string, got int")
    except Exception as e:
        # Log the error to Opik
        opik_client.log_traces(
            traces=[{
                "name": "processing_error",
                "input": "malformed_input",
                "output": str(e),
                "metadata": {
                    "error_type": type(e).__name__,
                    "status": "failed",
                    "timestamp": time.time()
                },
                "tags": ["error", "debug"]
            }]
        )
        
        print("❌ Error occurred and logged to Opik")
        print(f"   Error: {e}")
        print("   View error details at: https://www.comet.com/opik")


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """
    Main function with example menu.
    """
    # Check API key
    if not os.getenv("OPIK_API_KEY"):
        print("\n❌ OPIK_API_KEY not found!")
        print("   Set it in your .env file or environment:")
        print("   $env:OPIK_API_KEY=\"your_key_here\"")
        print("\n   Get your key from: https://www.comet.com/opik")
        return
    
    print("\n" + "="*60)
    print("🔍 Opik Integration Examples")
    print("="*60)
    print("\nAvailable examples:")
    print("1. Basic Initialization")
    print("2. Manual Logging")
    print("3. Decorator-based Tracking")
    print("4. Full RAG Pipeline with Opik")
    print("5. Batch Logging")
    print("6. Error Tracking")
    print("7. Run All Examples")
    print("0. Exit")
    
    choice = input("\nSelect example (0-7): ").strip()
    
    examples = {
        '1': example_basic_initialization,
        '2': example_manual_logging,
        '3': example_decorator_tracking,
        '4': example_full_rag_with_opik,
        '5': example_batch_logging,
        '6': example_error_tracking,
    }
    
    if choice == '0':
        print("👋 Goodbye!")
        return
    elif choice == '7':
        # Run all examples
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"❌ Error in example: {e}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")
    
    print("\n" + "="*60)
    print("✅ Examples completed!")
    print("   View all traces at: https://www.comet.com/opik")
    print("="*60)


if __name__ == "__main__":
    main()
