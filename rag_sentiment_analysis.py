"""
RAG System for Sentiment Analysis Research Papers
===================================================
This script implements a Retrieval-Augmented Generation (RAG) system that:
1. Loads and processes PDF research papers
2. Extracts text using OCR when needed
3. Creates embeddings using local Hugging Face models
4. Stores embeddings in a FAISS vector database
5. Provides a QA interface with conversation memory
6. Includes LangSmith tracing for observability

Author: Expert Python Engineer
Date: January 2026
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import torch
from PIL import Image
import pytesseract
import pdf2image
from PyPDF2 import PdfReader

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Transformers for local LLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)

# Opik tracing (optional)
try:
    from opik import Opik, configure, opik_context, track
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    print("⚠️  Opik not available. Install with: pip install opik")
    # Create dummy decorator when Opik is not available
    def track(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

# Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama library not available. Install with: pip install ollama")


class PDFProcessor:
    """
    Handles PDF loading, text extraction, and OCR processing.
    """
    
    def __init__(self, ocr_enabled: bool = True):
        """
        Initialize the PDF processor.
        
        Args:
            ocr_enabled: Whether to use OCR for scanned PDFs
        """
        self.ocr_enabled = ocr_enabled
        print("🔧 PDF Processor initialized")
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Detect if a PDF is scanned (image-based) or text-based.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF is scanned, False otherwise
        """
        try:
            reader = PdfReader(pdf_path)
            # Check first page for text content
            first_page = reader.pages[0]
            text = first_page.extract_text().strip()
            
            # If very little text is extracted, it's likely scanned
            return len(text) < 50
        except Exception as e:
            print(f"⚠️  Error checking PDF type: {e}")
            return False
    
    def extract_text_with_ocr(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from scanned PDF using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page number and extracted text
        """
        print(f"🔍 Performing OCR on scanned PDF: {pdf_path}")
        documents = []
        
        try:
            # Convert PDF pages to images
            images = pdf2image.convert_from_path(pdf_path)
            
            for page_num, image in enumerate(images, start=1):
                # Perform OCR on the image
                text = pytesseract.image_to_string(image)
                
                if text.strip():
                    documents.append({
                        'page': page_num,
                        'content': text,
                        'source': os.path.basename(pdf_path)
                    })
                    print(f"  ✓ Page {page_num} processed ({len(text)} chars)")
        
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
        
        return documents
    
    def extract_text_normal(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from regular (text-based) PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page number and extracted text
        """
        print(f"📄 Extracting text from PDF: {pdf_path}")
        documents = []
        
        try:
            reader = PdfReader(pdf_path)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                if text.strip():
                    documents.append({
                        'page': page_num,
                        'content': text,
                        'source': os.path.basename(pdf_path)
                    })
                    print(f"  ✓ Page {page_num} processed ({len(text)} chars)")
        
        except Exception as e:
            print(f"❌ Text extraction failed: {e}")
        
        return documents
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Main method to process a PDF file.
        Automatically detects if OCR is needed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted documents with metadata
        """
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            return []
        
        # Detect if PDF is scanned
        is_scanned = self.is_scanned_pdf(pdf_path)
        
        if is_scanned and self.ocr_enabled:
            return self.extract_text_with_ocr(pdf_path)
        else:
            return self.extract_text_normal(pdf_path)
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, any]]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Combined list of all extracted documents
        """
        all_documents = []
        
        for pdf_path in pdf_paths:
            print(f"\n{'='*60}")
            docs = self.process_pdf(pdf_path)
            all_documents.extend(docs)
        
        print(f"\n✅ Total documents extracted: {len(all_documents)}")
        return all_documents


class EmbeddingManager:
    """
    Manages text chunking and embedding generation using local HuggingFace models.
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model
        """
        print(f"\n🤖 Loading embedding model: {embedding_model_name}")
        
        # Use a smaller, efficient embedding model
        # (LLaMA is too large for embeddings, we'll use it for generation only)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Size of each chunk
            chunk_overlap=200,  # Overlap between chunks for context preservation
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        print("✅ Embedding manager initialized")
    
    def chunk_documents(self, documents: List[Dict[str, any]]) -> List[Document]:
        """
        Split documents into smaller chunks suitable for embeddings.
        
        Args:
            documents: List of document dictionaries with content and metadata
            
        Returns:
            List of LangChain Document objects
        """
        print(f"\n📝 Chunking {len(documents)} documents...")
        
        langchain_docs = []
        
        for doc in documents:
            # Create LangChain Document with metadata
            langchain_doc = Document(
                page_content=doc['content'],
                metadata={
                    'source': doc['source'],
                    'page': doc['page']
                }
            )
            langchain_docs.append(langchain_doc)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(langchain_docs)
        
        print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_vector_store(self, chunks: List[Document], persist_directory: str = "./faiss_index") -> FAISS:
        """
        Create FAISS vector store from document chunks.
        
        Args:
            chunks: List of document chunks
            persist_directory: Directory to save the FAISS index
            
        Returns:
            FAISS vector store object
        """
        print(f"\n🗄️  Creating FAISS vector store...")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Save the vector store locally
        vectorstore.save_local(persist_directory)
        print(f"✅ Vector store created and saved to {persist_directory}")
        
        return vectorstore
    
    def load_vector_store(self, persist_directory: str = "./faiss_index") -> Optional[FAISS]:
        """
        Load existing FAISS vector store.
        
        Args:
            persist_directory: Directory containing the FAISS index
            
        Returns:
            FAISS vector store object or None if not found
        """
        try:
            vectorstore = FAISS.load_local(
                persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Loaded existing vector store from {persist_directory}")
            return vectorstore
        except Exception as e:
            print(f"⚠️  Could not load vector store: {e}")
            return None


class LLaMAQAAgent:
    """
    Manages the LLaMA model and conversational QA chain.
    Supports both local models and remote OpenAI-compatible APIs.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        use_4bit: bool = True,
        opik_client=None,
        use_remote: bool = False,
        api_base: str = None
    ):
        """
        Initialize the QA agent with LLaMA model or remote API.
        
        Args:
            model_name: HuggingFace model name or remote model name
            use_4bit: Whether to use 4-bit quantization (local only)
            opik_client: Optional Opik client for logging
            use_remote: Use remote OpenAI-compatible API instead of local model
            api_base: Base URL for remote API (e.g., http://192.168.2.134:31180)
        """
        self.model_name = model_name
        self.opik_client = opik_client
        self.use_remote = use_remote
        
        if use_remote:
            # Use OpenAI-compatible API (Ollama) with Opik tracking
            print(f"\n🌐 Connecting to remote LLM (OpenAI-compatible): {api_base}")
            print(f"   Model: {model_name}")
            
            import requests
            from langchain.llms.base import LLM
            from typing import Optional, List, Any
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY", "not-needed")
            
            # Clean base URL
            clean_base = api_base.rstrip('/').replace('/chat/completions', '').replace('/v1', '')
            
            # OpenAI-compatible LLM with Opik tracking
            class OpenAICompatibleLLM(LLM):
                api_base: str
                api_key: str
                model: str
                temperature: float = 0.7
                max_tokens: int = 512
                opik_enabled: bool = OPIK_AVAILABLE
                
                @property
                def _llm_type(self) -> str:
                    return "openai_compatible"
                
                @track(
                    name="llm_call", 
                    tags=["ollama", "openai_compatible", "rag"],
                    project_name="sentiment-analysis-rag"
                )
                def _call(
                    self,
                    prompt: str,
                    stop: Optional[List[str]] = None,
                    run_manager: Optional[Any] = None,
                    **kwargs: Any,
                ) -> str:
                    """Call OpenAI-compatible API with Opik tracking"""
                    try:
                        # Make direct API call
                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        payload = {
                            "model": self.model,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens
                        }
                        
                        response = requests.post(
                            f"{self.api_base}/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=60
                        )
                        
                        response.raise_for_status()
                        result = response.json()
                        
                        # Update Opik context with metadata if available
                        if self.opik_enabled:
                            try:
                                usage = result.get('usage', {})
                                opik_context.update_current_span(
                                    metadata={
                                        "model": result.get('model', self.model),
                                        "api_base": self.api_base,
                                        "finish_reason": result.get('choices', [{}])[0].get('finish_reason'),
                                    },
                                    usage={
                                        "completion_tokens": usage.get('completion_tokens', 0),
                                        "prompt_tokens": usage.get('prompt_tokens', 0),
                                        "total_tokens": usage.get('total_tokens', 0),
                                    },
                                )
                            except Exception as e:
                                print(f"⚠️  Opik tracking warning: {e}")
                        
                        # Extract content
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            print(f"✅ LLM response received ({len(content)} chars)")
                            return content
                        else:
                            print(f"⚠️  Unexpected response format: {result}")
                            return "ERROR: No choices in response"
                            
                    except Exception as e:
                        print(f"\n❌ API Error: {e}")
                        import traceback
                        traceback.print_exc()
                        return f"ERROR: {str(e)}"
            
            # Initialize LLM
            self.llm = OpenAICompatibleLLM(
                api_base=clean_base,
                api_key=api_key,
                model=model_name,
                temperature=0.7,
                max_tokens=512
            )
            
            self.device = "remote"
            print(f"✅ Connected to remote LLM at {clean_base}/v1/chat/completions")
            print(f"✅ Opik tracking: {'enabled' if OPIK_AVAILABLE else 'disabled'}")
            return
        
        # Local model loading (original code)
        print(f"\n🦙 Loading LLaMA model: {model_name}")
        print("⚠️  Note: This may take several minutes on first run...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Only load local model if not using remote
        if not use_remote:
            # Configure quantization for memory efficiency
            if use_4bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Create text generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            # Wrap as LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            
            print(f"✅ LLaMA model loaded on {self.device}")
    
    def create_qa_chain(self, vectorstore: FAISS) -> ConversationalRetrievalChain:
        """
        Create a conversational retrieval QA chain.
        
        Args:
            vectorstore: FAISS vector store for retrieval
            
        Returns:
            ConversationalRetrievalChain object
        """
        print("\n🔗 Building conversational QA chain...")
        
        # Create custom prompt template
        template = """You are an expert assistant specialized in sentiment analysis research.
Use the following pieces of context from research papers to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite the source document and page number when providing information.

Context:
{context}

Question: {question}

Helpful Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create retriever from vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )
        
        # Create conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            verbose=True
        )
        
        print("✅ QA chain created successfully")
        return qa_chain
    
    @track(
        name="rag_qa_question", 
        tags=["rag", "qa"],
        project_name="sentiment-analysis-rag"
    )
    def ask_question(
        self,
        qa_chain: ConversationalRetrievalChain,
        question: str
    ) -> Tuple[str, List[Document]]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            qa_chain: The QA chain to use
            question: User's question
            
        Returns:
            Tuple of (answer, source_documents)
        """
        print(f"\n❓ Question: {question}")
        print("🤔 Thinking...")
        
        import time
        start_time = time.time()
        
        # Get answer from the chain
        result = qa_chain({"question": question})
        
        answer = result['answer']
        source_docs = result['source_documents']
        
        # Extract PDF context (the actual text from retrieved documents)
        pdf_context = "\n\n---\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})\n{doc.page_content}"
            for doc in source_docs
        ])
        
        # Update Opik context with RAG metadata
        if OPIK_AVAILABLE:
            try:
                # Add to current span metadata
                opik_context.update_current_span(
                    metadata={
                        "question": question,
                        "num_source_documents": len(source_docs),
                        "sources": [doc.metadata.get('source', 'Unknown') for doc in source_docs],
                    }
                )
                
                # Also add as tags for easier filtering
                opik_context.update_current_trace(
                    tags=["rag", "qa", "evaluated"]
                )
                
                # IMPORTANT: Set these as output fields for evaluation
                opik_context.update_current_span(
                    output={
                        "answer": answer,
                        "chatbot_response": answer,
                        "pdf_context": pdf_context,
                        "sources": source_docs
                    }
                )
                
            except Exception as e:
                print(f"⚠️  Opik context update warning: {e}")
        
        # Log to Opik if available
        if self.opik_client:
            try:
                end_time = time.time()
                duration = end_time - start_time
                
                # Prepare source metadata
                sources = [
                    {
                        "source": doc.metadata.get('source', 'Unknown'),
                        "page": doc.metadata.get('page', 'Unknown'),
                        "content_preview": doc.page_content[:200]
                    }
                    for doc in source_docs
                ]
                
                # Log the request to Opik
                self.opik_client.log_traces(
                    traces=[{
                        "name": "sentiment_analysis_qa",
                        "input": question,
                        "output": answer,
                        "metadata": {
                            "model": self.model_name,
                            "num_sources": len(source_docs),
                            "sources": sources,
                            "duration_seconds": duration,
                            "device": self.device
                        },
                        "tags": ["rag", "sentiment-analysis", "llama"]
                    }]
                )
            except Exception as e:
                print(f"⚠️  Failed to log to Opik: {e}")
        
        return answer, source_docs
    
    def format_answer_with_sources(
        self,
        answer: str,
        source_docs: List[Document]
    ) -> str:
        """
        Format the answer with source citations.
        
        Args:
            answer: The generated answer
            source_docs: List of source documents used
            
        Returns:
            Formatted answer string with sources
        """
        output = f"\n{'='*60}\n"
        output += "📝 ANSWER:\n"
        output += f"{'='*60}\n"
        output += f"{answer}\n"
        output += f"\n{'='*60}\n"
        output += "📚 SOURCES:\n"
        output += f"{'='*60}\n"
        
        # Track unique sources to avoid duplicates
        seen_sources = set()
        
        for i, doc in enumerate(source_docs, 1):
            source_key = (doc.metadata.get('source', 'Unknown'), doc.metadata.get('page', 'Unknown'))
            
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                output += f"\n[{i}] Source: {doc.metadata.get('source', 'Unknown')}\n"
                output += f"    Page: {doc.metadata.get('page', 'Unknown')}\n"
                output += f"    Excerpt: {doc.page_content[:200]}...\n"
        
        return output


def setup_opik_tracing():
    """
    Configure Opik tracing for observability.
    Requires OPIK_API_KEY environment variable.
    """
    if not OPIK_AVAILABLE:
        return None

    api_key = os.getenv("OPIK_API_KEY")
    workspace = os.getenv("OPIK_WORKSPACE", "default")
    project_name = os.getenv("OPIK_PROJECT", "sentiment-analysis-rag")

    if not api_key:
        print("\n⚠️  Opik API key not found. Set OPIK_API_KEY environment variable.")
        print("    Get your API key from: https://www.comet.com/opik")
        return None

    # Configure Opik globally (for decorator tracking)
    try:
        configure(
            api_key=api_key,
            workspace=workspace
        )
        
        # Initialize Opik client
        opik_client = Opik(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name
        )

        print("\n✅ Opik tracing enabled")
        print(f"   Project: {project_name}")
        print(f"   Workspace: {workspace}")
        print(f"   View traces at: https://www.comet.com/opik")

        return opik_client
    except Exception as e:
        print(f"\n⚠️  Failed to initialize Opik: {e}")
        return None
    
    return None


def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("🚀 RAG System for Sentiment Analysis Research Papers")
    print("="*60)
    
    # Setup Opik tracing (optional)
    opik_client = setup_opik_tracing()
    
    # ========================================================================
    # STEP 1: Configure PDF paths
    # ========================================================================
    # Add your PDF files here
    PDF_DIRECTORY = "./documents"
    
    # Example: Create directory if it doesn't exist
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    
    # ========================================================================
    # STEP 2: Check if vector store already exists
    # ========================================================================
    VECTOR_STORE_PATH = "./faiss_index"
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_vector_store(VECTOR_STORE_PATH)
    
    if vectorstore is not None:
        # Vector store exists - skip PDF processing!
        print("\n✅ Using existing vector store (skipping PDF processing)")
        print("   Vector store contains pre-processed embeddings")
        print("   To rebuild from PDFs, delete the ./faiss_index folder")
    else:
        # Vector store doesn't exist - process PDFs
        print("\n📄 No existing vector store found - will process PDFs...")
        
        # Get all PDF files from directory
        pdf_files = list(Path(PDF_DIRECTORY).glob("*.pdf"))
        
        if not pdf_files:
            print(f"\n⚠️  No PDF files found in {PDF_DIRECTORY}")
            print("   Please add PDF files to the directory and run again.")
            print("\n   Example: Place your research papers in ./documents/")
            
            # For demonstration, create a sample instruction
            print("\n" + "="*60)
            print("📖 INSTRUCTIONS:")
            print("="*60)
            print("1. Create a 'documents' folder in this directory")
            print("2. Add your PDF research papers on sentiment analysis")
            print("3. Run this script again")
            print("\nThe script will automatically:")
            print("  - Detect if PDFs need OCR")
            print("  - Extract and chunk the text")
            print("  - Create embeddings")
            print("  - Build a QA system")
            return
        
        pdf_paths = [str(p) for p in pdf_files]
        print(f"\n📁 Found {len(pdf_paths)} PDF files:")
        for path in pdf_paths:
            print(f"   - {os.path.basename(path)}")
        
        # ====================================================================
        # STEP 3: Extract text from PDFs
        # ====================================================================
        processor = PDFProcessor(ocr_enabled=True)
        documents = processor.process_multiple_pdfs(pdf_paths)
        
        if not documents:
            print("\n❌ No documents extracted. Exiting.")
            return
        
        # ====================================================================
        # STEP 4: Create embeddings and vector store
        # ====================================================================
        chunks = embedding_manager.chunk_documents(documents)
        vectorstore = embedding_manager.create_vector_store(chunks, VECTOR_STORE_PATH)
    
    # ========================================================================
    # STEP 5: Initialize LLaMA QA Agent
    # ========================================================================
    # Check if using remote API
    use_remote = os.getenv("USE_REMOTE_LLM", "false").lower() == "true"
    api_base = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    
    qa_agent = LLaMAQAAgent(
        model_name=model_name,
        use_4bit=True,  # Use 4-bit quantization (for local models only)
        opik_client=opik_client,  # Pass Opik client for logging
        use_remote=use_remote,
        api_base=api_base
    )
    
    # Create QA chain
    qa_chain = qa_agent.create_qa_chain(vectorstore)
    
    # ========================================================================
    # STEP 6: Interactive QA Loop
    # ========================================================================
    print("\n" + "="*60)
    print("💬 INTERACTIVE QA MODE")
    print("="*60)
    print("Ask questions about sentiment analysis from your research papers.")
    print("Type 'quit' or 'exit' to stop.")
    print("The system has conversation memory, so you can ask follow-up questions!")
    print("="*60)
    
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
    main()
