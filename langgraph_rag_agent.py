"""
LangGraph RAG Agent
===================
An adaptive RAG system that uses LangGraph to:
- Analyze question complexity
- Adapt retrieval strategy dynamically
- Self-reflect on answer quality using LangSmith evaluators
- Refine queries based on evaluator feedback
- Iterate until quality threshold is met

Author: Expert Python Engineer
Date: January 2026
"""

import os
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Import the existing evaluator
from langsmith_integration import create_relevance_evaluator


class RAGState(TypedDict):
    """State that flows through the LangGraph."""
    question: str
    question_type: str  # "simple", "complex", "comparative", "synthesis"
    retrieval_k: int
    retrieved_docs: List[Document]
    context: str
    answer: str
    reflection_score: float  # From your LangSmith evaluator (0-1 scale)
    reflection_comment: str  # Detailed feedback from evaluator
    refinement_count: int
    score_history: List[Dict[str, Any]]  # Track score progression across iterations
    previous_issues: str  # Track what was wrong before refinement
    chat_history: List[dict]


class LangGraphRAGAgent:
    """
    LangGraph-based RAG agent with adaptive retrieval and self-reflection.
    """
    
    def __init__(self, vectorstore, llm):
        """
        Initialize the LangGraph RAG agent.
        
        Args:
            vectorstore: FAISS vector store
            llm: Language model instance
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.evaluator = create_relevance_evaluator()
        
        # Create a lightweight LLM for quick analysis tasks
        self.analysis_llm = ChatOpenAI(
            model=os.getenv("JUDGE_MODEL", "openai/llama3:8b"),
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
            temperature=0.0,
        )
        
        print("✅ LangGraph RAG Agent initialized")
    
    def analyze_query_node(self, state: RAGState) -> RAGState:
        """
        Analyze question to determine retrieval strategy.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with question_type and retrieval_k
        """
        question = state["question"]
        
        print(f"\n🔍 Analyzing question complexity...")
        
        # Use LLM to classify question type
        analysis_prompt = """Classify this question into ONE category:
- simple: Direct factual question (e.g., "What is X?", "Define Y")
- complex: Requires synthesis of multiple concepts (e.g., "How do X and Y relate?")
- comparative: Comparing multiple things (e.g., "Compare X and Y")
- synthesis: Building comprehensive understanding (e.g., "Explain the full process of X")

Question: {question}

Return ONLY the category name (simple/complex/comparative/synthesis):"""
        
        try:
            result = self.analysis_llm.invoke(analysis_prompt.format(question=question))
            question_type = result.content.strip().lower()
            
            # Validate question type
            valid_types = {"simple", "complex", "comparative", "synthesis"}
            if question_type not in valid_types:
                # Default to complex if invalid
                question_type = "complex"
        except Exception as e:
            print(f"⚠️  Analysis error: {e}, defaulting to 'complex'")
            question_type = "complex"
        
        # Set retrieval strategy based on type
        k_mapping = {
            "simple": 3,
            "complex": 6,
            "comparative": 8,
            "synthesis": 6
        }
        
        retrieval_k = k_mapping.get(question_type, 4)
        
        print(f"   Question type: {question_type}")
        print(f"   Retrieval k: {retrieval_k}")
        
        return {
            **state,
            "question_type": question_type,
            "retrieval_k": retrieval_k
        }
    
    def retrieve_node(self, state: RAGState) -> RAGState:
        """
        Retrieve documents with adaptive k.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with retrieved_docs and context
        """
        print(f"\n📚 Retrieving documents (k={state['retrieval_k']})...")
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": state["retrieval_k"]}
        )
        
        # #region agent log
        import json
        with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"location":"langgraph_rag_agent.py:142","message":"Retriever type and methods","data":{"retriever_type":str(type(retriever)),"has_get_relevant_documents":hasattr(retriever, "get_relevant_documents"),"has_invoke":hasattr(retriever, "invoke"),"available_methods":[m for m in dir(retriever) if not m.startswith('_')]},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","hypothesisId":"H1-H2"}) + '\n')
        # #endregion
        
        # Use invoke() for newer LangChain versions, fallback to get_relevant_documents()
        try:
            # #region agent log
            with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location":"langgraph_rag_agent.py:148","message":"Attempting invoke() method","data":{"question":state["question"][:50]},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","hypothesisId":"H1"}) + '\n')
            # #endregion
            
            docs = retriever.invoke(state["question"])
            
            # #region agent log
            with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location":"langgraph_rag_agent.py:151","message":"invoke() succeeded","data":{"num_docs":len(docs),"doc_types":[str(type(d)) for d in docs]},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","hypothesisId":"H1"}) + '\n')
            # #endregion
        except AttributeError:
            # #region agent log
            with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location":"langgraph_rag_agent.py:155","message":"invoke() failed, trying get_relevant_documents()","data":{},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","hypothesisId":"H1-fallback"}) + '\n')
            # #endregion
            
            docs = retriever.get_relevant_documents(state["question"])
            
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # #region agent log
        with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"location":"langgraph_rag_agent.py:162","message":"Documents retrieved successfully","data":{"num_docs":len(docs),"context_length":len(context)},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","hypothesisId":"H1"}) + '\n')
        # #endregion
        
        print(f"   Retrieved {len(docs)} documents")
        
        return {
            **state,
            "retrieved_docs": docs,
            "context": context
        }
    
    def generate_node(self, state: RAGState) -> RAGState:
        """
        Generate answer using LLM.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with answer
        """
        print(f"\n✍️  Generating answer...")
        
        # Create a comprehensive prompt
        prompt = PromptTemplate.from_template("""You are an expert assistant specializing in sentiment analysis research.

Context from research papers:
{context}

Question: {question}

Instructions:
- Provide a detailed, accurate answer based on the context above
- Reference specific concepts from the papers when relevant
- If the context doesn't contain enough information, acknowledge it
- Be comprehensive but concise

Answer:""")
        
        # Format prompt
        formatted_prompt = prompt.format(
            context=state["context"],
            question=state["question"]
        )
        
        # Get answer from LLM
        try:
            # Handle different LLM types
            if hasattr(self.llm, 'invoke'):
                # LangChain ChatOpenAI or similar
                result = self.llm.invoke(formatted_prompt)
                answer = result.content if hasattr(result, 'content') else str(result)
            elif hasattr(self.llm, 'predict'):
                # HuggingFacePipeline
                answer = self.llm.predict(formatted_prompt)
            else:
                # Fallback
                answer = str(self.llm(formatted_prompt))
            
            answer = answer.strip()
        except Exception as e:
            print(f"⚠️  Generation error: {e}")
            answer = f"Error generating answer: {str(e)}"
        
        print(f"   Generated {len(answer)} characters")
        
        return {**state, "answer": answer}
    
    def reflect_node(self, state: RAGState) -> RAGState:
        """
        Evaluate answer quality using existing LangSmith evaluator.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with reflection_score, reflection_comment, and score_history
        """
        print(f"\n🤔 Reflecting on answer quality...")
        
        # Create mock run object (same pattern as in langsmith_integration.py)
        class MockRun:
            def __init__(self, inputs, outputs):
                self.inputs = inputs
                self.outputs = outputs
                self.id = "reflection-run"
        
        mock_run = MockRun(
            inputs={
                "question": state["question"],
                "context": state["context"]
            },
            outputs={"answer": state["answer"]}
        )
        
        # Get evaluation with detailed feedback from your existing evaluator
        try:
            eval_result = self.evaluator(mock_run, example=None)
            
            score = eval_result.get("score", 0.5)
            comment = eval_result.get("comment", "No comment provided")
            
            print(f"   Quality score: {score:.2f}")
            print(f"   Feedback: {comment[:100]}...")
        except Exception as e:
            print(f"⚠️  Evaluation error: {e}, using default score")
            score = 0.5
            comment = f"Evaluation failed: {str(e)}"
        
        # Track score history
        iteration = state.get("refinement_count", 0) + 1
        score_history = state.get("score_history", [])
        score_history.append({
            "iteration": iteration,
            "score": score,
            "comment": comment[:100]  # Truncated for history
        })
        
        # Log with comparison (professional text only)
        if len(score_history) > 1:
            prev_score = score_history[-2]["score"]
            improvement = score - prev_score
            print(f"   [Iteration {iteration}] Quality Score: {score:.2f} (Change: {improvement:+.2f})")
        else:
            print(f"   [Iteration {iteration}] Quality Score: {score:.2f}")
        
        return {
            **state,
            "reflection_score": score,
            "reflection_comment": comment,
            "score_history": score_history
        }
    
    def refine_query_node(self, state: RAGState) -> RAGState:
        """
        Refine query based on evaluator feedback.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with refined question and increased retrieval_k
        """
        print(f"\n🔄 Refining query based on feedback...")
        
        # Parse evaluator comment to extract specific issues
        comment = state["reflection_comment"]
        original_question = state["question"]
        
        # Use LLM to create targeted refinement based on feedback
        refinement_prompt = f"""The previous answer was evaluated and found to need improvement.

Original Question: {original_question}

Evaluator Feedback: {comment}

Previous Answer Quality Score: {state["reflection_score"]:.2f} (out of 1.0)

Based on the evaluator's specific feedback, create an improved query that:
1. Addresses the issues mentioned in the feedback
2. Asks for more specific information where the answer was vague
3. Requests clarification on points that were missing

Return ONLY the improved query, nothing else:"""
        
        try:
            result = self.analysis_llm.invoke(refinement_prompt)
            refined_question = result.content.strip()
            
            # Validate refinement
            if not refined_question or len(refined_question) < 10:
                # Fallback to generic refinement
                refined_question = f"{original_question} (provide more detailed and comprehensive answer addressing: {comment[:100]})"
        except Exception as e:
            print(f"⚠️  Refinement error: {e}, using fallback")
            refined_question = f"{original_question} (provide more detailed and comprehensive answer)"
        
        refinement_count = state.get("refinement_count", 0) + 1
        new_k = min(state["retrieval_k"] + 2, 10)
        
        print(f"   Refinement #{refinement_count}")
        print(f"   New retrieval k: {new_k}")
        print(f"   Refined question: {refined_question[:100]}...")
        
        return {
            **state,
            "question": refined_question,
            "refinement_count": refinement_count,
            "retrieval_k": new_k,
            "previous_issues": comment
        }
    
    def should_refine(self, state: RAGState) -> str:
        """
        Decide whether to refine or finish based on quality score.
        
        Args:
            state: Current RAG state
            
        Returns:
            "refine" or "end"
        """
        score = state.get("reflection_score", 0)
        refinement_count = state.get("refinement_count", 0)
        
        # Refine if score is low and we haven't tried too many times
        if score < 0.7 and refinement_count < 2:
            print(f"\n⚠️  Score {score:.2f} below threshold (0.7), will refine")
            return "refine"
        else:
            if refinement_count >= 2:
                print(f"\n✅ Max refinements reached ({refinement_count}), returning best answer")
            else:
                print(f"\n✅ Score {score:.2f} meets threshold, answer is good")
            return "end"
    
    def build_graph(self) -> StateGraph:
        """
        Build the RAG graph with all nodes and edges.
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze", self.analyze_query_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("reflect", self.reflect_node)
        workflow.add_node("refine", self.refine_query_node)
        
        # Define edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "reflect")
        
        # Conditional edge: refine or finish?
        workflow.add_conditional_edges(
            "reflect",
            self.should_refine,
            {"refine": "refine", "end": END}
        )
        
        # After refining, go back to retrieve (with new k and refined question)
        workflow.add_edge("refine", "retrieve")
        
        return workflow.compile()
    
    def invoke(self, question: str, chat_history: Optional[List[dict]] = None) -> Dict[str, Any]:
        """
        Run the graph with a question.
        
        Args:
            question: User's question
            chat_history: Optional conversation history
            
        Returns:
            Final state with answer and metadata
        """
        graph = self.build_graph()
        
        # Initial state
        initial_state = {
            "question": question,
            "question_type": "",
            "retrieval_k": 4,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "reflection_score": 0.0,
            "reflection_comment": "",
            "refinement_count": 0,
            "score_history": [],
            "previous_issues": "",
            "chat_history": chat_history or []
        }
        
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        return final_state


def build_rag_graph(vectorstore, llm):
    """
    Convenience function to build and return a compiled RAG graph.
    
    Args:
        vectorstore: FAISS vector store
        llm: Language model instance
        
    Returns:
        LangGraphRAGAgent instance
    """
    return LangGraphRAGAgent(vectorstore, llm)
