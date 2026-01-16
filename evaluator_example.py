"""
Example: Using Custom Evaluator Objects

This demonstrates how to use the evaluator objects created in langsmith_integration.py
"""

from langsmith_integration import (
    create_relevance_evaluator,
    create_helpfulness_evaluator,
    get_all_evaluators,
    get_default_evaluators,
    log_and_evaluate_rag_response,
)

# Example 1: Use default evaluators (relevance only)
def example_default_evaluation():
    """Use default evaluators."""
    result = log_and_evaluate_rag_response(
        question="What are the main challenges in sentiment analysis?",
        context="Sentiment analysis faces challenges in...",
        answer="The main challenges include...",
    )
    print("Default evaluation result:", result)


# Example 2: Use specific evaluators
def example_custom_evaluators():
    """Use specific evaluators."""
    # Create evaluators
    relevance_eval = create_relevance_evaluator()
    helpfulness_eval = create_helpfulness_evaluator()
    
    # Use both evaluators
    result = log_and_evaluate_rag_response(
        question="What are the main challenges in sentiment analysis?",
        context="Sentiment analysis faces challenges in...",
        answer="The main challenges include...",
        evaluators=[relevance_eval, helpfulness_eval],
    )
    print("Custom evaluation result:", result)


# Example 3: Get all available evaluators
def example_get_all_evaluators():
    """Get all available evaluators."""
    all_evals = get_all_evaluators()
    print("Available evaluators:", list(all_evals.keys()))
    
    # Use all evaluators
    result = log_and_evaluate_rag_response(
        question="What are the main challenges in sentiment analysis?",
        context="Sentiment analysis faces challenges in...",
        answer="The main challenges include...",
        evaluators=list(all_evals.values()),
    )
    print("All evaluators result:", result)


# Example 4: Use default evaluators list
def example_default_list():
    """Use the default evaluators list."""
    default_evals = get_default_evaluators()
    print(f"Default evaluators count: {len(default_evals)}")
    
    result = log_and_evaluate_rag_response(
        question="What are the main challenges in sentiment analysis?",
        context="Sentiment analysis faces challenges in...",
        answer="The main challenges include...",
        evaluators=default_evals,
    )
    print("Default list result:", result)


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluator Examples")
    print("=" * 60)
    
    print("\n1. Default evaluation:")
    example_default_evaluation()
    
    print("\n2. Custom evaluators:")
    example_custom_evaluators()
    
    print("\n3. Get all evaluators:")
    example_get_all_evaluators()
    
    print("\n4. Default evaluators list:")
    example_default_list()
