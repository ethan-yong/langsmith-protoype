"""
Examples: Evaluating Queries and Outputs Later

This demonstrates how to evaluate user queries and outputs:
1. Immediately after generation (with active trace)
2. Later without an active trace
3. Evaluating existing runs from LangSmith
"""

from langsmith_integration import (
    evaluate_query_and_output,
    evaluate_existing_runs,
    get_all_evaluators,
    log_and_evaluate_rag_response,
)


# Example 1: Evaluate immediately (with active trace)
# This is what you're already doing in run_with_existing_pdfs.py
def example_immediate_evaluation():
    """Evaluate immediately when you have an active trace."""
    result = log_and_evaluate_rag_response(
        question="What are the main challenges in sentiment analysis?",
        context="Sentiment analysis faces challenges in...",
        answer="The main challenges include...",
    )
    print("Immediate evaluation result:", result)
    print("Run ID:", result.get("evaluation", {}).get("run_id"))


# Example 2: Evaluate later without an active trace
def example_evaluate_later():
    """Evaluate queries and outputs later, even without an active trace."""
    
    # Store your queries and outputs somewhere (database, file, etc.)
    queries_and_outputs = [
        {
            "question": "What are the main challenges in sentiment analysis?",
            "context": "Sentiment analysis faces challenges in handling sarcasm...",
            "answer": "The main challenges include handling sarcasm, context understanding...",
        },
        {
            "question": "How do linguistic approaches differ from machine learning?",
            "context": "Linguistic approaches use rule-based methods...",
            "answer": "Linguistic approaches use rule-based methods while ML uses statistical patterns...",
        },
    ]
    
    # Evaluate them later
    results = []
    for item in queries_and_outputs:
        result = evaluate_query_and_output(
            question=item["question"],
            context=item["context"],
            answer=item["answer"],
            # project_name="my-rag-project",  # Optional
        )
        results.append(result)
        print(f"\n✅ Evaluated: {item['question'][:50]}...")
        print(f"   Scores: {result.get('scores', {})}")
        print(f"   Run ID: {result.get('run_id')}")
    
    return results


# Example 3: Evaluate existing runs from LangSmith
def example_evaluate_existing_runs():
    """Evaluate runs that were created earlier in LangSmith."""
    
    # Get run IDs from LangSmith dashboard or API
    # You can find these in the LangSmith UI or by querying the API
    run_ids = [
        "abc123-def456-ghi789",  # Replace with actual run IDs
        "xyz789-uvw456-rst123",
    ]
    
    # Evaluate them
    results = evaluate_existing_runs(run_ids)
    
    for run_id, result in results.items():
        print(f"\n✅ Run {run_id}:")
        print(f"   Scores: {result.get('scores', {})}")
        print(f"   Evaluation: {result.get('evaluation', {})}")


# Example 4: Evaluate with multiple evaluators
def example_multiple_evaluators():
    """Use multiple evaluators (relevance + helpfulness)."""
    
    all_evals = get_all_evaluators()
    
    result = evaluate_query_and_output(
        question="What are the main challenges in sentiment analysis?",
        context="Sentiment analysis faces challenges in...",
        answer="The main challenges include...",
        evaluators=list(all_evals.values()),  # Use all evaluators
    )
    
    print("Multiple evaluators result:", result)
    print("Scores:", result.get("scores", {}))


# Example 5: Evaluate and link to existing run
def example_link_to_existing_run():
    """Evaluate and link feedback to an existing run."""
    
    # If you have an existing run ID from a previous trace
    existing_run_id = "abc123-def456-ghi789"  # Replace with actual run ID
    
    result = evaluate_query_and_output(
        question="What are the main challenges in sentiment analysis?",
        context="Sentiment analysis faces challenges in...",
        answer="The main challenges include...",
        run_id=existing_run_id,  # Link to existing run
    )
    
    print("Linked to existing run:", result)


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluating Queries and Outputs Later")
    print("=" * 60)
    
    print("\n1. Immediate evaluation (with active trace):")
    print("   (This is what you're already doing)")
    # example_immediate_evaluation()
    
    print("\n2. Evaluate later (without active trace):")
    example_evaluate_later()
    
    print("\n3. Evaluate existing runs:")
    print("   (Uncomment and add real run IDs)")
    # example_evaluate_existing_runs()
    
    print("\n4. Multiple evaluators:")
    example_multiple_evaluators()
    
    print("\n5. Link to existing run:")
    print("   (Uncomment and add real run ID)")
    # example_link_to_existing_run()
