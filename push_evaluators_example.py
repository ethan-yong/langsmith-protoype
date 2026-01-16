"""
Example: Pushing Evaluators to LangSmith Dashboard

This shows how to push your custom evaluators to LangSmith so they appear
in the dashboard and can be used in the UI.
"""

from langsmith_integration import push_evaluators_to_dashboard


def push_all_evaluators():
    """Push all available evaluators to LangSmith dashboard."""
    print("📤 Pushing all evaluators to LangSmith dashboard...")
    result = push_evaluators_to_dashboard()
    print(f"\n✅ Pushed evaluators: {result}")
    return result


def push_specific_evaluators():
    """Push only specific evaluators."""
    print("📤 Pushing specific evaluators...")
    result = push_evaluators_to_dashboard(
        evaluator_names=["relevance"],  # Only push relevance
        prompt_prefix="my_custom_",  # Custom prefix
    )
    print(f"\n✅ Pushed evaluators: {result}")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Pushing Evaluators to LangSmith Dashboard")
    print("=" * 60)
    
    print("\n1. Push all evaluators:")
    push_all_evaluators()
    
    print("\n2. Push specific evaluators:")
    # push_specific_evaluators()
