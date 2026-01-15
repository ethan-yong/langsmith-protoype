from typing import Any, Dict, Optional

from langsmith import traceable

HALLUCINATION_EVAL_PROMPT = """You are an expert data labeler evaluating model outputs for hallucinations.

Rubric:
- The response contains only verifiable facts supported by the input context.
- It makes no unsupported claims or assumptions.
- It does not add speculative or imagined details.
- Dates, numbers, and specific details are accurate.
- It indicates uncertainty when information is incomplete.

Instructions:
- Read the input context thoroughly.
- Identify all claims in the output.
- Cross-reference each claim with the input context.
- Note any unsupported or contradictory information.
- Consider the severity and quantity of hallucinations.

Use the following context to evaluate:
Context:
{inputs}

Output:
{outputs}

Reference (if available):
{reference_outputs}

Return your reasoning as exactly 3 bullet points in order of importance.
"""


@traceable(
    run_type="chain",
    name="rag_response_for_eval",
    metadata={
        "evaluation_prompt": HALLUCINATION_EVAL_PROMPT,
        "evaluation_summary_format": "3 bullet points",
    },
    process_inputs=lambda data: {
        "question": data.get("question"),
        "context": data.get("context"),
    },
)
def log_rag_response_for_dashboard(
    question: str,
    context: str,
    answer: str,
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Log inputs/outputs to LangSmith for dashboard-based evaluation.
    """
    payload: Dict[str, Any] = {"answer": answer}
    if reference_outputs is not None:
        payload["reference_outputs"] = reference_outputs
    return payload