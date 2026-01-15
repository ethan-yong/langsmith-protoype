from typing import Any, Dict, Optional

from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openai import OpenAI

client = Client()

# Wrap the OpenAI client for LangSmith tracing (used by the judge)
openai_client = wrappers.wrap_openai(OpenAI())


def correctness_evaluator(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    return evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,
    )


def evaluate_rag_output(
    user_input: str,
    rag_output: str,
    reference_output: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single RAG response using LangSmith + LLM-as-judge.

    Args:
        user_input: The question from the user.
        rag_output: The answer returned by the RAG model.
        reference_output: Optional ground-truth answer for correctness eval.

    Returns:
        A dictionary containing inputs, outputs, and optional feedback.
    """
    inputs = {"question": user_input}
    outputs = {"answer": rag_output}

    if reference_output is None:
        return {
            "inputs": inputs,
            "outputs": outputs,
            "feedback": None,
            "note": "No reference_output provided; correctness was not scored.",
        }

    reference_outputs = {"answer": reference_output}
    feedback = correctness_evaluator(inputs, outputs, reference_outputs)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "reference_outputs": reference_outputs,
        "feedback": feedback,
    }