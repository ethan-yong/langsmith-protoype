import os
import re
import json
from typing import Any, Dict, Optional, Callable

from openai import OpenAI, AsyncOpenAI
from langsmith import Client, traceable, RunTree, wrappers, evaluate
from langsmith.run_helpers import get_current_run_tree
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Initialize LangSmith client
_client = Client()

# Judge model configuration
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/llama3:8b")
JUDGE_API_BASE = os.getenv("OPENAI_API_BASE")
JUDGE_API_KEY = os.getenv("OPENAI_API_KEY")

# Cache for evaluators (created once, reused automatically)
_evaluator_cache: Dict[str, Callable] = {}
_template_version = "v3"  # Increment this to force cache invalidation after template changes

# IMPORTANT: Clear cache on module load if version doesn't match
# This ensures template fixes are picked up immediately
if "_version" not in _evaluator_cache or _evaluator_cache.get("_version") != _template_version:
    _evaluator_cache.clear()
    _evaluator_cache["_version"] = _template_version


def _create_judge_llm():
    """Create LLM instance for evaluation with LangSmith tracing."""
    try:
        # Create ChatOpenAI with configuration
        # LangSmith tracing will work automatically if LANGSMITH_TRACING=true is set
        # We pass base_url and api_key directly to ChatOpenAI
        if JUDGE_API_BASE:
            # Use custom API base (e.g., LiteLLM)
            return ChatOpenAI(
                model=JUDGE_MODEL,
                base_url=JUDGE_API_BASE,
                api_key=JUDGE_API_KEY or "not-needed",
                temperature=0.0,
            )
        else:
            # Use default OpenAI
            return ChatOpenAI(
                model=JUDGE_MODEL,
                api_key=JUDGE_API_KEY,
                temperature=0.0,
            )
    except Exception as e:
        print(f"⚠️  Failed to create judge LLM: {e}")
        return None


def _parse_answer_for_opik(answer: str, context: str) -> Dict[str, Any]:
    """
    Parse an answer text to create structured Opik-compatible trace.
    
    Args:
        answer: The answer text to parse
        context: The context used to generate the answer
        
    Returns:
        Dictionary with answer_summary, key_points, and concepts_from_context
    """
    # Extract summary: first 2-3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    summary_sentences = sentences[:min(3, len(sentences))]
    answer_summary = ' '.join(summary_sentences)
    
    # Extract key points: look for bullet points, numbered lists, or meaningful sentences
    key_points = []
    
    # First, check for explicit bullet points or numbered lists
    bullet_pattern = r'(?:^|\n)\s*(?:[-•*]|\d+\.)\s+(.+?)(?=\n|$)'
    bullets = re.findall(bullet_pattern, answer, re.MULTILINE)
    
    if bullets:
        # Use the bullets/numbered items
        key_points = [b.strip() for b in bullets[:5]]
    else:
        # Extract meaningful sentences (longer than 20 chars, not questions)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and not sent.endswith('?'):
                # Remove common filler starts
                sent = re.sub(r'^(However|Additionally|Furthermore|Moreover|In addition),?\s*', '', sent)
                key_points.append(sent)
                if len(key_points) >= 5:
                    break
    
    # Extract concepts from context: find capitalized terms and technical phrases
    concepts = []
    
    # Common technical terms in the domain
    common_terms = [
        'sentiment analysis', 'machine learning', 'natural language processing',
        'nlp', 'deep learning', 'neural network', 'classification',
        'feature extraction', 'opinion mining', 'subjectivity', 'polarity',
        'recurrent neural network', 'rnn', 'lstm', 'transformer',
        'word embedding', 'word2vec', 'glove', 'bert',
        'supervised learning', 'unsupervised learning', 'lexicon',
        'bag of words', 'tf-idf', 'convolutional neural network', 'cnn'
    ]
    
    # Find terms that appear in both answer and context
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    for term in common_terms:
        if term in answer_lower and term in context_lower:
            if term not in concepts:
                concepts.append(term)
    
    # Also extract capitalized phrases (2-4 words) that appear in both
    cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    cap_phrases = re.findall(cap_pattern, answer)
    
    for phrase in cap_phrases:
        phrase_lower = phrase.lower()
        if phrase_lower in context_lower and phrase_lower not in [c.lower() for c in concepts]:
            concepts.append(phrase)
            if len(concepts) >= 10:
                break
    
    # Limit to 8 most relevant concepts
    concepts = concepts[:8]
    
    return {
        "answer_summary": answer_summary,
        "key_points": key_points[:5],  # Limit to 5 key points
        "concepts_from_context": concepts
    }


def create_relevance_evaluator() -> Callable:
    """
    Create a relevance evaluator that assesses if the output is relevant to the input.

    Returns:
        Evaluator function that can be used with LangSmith
    """
    judge_llm = _create_judge_llm()
    if not judge_llm:
        return None

    # Define the evaluation prompt
    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert evaluator assessing whether outputs are relevant to the given input.

You MUST respond with valid JSON only. Do not include any conversational text before or after the JSON.

Your response must be a JSON object with exactly these fields:
- "comment": A detailed analysis that identifies any off-topic tangents or irrelevant information in the answer
- "answer_relevance": A string score from "1" to "10" where "10" means the output answer directly and effectively addresses the original input question, and "1" means it's not relevant at all.

Score from 1 to 10. 10 if the output answer directly and effectively addresses the original input question based on the specified criteria, 1 otherwise."""),
        ("human", """Evaluate the relevance of the output to the input.

<input>
{inputs}
</input>

<output>
{outputs}
</output>

<context>
{context}
</context>

{format_instructions}

IMPORTANT: Respond with ONLY valid JSON. No additional text before or after. The JSON must match this exact structure:
{{
  "comment": "your analysis here",
  "answer_relevance": "5"
}}""")
    ])

    # Create structured output schema
    from langchain_core.output_parsers import JsonOutputParser
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        # Fallback for older pydantic versions
        from pydantic.v1 import BaseModel, Field

    class RelevanceEvaluation(BaseModel):
        comment: str = Field(description="A detailed analysis that identifies any off-topic tangents or irrelevant information in the answer")
        answer_relevance: str = Field(description="Score from 1 to 10. 10 if the output answer directly and effectively addresses the original input question based on the specified criteria, 1 otherwise.", enum=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

    parser = JsonOutputParser(pydantic_object=RelevanceEvaluation)
    
    # Add parser instructions to the prompt
    formatted_prompt = relevance_prompt.partial(format_instructions=parser.get_format_instructions())
    
    # #region agent log
    import json
    try:
        with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"C","location":"langsmith_integration.py:113","message":"formatted_prompt created","data":{"prompt_type":str(type(formatted_prompt)),"prompt_template_messages":str(relevance_prompt.messages[:2]) if hasattr(relevance_prompt, 'messages') else "N/A"},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion

    # Create the evaluator chain
    # First get the LLM output, then parse it
    llm_chain = formatted_prompt | judge_llm
    
    def parse_with_fallback(llm_output):
        """Parse LLM output with fallback to manual parsing."""
        try:
            # Try the parser first
            return parser.parse(llm_output.content if hasattr(llm_output, 'content') else str(llm_output))
        except Exception as parse_err:
            # If parser fails, try manual JSON extraction
            output_str = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)
            print(f"⚠️  Parser failed, attempting manual extraction...")
            print(f"   Raw output: {output_str[:300]}...")
            
            # Try to find JSON in the output
            json_match = re.search(r'\{[^}]+\}', output_str, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, raise the original error
            raise parse_err
    
    # Chain: prompt -> LLM -> parse
    evaluator_chain = llm_chain | parse_with_fallback

    def relevance_evaluator(run, example=None) -> Dict[str, Any]:
        """
        Evaluate relevance of the output to the input.

        Args:
            run: LangSmith Run object
            example: Optional example with reference outputs

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Extract inputs and outputs from run
            inputs = run.inputs if hasattr(run, 'inputs') else {}
            outputs = run.outputs if hasattr(run, 'outputs') else {}

            # Format inputs for the evaluator
            question = inputs.get("question", inputs.get("inputs", ""))
            answer = outputs.get("output") or outputs.get("answer") or outputs.get("outputs", "")
            
            # Debug: Print what we're sending
            print(f"🔍 [DEBUG] Relevance evaluator inputs:")
            print(f"   question: {question[:100] if question else 'EMPTY'}...")
            print(f"   answer: {answer[:100] if answer else 'EMPTY'}...")
            print(f"   inputs dict: {inputs}")
            print(f"   outputs dict: {outputs}")
            
            # Validate that we have required inputs
            if not question or not answer:
                error_msg = f"Missing required data: question={bool(question)}, answer={bool(answer)}"
                print(f"⚠️  {error_msg}")
                return {
                    "key": "relevance_score",
                    "score": None,
                    "comment": error_msg,
                }
            
            eval_inputs = {
                "inputs": question,
                "outputs": answer,
                "context": inputs.get("context", ""),  # Always include context, even if empty
            }

            # Invoke the evaluator chain (prompt -> LLM -> parser)
            # The parser should automatically parse the LLM output to a dict
            try:
                result = evaluator_chain.invoke(eval_inputs)
            except Exception as parse_error:
                # If parsing fails, try to get raw LLM output and parse manually
                print(f"⚠️  Parser error, attempting manual parsing: {parse_error}")
                # Get the LLM output directly (without parser)
                llm_chain = formatted_prompt | judge_llm
                raw_output = llm_chain.invoke(eval_inputs)
                raw_str = raw_output.content if hasattr(raw_output, 'content') else str(raw_output)
                print(f"   Raw LLM output (first 500 chars): {raw_str[:500]}")
                print(f"   Full raw output length: {len(raw_str)}")
                
                # Try to extract JSON manually - use a better regex that handles nested braces
                # Look for JSON objects, handling nested braces properly
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_matches = re.findall(json_pattern, raw_str, re.DOTALL)
                
                # Also try to find JSON in code blocks
                code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                code_block_matches = re.findall(code_block_pattern, raw_str, re.DOTALL)
                
                # Try all found JSON candidates
                all_candidates = json_matches + code_block_matches
                
                for candidate in all_candidates:
                    try:
                        result = json.loads(candidate)
                        print(f"   ✅ Successfully parsed JSON manually from candidate")
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    # If no candidate worked, try to find the largest JSON-like structure
                    # Use a more sophisticated approach: find balanced braces
                    brace_count = 0
                    start_idx = raw_str.find('{')
                    if start_idx != -1:
                        for i in range(start_idx, len(raw_str)):
                            if raw_str[i] == '{':
                                brace_count += 1
                            elif raw_str[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    try:
                                        result = json.loads(raw_str[start_idx:i+1])
                                        print(f"   ✅ Successfully parsed JSON using balanced braces")
                                        break
                                    except json.JSONDecodeError:
                                        pass
                    else:
                        raise ValueError(f"No JSON found in LLM output. Raw output: {raw_str[:500]}")
            
            # Verify result is a dict (should be after parsing)
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict from parser, got {type(result)}: {result}")

            # Extract score
            score_str = result.get("answer_relevance", "5")
            try:
                score = float(score_str) / 10.0  # Normalize to 0-1
            except (ValueError, TypeError):
                score = 0.5

            return {
                "key": "relevance_score",
                "score": score,
                "comment": result.get("comment", ""),
                "raw_result": result,
            }
        except Exception as e:
            print(f"⚠️  Relevance evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "key": "relevance_score",
                "score": None,
                "comment": f"Evaluation error: {str(e)}",
            }

    return relevance_evaluator


def create_helpfulness_evaluator() -> Callable:
    """
    Create a helpfulness evaluator that assesses how helpful the output is.

    Returns:
        Evaluator function that can be used with LangSmith
    """
    judge_llm = _create_judge_llm()
    if not judge_llm:
        return None

    helpfulness_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert evaluator assessing how helpful an output is in addressing the user's question.

You MUST respond with valid JSON only. Do not include any conversational text before or after the JSON.

Your response must be a JSON object with exactly these fields:
- "comment": A detailed analysis of how helpful the answer is, identifying how well it addresses the question and noting any missing information
- "helpfulness_score": A string score from "1" to "10" where "10" means the output is extremely helpful and comprehensive, and "1" means it's not helpful at all.

Score from 1 to 10. 10 if the output is extremely helpful and comprehensive, 1 if it's not helpful at all."""),
        ("human", """Evaluate how helpful the output is in addressing the input.

<input>
{inputs}
</input>

<output>
{outputs}
</output>

<context>
{context}
</context>

{format_instructions}

IMPORTANT: Respond with ONLY valid JSON. No additional text before or after. The JSON must match this exact structure:
{{
  "comment": "your analysis here",
  "helpfulness_score": "5"
}}""")
    ])

    from langchain_core.output_parsers import JsonOutputParser
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        # Fallback for older pydantic versions
        from pydantic.v1 import BaseModel, Field

    class HelpfulnessEvaluation(BaseModel):
        comment: str = Field(description="A detailed analysis of how helpful the answer is")
        helpfulness_score: str = Field(description="Score from 1 to 10. 10 if the output is extremely helpful and comprehensive, 1 if it's not helpful at all.", enum=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

    parser = JsonOutputParser(pydantic_object=HelpfulnessEvaluation)
    
    # Add parser instructions to the prompt
    formatted_prompt = helpfulness_prompt.partial(format_instructions=parser.get_format_instructions())
    
    # Create the evaluator chain: prompt -> LLM -> parser
    evaluator_chain = formatted_prompt | judge_llm | parser

    def helpfulness_evaluator(run, example=None) -> Dict[str, Any]:
        """Evaluate helpfulness of the output."""
        try:
            inputs = run.inputs if hasattr(run, 'inputs') else {}
            outputs = run.outputs if hasattr(run, 'outputs') else {}

            # Format inputs for the evaluator
            question = inputs.get("question", inputs.get("inputs", ""))
            answer = outputs.get("output") or outputs.get("answer") or outputs.get("outputs", "")
            
            # Debug: Print what we're sending
            print(f"🔍 [DEBUG] Helpfulness evaluator inputs:")
            print(f"   question: {question[:100] if question else 'EMPTY'}...")
            print(f"   answer: {answer[:100] if answer else 'EMPTY'}...")
            
            # Validate that we have required inputs
            if not question or not answer:
                error_msg = f"Missing required data: question={bool(question)}, answer={bool(answer)}"
                print(f"⚠️  {error_msg}")
                return {
                    "key": "helpfulness_score",
                    "score": None,
                    "comment": error_msg,
                }
            
            eval_inputs = {
                "inputs": question,
                "outputs": answer,
                "context": inputs.get("context", ""),  # Always include context, even if empty
            }

            # Invoke the evaluator chain (prompt -> LLM -> parser)
            # The parser should automatically parse the LLM output to a dict
            try:
                result = evaluator_chain.invoke(eval_inputs)
            except Exception as parse_error:
                # If parsing fails, try to get raw LLM output and parse manually
                print(f"⚠️  Parser error, attempting manual parsing: {parse_error}")
                # Get the LLM output directly (without parser)
                llm_chain = formatted_prompt | judge_llm
                raw_output = llm_chain.invoke(eval_inputs)
                raw_str = raw_output.content if hasattr(raw_output, 'content') else str(raw_output)
                print(f"   Raw LLM output (first 500 chars): {raw_str[:500]}")
                print(f"   Full raw output length: {len(raw_str)}")
                
                # Try to extract JSON manually - use a better regex that handles nested braces
                # Look for JSON objects, handling nested braces properly
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_matches = re.findall(json_pattern, raw_str, re.DOTALL)
                
                # Also try to find JSON in code blocks
                code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                code_block_matches = re.findall(code_block_pattern, raw_str, re.DOTALL)
                
                # Try all found JSON candidates
                all_candidates = json_matches + code_block_matches
                
                for candidate in all_candidates:
                    try:
                        result = json.loads(candidate)
                        print(f"   ✅ Successfully parsed JSON manually from candidate")
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    # If no candidate worked, try to find the largest JSON-like structure
                    # Use a more sophisticated approach: find balanced braces
                    brace_count = 0
                    start_idx = raw_str.find('{')
                    if start_idx != -1:
                        for i in range(start_idx, len(raw_str)):
                            if raw_str[i] == '{':
                                brace_count += 1
                            elif raw_str[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    try:
                                        result = json.loads(raw_str[start_idx:i+1])
                                        print(f"   ✅ Successfully parsed JSON using balanced braces")
                                        break
                                    except json.JSONDecodeError:
                                        pass
                    else:
                        raise ValueError(f"No JSON found in LLM output. Raw output: {raw_str[:500]}")
            
            # Verify result is a dict (should be after parsing)
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict from parser, got {type(result)}: {result}")
            
            score_str = result.get("helpfulness_score", "5")
            try:
                score = float(score_str) / 10.0
            except (ValueError, TypeError):
                score = 0.5

            return {
                "key": "helpfulness_score",
                "score": score,
                "comment": result.get("comment", ""),
                "raw_result": result,
            }
        except Exception as e:
            print(f"⚠️  Helpfulness evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "key": "helpfulness_score",
                "score": None,
                "comment": f"Evaluation error: {str(e)}",
            }

    return helpfulness_evaluator


def get_all_evaluators(use_cache: bool = True) -> Dict[str, Callable]:
    """
    Get all available evaluators (cached for efficiency).

    Evaluators are automatically cached after first creation, so you don't need to
    recreate them on every evaluation call. Just call log_and_evaluate_rag_response()
    and it will use cached evaluators automatically.

    Args:
        use_cache: If True, use cached evaluators. If False, create new ones.

    Returns:
        Dictionary mapping evaluator names to evaluator functions
    """
    # #region agent log
    import json as _json
    try:
        with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as _f:
            _f.write(_json.dumps({"sessionId":"debug-session","runId":"current","hypothesisId":"G","location":"langsmith_integration.py:get_all_evaluators","message":"Called","data":{"use_cache":use_cache,"cache_keys":list(_evaluator_cache.keys()),"template_version":_template_version},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    # Check if cache is valid (has template version marker AND actual evaluators)
    cache_is_valid = use_cache and _evaluator_cache and "_version" in _evaluator_cache and _evaluator_cache["_version"] == _template_version
    cache_has_evaluators = cache_is_valid and any(k != "_version" for k in _evaluator_cache.keys())
    
    # #region agent log
    try:
        with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as _f:
            _f.write(_json.dumps({"sessionId":"debug-session","runId":"current","hypothesisId":"G","location":"langsmith_integration.py:get_all_evaluators","message":"Cache validity checked","data":{"cache_is_valid":cache_is_valid,"cache_has_evaluators":cache_has_evaluators,"has_version_key":"_version" in _evaluator_cache,"cache_version":_evaluator_cache.get("_version"),"expected_version":_template_version},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
    if cache_is_valid and cache_has_evaluators:
        # Return cached evaluators (excluding the version marker)
        result = {k: v for k, v in _evaluator_cache.items() if k != "_version"}
        # #region agent log
        try:
            with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as _f:
                _f.write(_json.dumps({"sessionId":"debug-session","runId":"current","hypothesisId":"G","location":"langsmith_integration.py:get_all_evaluators","message":"Returning cached evaluators","data":{"evaluator_names":list(result.keys())},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except: pass
        # #endregion
        return result

    # #region agent log
    try:
        with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as _f:
            _f.write(_json.dumps({"sessionId":"debug-session","runId":"current","hypothesisId":"G","location":"langsmith_integration.py:get_all_evaluators","message":"Creating new evaluators","data":{},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion

    evaluators = {}

    # Check cache first
    if cache_has_evaluators and "relevance" in _evaluator_cache:
        evaluators["relevance"] = _evaluator_cache["relevance"]
    else:
        # #region agent log
        try:
            with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as _f:
                _f.write(_json.dumps({"sessionId":"debug-session","runId":"current","hypothesisId":"G","location":"langsmith_integration.py:get_all_evaluators","message":"Calling create_relevance_evaluator","data":{},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except: pass
        # #endregion
        relevance_eval = create_relevance_evaluator()
        # #region agent log
        try:
            with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as _f:
                _f.write(_json.dumps({"sessionId":"debug-session","runId":"current","hypothesisId":"G","location":"langsmith_integration.py:get_all_evaluators","message":"create_relevance_evaluator returned","data":{"is_none":relevance_eval is None},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except: pass
        # #endregion
        if relevance_eval:
            evaluators["relevance"] = relevance_eval
            if use_cache:
                _evaluator_cache["relevance"] = relevance_eval

    if cache_is_valid and "helpfulness" in _evaluator_cache:
        evaluators["helpfulness"] = _evaluator_cache["helpfulness"]
    else:
        helpfulness_eval = create_helpfulness_evaluator()
        if helpfulness_eval:
            evaluators["helpfulness"] = helpfulness_eval
            if use_cache:
                _evaluator_cache["helpfulness"] = helpfulness_eval
    
    # Mark cache with version
    if use_cache:
        _evaluator_cache["_version"] = _template_version

    # #region agent log
    try:
        with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as _f:
            _f.write(_json.dumps({"sessionId":"debug-session","runId":"current","hypothesisId":"G","location":"langsmith_integration.py:get_all_evaluators","message":"Returning evaluators","data":{"evaluator_names":list(evaluators.keys()),"count":len(evaluators)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion

    return evaluators


def get_default_evaluators(use_cache: bool = True) -> list:
    """
    Get default list of evaluators to use (cached for efficiency).

    Args:
        use_cache: If True, use cached evaluators. If False, create new ones.

    Returns:
        List of evaluator functions
    """
    all_evals = get_all_evaluators(use_cache=use_cache)
    # Return relevance evaluator by default
    if "relevance" in all_evals:
        return [all_evals["relevance"]]
    return list(all_evals.values())


def clear_evaluator_cache():
    """Clear the evaluator cache (useful for testing or reconfiguration)."""
    global _evaluator_cache
    _evaluator_cache.clear()
    print("✅ Evaluator cache cleared")


def push_evaluators_to_dashboard(
    evaluator_names: Optional[list] = None,
    prompt_prefix: str = "eval_",
    include_model: bool = False,
) -> Dict[str, str]:
    """
    Push evaluator prompts to LangSmith dashboard so they appear as reusable evaluators.

    IMPORTANT: This pushes the PROMPT TEMPLATE only, not the model configuration.
    When you use these prompts in LangSmith UI, you'll need to:
    1. Select the prompt you pushed
    2. Configure the model (model name, API base, API key) in the LangSmith UI

    This is by design - model configuration and API keys should be set in LangSmith
    for security and flexibility, not hardcoded in the prompt.

    This allows you to use the evaluators in the LangSmith UI for:
    - Online evaluations (continuous evaluation on traces)
    - Dataset experiments
    - Manual evaluation runs

    After pushing, you can find these prompts in the LangSmith dashboard under Prompts,
    and use them to configure evaluators in the UI.

    Args:
        evaluator_names: List of evaluator names to push (e.g., ["relevance", "helpfulness"]).
                        If None, pushes all available evaluators.
        prompt_prefix: Prefix for prompt names in LangSmith (e.g., "eval_relevance_score")
        include_model: If True, includes model info in the push (but NOT API keys).
                      If False, pushes only the prompt template (recommended).

    Returns:
        Dictionary mapping evaluator names to their prompt IDs in LangSmith
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        # Fallback for older pydantic versions
        from pydantic.v1 import BaseModel, Field

    pushed_prompts = {}

    # Get evaluators to push
    if evaluator_names is None:
        evaluator_names = list(get_all_evaluators().keys())

    # Push relevance evaluator
    if "relevance" in evaluator_names:
        try:
            judge_llm = _create_judge_llm()
            if not judge_llm:
                print("⚠️  Cannot push relevance evaluator: judge LLM not available")
            else:
                # Create the prompt structure (same as in create_relevance_evaluator)
                relevance_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an expert evaluator assessing whether outputs are relevant to the given input.

A detailed analysis that: 1) Identifies any off-topic tangents or irrelevant information in the answer, and 2) Ends with "Thus, the score should be: X" where X reflects whether the output answer is relevant to the original input question.

Score from 1 to 10. 10 if the output answer directly and effectively addresses the original input question based on the specified criteria, 1 otherwise."""),
                    ("human", """<input>
{inputs}
</input>

<output>
{outputs}
</output>

<context>
{context}
</context>

{format_instructions}

Provide your evaluation in the required JSON format:""")
                ])

                # Create schema
                class RelevanceEvaluation(BaseModel):
                    comment: str = Field(description="A detailed analysis that identifies any off-topic tangents or irrelevant information in the answer")
                    answer_relevance: str = Field(description="Score from 1 to 10. 10 if the output answer directly and effectively addresses the original input question based on the specified criteria, 1 otherwise.", enum=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

                parser = JsonOutputParser(pydantic_object=RelevanceEvaluation)
                evaluator_chain = relevance_prompt | judge_llm | parser

                # Push to LangSmith
                prompt_name = f"{prompt_prefix}relevance_score"
                try:
                    _client.push_prompt(
                        prompt_name,
                        object=evaluator_chain,
                    )
                    pushed_prompts["relevance"] = prompt_name
                    print(f"✅ Pushed relevance evaluator as: {prompt_name}")
                    print(f"   You can now use this in the LangSmith dashboard!")
                except Exception as e:
                    print(f"⚠️  Failed to push relevance evaluator: {e}")
        except Exception as e:
            print(f"⚠️  Error pushing relevance evaluator: {e}")

    # Push helpfulness evaluator
    if "helpfulness" in evaluator_names:
        try:
            judge_llm = _create_judge_llm()
            if not judge_llm:
                print("⚠️  Cannot push helpfulness evaluator: judge LLM not available")
            else:
                helpfulness_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an expert evaluator assessing how helpful an output is in addressing the user's question.

A detailed analysis that: 1) Identifies how well the answer addresses the question, 2) Notes any missing information or areas for improvement, and 3) Ends with "Thus, the score should be: X" where X reflects how helpful the output is.

Score from 1 to 10. 10 if the output is extremely helpful and comprehensive, 1 if it's not helpful at all."""),
                    ("human", """<input>
{inputs}
</input>

<output>
{outputs}
</output>

<context>
{context}
</context>

{format_instructions}

Provide your evaluation in the required JSON format:""")
                ])

                class HelpfulnessEvaluation(BaseModel):
                    comment: str = Field(description="A detailed analysis of how helpful the answer is")
                    helpfulness_score: str = Field(description="Score from 1 to 10. 10 if the output is extremely helpful and comprehensive, 1 if it's not helpful at all.", enum=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

                parser = JsonOutputParser(pydantic_object=HelpfulnessEvaluation)
                evaluator_chain = helpfulness_prompt | judge_llm | parser

                prompt_name = f"{prompt_prefix}helpfulness_score"
                try:
                    _client.push_prompt(
                        prompt_name,
                        object=evaluator_chain,
                    )
                    pushed_prompts["helpfulness"] = prompt_name
                    print(f"✅ Pushed helpfulness evaluator as: {prompt_name}")
                    print(f"   You can now use this in the LangSmith dashboard!")
                except Exception as e:
                    print(f"⚠️  Failed to push helpfulness evaluator: {e}")
        except Exception as e:
            print(f"⚠️  Error pushing helpfulness evaluator: {e}")

    if pushed_prompts:
        print(f"\n📊 Summary: Pushed {len(pushed_prompts)} evaluator(s) to LangSmith")
        print("   To use them in the dashboard:")
        print("   1. Go to LangSmith Dashboard > Prompts")
        print("   2. Find your pushed prompts (e.g., 'eval_relevance_score')")
        print("   3. Use them to configure evaluators in Tracing Projects or Datasets")

    return pushed_prompts


def _extract_score_from_evaluation(evaluation_result: Any) -> Optional[float]:
    """Extract numeric score from evaluation result."""
    if evaluation_result is None:
        return None

    # If it's a string, try to extract a number
    if isinstance(evaluation_result, str):
        # Look for patterns like "Score: 4.5" or "4.5/5" or just "4.5"
        numbers = re.findall(r'\d+\.?\d*', evaluation_result)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass

    # If it's a dict, look for common score keys
    if isinstance(evaluation_result, dict):
        for key in ["score", "rating", "value", "relevance_score"]:
            if key in evaluation_result:
                val = evaluation_result[key]
                if isinstance(val, (int, float)):
                    return float(val)
                elif isinstance(val, str):
                    numbers = re.findall(r'\d+\.?\d*', val)
                    if numbers:
                        try:
                            return float(numbers[0])
                        except ValueError:
                            pass

    # If it's a number directly
    if isinstance(evaluation_result, (int, float)):
        return float(evaluation_result)

    return None


@traceable(
    run_type="chain",
    name="rag_response_for_eval",
    process_inputs=lambda data: {
        "question": data.get("question"),
        "context": data.get("context"),
    },
    process_outputs=lambda output: {
        "answer": output.get("answer"),
        "evaluation": output.get("evaluation"),
        "scores": output.get("scores"),
    } if isinstance(output, dict) else output,
)
def log_and_evaluate_rag_response(
    question: str,
    context: str,
    answer: str,
    reference_outputs: Optional[Dict[str, Any]] = None,
    evaluators: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Log inputs/outputs to LangSmith and evaluate using custom evaluator objects.
    Creates feedback with score for dashboard display.

    Evaluators are automatically cached, so you don't need to call create_relevance_evaluator()
    every time. Just call this function and it will use cached evaluators automatically.

    Args:
        question: User's question
        context: PDF context used for answering
        answer: Generated answer
        reference_outputs: Optional reference outputs for comparison
        evaluators: Optional list of evaluator functions. If None, uses default cached relevance evaluator.

    Returns:
        Dictionary with answer, evaluation results, and scores
    """
    # Get current run tree for evaluation
    run_tree = get_current_run_tree()

    if not run_tree or not run_tree.id:
        print("⚠️  No active run tree found. Evaluation skipped.")
        return {
            "answer": answer,
            "evaluation": None,
        }

    # Parse answer for Opik-compatible structured trace
    opik_trace = _parse_answer_for_opik(answer, context)
    
    # Create a mock run object for evaluators
    class MockRun:
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs
            self.id = run_tree.id

    mock_run = MockRun(
        inputs={"question": question, "context": context},
        outputs={
            "output": answer,  # LangSmith-compatible format
            "opik_trace": opik_trace  # Opik-compatible structured trace
        }
    )

    # Use provided evaluators or default evaluators (cached)
    if evaluators is None:
        # #region agent log
        import json as _json
        try:
            with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as _f:
                _f.write(_json.dumps({"sessionId":"debug-session","runId":"current","hypothesisId":"G","location":"langsmith_integration.py:log_and_evaluate","message":"Calling get_default_evaluators","data":{},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except: pass
        # #endregion
        evaluators = get_default_evaluators(use_cache=True)
        # #region agent log
        try:
            with open(r'c:\Users\EthanYongYuHeng\Desktop\langsmith-protoype\.cursor\debug.log', 'a', encoding='utf-8') as _f:
                _f.write(_json.dumps({"sessionId":"debug-session","runId":"current","hypothesisId":"G","location":"langsmith_integration.py:log_and_evaluate","message":"get_default_evaluators returned","data":{"evaluators_count":len(evaluators) if evaluators else 0,"is_none":evaluators is None,"is_empty":not evaluators},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except: pass
        # #endregion
        if not evaluators:
            print("⚠️  Could not create evaluators. Evaluation skipped.")
            return {
                "answer": answer,
                "evaluation": None,
            }

    # Run all evaluators
    evaluation_results = {}
    all_scores = {}

    for evaluator in evaluators:
        try:
            result = evaluator(mock_run, example=None)
            if result:
                key = result.get("key", "evaluation")
                evaluation_results[key] = result

                # Create feedback in LangSmith
                score = result.get("score")
                comment = result.get("comment", "")

                if score is not None:
                    all_scores[key] = score
                    try:
                        _client.create_feedback(
                            run_id=run_tree.id,
                            key=key,
                            score=score,
                            comment=comment if comment else None,
                        )
                    except Exception as e:
                        print(f"⚠️  Failed to create feedback for {key}: {e}")
        except Exception as e:
            print(f"⚠️  Evaluator {evaluator.__name__ if hasattr(evaluator, '__name__') else 'unknown'} failed: {e}")

    payload: Dict[str, Any] = {
        "answer": answer,  # Backward compatibility
        "output": answer,  # LangSmith format
        "opik_trace": opik_trace,  # Opik structured format
        "evaluation": evaluation_results,
    }
    if all_scores:
        payload["scores"] = all_scores
    if reference_outputs is not None:
        payload["reference_outputs"] = reference_outputs

    return payload


def evaluate_query_and_output(
    question: str,
    context: str,
    answer: str,
    run_id: Optional[str] = None,
    project_name: Optional[str] = None,
    evaluators: Optional[list] = None,
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a query and output pair, optionally linking to an existing run.

    This function can be used to evaluate queries and outputs later, even if
    there's no active trace. It will create a run in LangSmith if run_id is not provided.

    Args:
        question: User's question
        context: PDF context used for answering
        answer: Generated answer
        run_id: Optional existing run ID to attach feedback to. If None, creates a new run.
        project_name: Optional project name for the run (uses LANGSMITH_PROJECT env var if not provided)
        evaluators: Optional list of evaluator functions. If None, uses default cached relevance evaluator.
        reference_outputs: Optional reference outputs for comparison

    Returns:
        Dictionary with evaluation results, scores, and run_id
    """
    from datetime import datetime

    # Use provided evaluators or default evaluators (cached)
    if evaluators is None:
        evaluators = get_default_evaluators(use_cache=True)
        if not evaluators:
            print("⚠️  Could not create evaluators. Evaluation skipped.")
            return {
                "answer": answer,
                "evaluation": None,
            }

    # Create or use existing run
    if run_id is None:
        # Create a new run for this evaluation
        project = project_name or os.getenv("LANGSMITH_PROJECT", "rag-evaluation")

        try:
            # Create a run using RunTree
            run_tree = RunTree(
                name="rag_qa",
                run_type="chain",
                inputs={"question": question, "context": context},
                outputs={"answer": answer},
                project_name=project,
            )
            run_tree.end()
            run_id = run_tree.id
            print(f"✅ Created new run: {run_id}")
        except Exception as e:
            print(f"⚠️  Failed to create run: {e}")
            print("   Continuing with evaluation (results won't be linked to a run)")
            # Continue with evaluation even if run creation fails
            run_id = None
    else:
        print(f"✅ Using existing run: {run_id}")

    # Create a mock run object for evaluators
    class MockRun:
        def __init__(self, inputs, outputs, run_id):
            self.inputs = inputs
            self.outputs = outputs
            self.id = run_id

    mock_run = MockRun(
        inputs={"question": question, "context": context},
        outputs={"answer": answer},
        run_id=run_id or "no-run-id"
    )

    # Run all evaluators
    evaluation_results = {}
    all_scores = {}

    for evaluator in evaluators:
        try:
            result = evaluator(mock_run, example=None)
            if result:
                key = result.get("key", "evaluation")
                evaluation_results[key] = result

                # Create feedback in LangSmith if we have a run_id
                score = result.get("score")
                comment = result.get("comment", "")

                if score is not None:
                    all_scores[key] = score
                    if run_id:
                        try:
                            _client.create_feedback(
                                run_id=run_id,
                                key=key,
                                score=score,
                                comment=comment if comment else None,
                            )
                        except Exception as e:
                            print(f"⚠️  Failed to create feedback for {key}: {e}")
        except Exception as e:
            print(f"⚠️  Evaluator {evaluator.__name__ if hasattr(evaluator, '__name__') else 'unknown'} failed: {e}")

    payload: Dict[str, Any] = {
        "answer": answer,
        "evaluation": evaluation_results,
        "run_id": run_id,
    }
    if all_scores:
        payload["scores"] = all_scores
    if reference_outputs is not None:
        payload["reference_outputs"] = reference_outputs

    return payload


def evaluate_existing_runs(
    run_ids: list,
    evaluators: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Evaluate existing runs from LangSmith by their IDs.

    This is useful for evaluating runs that were created earlier.

    Args:
        run_ids: List of run IDs to evaluate
        evaluators: Optional list of evaluator functions. If None, uses default cached relevance evaluator.

    Returns:
        Dictionary mapping run_id to evaluation results
    """
    # Use provided evaluators or default evaluators (cached)
    if evaluators is None:
        evaluators = get_default_evaluators(use_cache=True)
        if not evaluators:
            print("⚠️  Could not create evaluators. Evaluation skipped.")
            return {}

    results = {}

    for run_id in run_ids:
        try:
            # Fetch the run from LangSmith
            run = _client.read_run(run_id)

            # Extract inputs and outputs
            inputs = run.inputs if hasattr(run, 'inputs') else {}
            outputs = run.outputs if hasattr(run, 'outputs') else {}

            # Create a mock run object for evaluators
            class MockRun:
                def __init__(self, inputs, outputs, run_id):
                    self.inputs = inputs
                    self.outputs = outputs
                    self.id = run_id

            mock_run = MockRun(inputs=inputs, outputs=outputs, run_id=run_id)

            # Run all evaluators
            evaluation_results = {}
            all_scores = {}

            for evaluator in evaluators:
                try:
                    result = evaluator(mock_run, example=None)
                    if result:
                        key = result.get("key", "evaluation")
                        evaluation_results[key] = result

                        # Create feedback in LangSmith
                        score = result.get("score")
                        comment = result.get("comment", "")

                        if score is not None:
                            all_scores[key] = score
                            try:
                                _client.create_feedback(
                                    run_id=run_id,
                                    key=key,
                                    score=score,
                                    comment=comment if comment else None,
                                )
                            except Exception as e:
                                print(f"⚠️  Failed to create feedback for {key} on run {run_id}: {e}")
                except Exception as e:
                    print(f"⚠️  Evaluator failed for run {run_id}: {e}")

            results[run_id] = {
                "evaluation": evaluation_results,
                "scores": all_scores,
            }

        except Exception as e:
            print(f"⚠️  Failed to evaluate run {run_id}: {e}")
            results[run_id] = {"error": str(e)}

    return results


# ============================================================================
# New LangSmith evaluate() API Compatible Evaluators
# ============================================================================
# These evaluators work with the new LangSmith evaluate() function
# They use the wrapped OpenAI client pattern and support both sync and async

# Create wrapped OpenAI client for evaluators (using async for better performance)
_async_judge_client = None

def _get_async_judge_client():
    """Get or create the async wrapped OpenAI client for evaluators."""
    global _async_judge_client
    if _async_judge_client is None:
        if JUDGE_API_BASE:
            openai_client = AsyncOpenAI(
                base_url=JUDGE_API_BASE,
                api_key=JUDGE_API_KEY or "not-needed",
            )
        else:
            openai_client = AsyncOpenAI(
                api_key=JUDGE_API_KEY,
            )
        _async_judge_client = wrappers.wrap_openai(openai_client)
    return _async_judge_client


# Pydantic models for structured output
class RelevanceEvaluation(BaseModel):
    """Structured output for relevance evaluation."""
    comment: str = Field(description="A detailed analysis that identifies any off-topic tangents or irrelevant information in the answer")
    answer_relevance: int = Field(description="Score from 1 to 10. 10 if the output answer directly and effectively addresses the original input question, 1 otherwise.", ge=1, le=10)


class HelpfulnessEvaluation(BaseModel):
    """Structured output for helpfulness evaluation."""
    comment: str = Field(description="A detailed analysis of how helpful the answer is")
    helpfulness_score: int = Field(description="Score from 1 to 10. 10 if the output is extremely helpful and comprehensive, 1 if it's not helpful at all.", ge=1, le=10)


async def relevance_evaluator(inputs: dict, outputs: dict) -> dict:
    """
    Evaluate relevance of the output to the input using LLM-as-judge.
    
    Compatible with LangSmith's evaluate() API.
    
    Args:
        inputs: Dictionary containing input data (e.g., {"question": "...", "context": "..."})
        outputs: Dictionary containing output data (e.g., {"answer": "..."})
    
    Returns:
        Dictionary with evaluation results including key, score (0-1), and comment
    """
    try:
        oai_client = _get_async_judge_client()
        
        # Extract question and answer
        question = inputs.get("question", inputs.get("inputs", ""))
        answer = outputs.get("answer", outputs.get("outputs", ""))
        context = inputs.get("context", "")
        
        # Build the evaluation prompt
        instructions = """You are an expert evaluator assessing whether outputs are relevant to the given input.

A detailed analysis that: 1) Identifies any off-topic tangents or irrelevant information in the answer, and 2) Provides a score from 1 to 10 where 10 means the output answer directly and effectively addresses the original input question, and 1 means it's not relevant at all."""

        # Build message content
        msg_parts = [f"Question: {question}", f"Answer: {answer}"]
        if context:
            msg_parts.insert(1, f"Context: {context}")
        msg = "\n".join(msg_parts)
        
        # Use structured output parsing
        response = await oai_client.beta.chat.completions.parse(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": msg}
            ],
            response_format=RelevanceEvaluation
        )
        
        parsed = response.choices[0].message.parsed
        score = parsed.answer_relevance / 10.0  # Normalize to 0-1
        
        return {
            "key": "relevance_score",
            "score": score,
            "comment": parsed.comment,
        }
    except Exception as e:
        print(f"⚠️  Relevance evaluation failed: {e}")
        return {
            "key": "relevance_score",
            "score": None,
            "comment": f"Evaluation error: {str(e)}",
        }


async def helpfulness_evaluator(inputs: dict, outputs: dict) -> dict:
    """
    Evaluate helpfulness of the output using LLM-as-judge.
    
    Compatible with LangSmith's evaluate() API.
    
    Args:
        inputs: Dictionary containing input data (e.g., {"question": "...", "context": "..."})
        outputs: Dictionary containing output data (e.g., {"answer": "..."})
    
    Returns:
        Dictionary with evaluation results including key, score (0-1), and comment
    """
    try:
        oai_client = _get_async_judge_client()
        
        # Extract question and answer
        question = inputs.get("question", inputs.get("inputs", ""))
        answer = outputs.get("answer", outputs.get("outputs", ""))
        context = inputs.get("context", "")
        
        # Build the evaluation prompt
        instructions = """You are an expert evaluator assessing how helpful an output is in addressing the user's question.

A detailed analysis that: 1) Identifies how well the answer addresses the question, 2) Notes any missing information or areas for improvement, and 3) Provides a score from 1 to 10 where 10 means the output is extremely helpful and comprehensive, and 1 means it's not helpful at all."""

        # Build message content
        msg_parts = [f"Question: {question}", f"Answer: {answer}"]
        if context:
            msg_parts.insert(1, f"Context: {context}")
        msg = "\n".join(msg_parts)
        
        # Use structured output parsing
        response = await oai_client.beta.chat.completions.parse(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": msg}
            ],
            response_format=HelpfulnessEvaluation
        )
        
        parsed = response.choices[0].message.parsed
        score = parsed.helpfulness_score / 10.0  # Normalize to 0-1
        
        return {
            "key": "helpfulness_score",
            "score": score,
            "comment": parsed.comment,
        }
    except Exception as e:
        print(f"⚠️  Helpfulness evaluation failed: {e}")
        return {
            "key": "helpfulness_score",
            "score": None,
            "comment": f"Evaluation error: {str(e)}",
        }


def run_evaluation_with_dataset(
    target_function: Callable,
    dataset_name: str,
    evaluators: Optional[list] = None,
    **evaluate_kwargs
) -> Any:
    """
    Run evaluation using LangSmith's evaluate() function with your dataset.
    
    This is a convenience wrapper that uses your relevance and helpfulness evaluators
    with the new LangSmith evaluate() API.
    
    Args:
        target_function: Your application function that takes inputs dict and returns outputs dict
        dataset_name: Name of the dataset in LangSmith (or dataset ID)
        evaluators: Optional list of evaluator functions. If None, uses relevance and helpfulness
        **evaluate_kwargs: Additional arguments to pass to evaluate() (e.g., max_concurrency, tags)
    
    Returns:
        Evaluation results from LangSmith
    
    Example:
        ```python
        def my_app(inputs: dict) -> dict:
            # Your application logic
            return {"answer": "..."}
        
        results = run_evaluation_with_dataset(
            my_app,
            dataset_name="my-rag-dataset",
        )
        ```
    """
    if evaluators is None:
        evaluators = [relevance_evaluator, helpfulness_evaluator]
    
    return evaluate(
        target_function,
        data=dataset_name,
        evaluators=evaluators,
        **evaluate_kwargs
    )