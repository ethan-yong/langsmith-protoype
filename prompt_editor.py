"""
Utility for editing LangSmith prompts interactively.
"""
import json
import os
from typing import Optional

from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from .env file
load_dotenv()


def edit_prompt(prompt_name: str, push_name: Optional[str] = None) -> None:
    """
    Pull a prompt from LangSmith, display it, allow editing, and push it back.
    
    Args:
        prompt_name: Name/ID of the prompt to pull from LangSmith
        push_name: Optional name to push the edited prompt as (defaults to prompt_name)
    """
    client = Client()
    
    # Always use owner from environment variable
    owner = os.getenv("LANGSMITH_OWNER")
    
    if not owner:
        print("⚠️  Warning: LANGSMITH_OWNER not set in environment. Private prompts may fail.")
    
    # Pull the prompt
    print(f"\n📥 Pulling prompt: {prompt_name}" + (f" (owner: {owner})" if owner else " (no owner specified)"))
    try:
        pull_kwargs = {"include_model": True}
        if owner:
            pull_kwargs["owner"] = owner
        prompt = client.pull_prompt(prompt_name, **pull_kwargs)
    except Exception as e:
        print(f"❌ Failed to pull prompt: {e}")
        return
    
    # Extract and display schema if available
    schema = None
    try:
        # Try to get schema from prompt
        if hasattr(prompt, 'schema_'):
            schema = prompt.schema_
        elif hasattr(prompt, 'model_dump'):
            prompt_dict = prompt.model_dump()
            if "first" in prompt_dict and "schema_" in prompt_dict["first"]:
                schema = prompt_dict["first"]["schema_"]
        elif hasattr(prompt, 'dict'):
            prompt_dict = prompt.dict()
            if "first" in prompt_dict and "schema_" in prompt_dict["first"]:
                schema = prompt_dict["first"]["schema_"]
        
        if schema:
            print("\n" + "="*60)
            print("📋 PROMPT SCHEMA:")
            print("="*60)
            print(json.dumps(schema, indent=2))
            print("="*60)
            
            # Ask if user wants to modify schema (optional, one-time)
            modify_schema = input("\n✏️  Modify schema? (yes/no, default: no): ").strip().lower()
            if modify_schema in ['yes', 'y']:
                # Update schema to use string type with enum 1-10
                if isinstance(schema, dict):
                    # Find the score/relevance field and update it
                    if "properties" in schema:
                        updated = False
                        for prop_name, prop_def in schema["properties"].items():
                            # Check for answer_relevance, score, relevance_score, rating, value fields
                            if (prop_name in ["answer_relevance", "score", "relevance_score", "rating", "value"] or 
                                "score" in prop_name.lower() or "relevance" in prop_name.lower()):
                                prop_def["type"] = "string"
                                prop_def["enum"] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                                prop_def["description"] = "Score from 1 to 10. 10 if the output answer directly and effectively addresses the original input question based on the specified criteria, 1 otherwise."
                                updated = True
                                break
                        # If no score/relevance field found, add one
                        if not updated:
                            schema["properties"]["answer_relevance"] = {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                                "description": "Score from 1 to 10. 10 if the output answer directly and effectively addresses the original input question based on the specified criteria, 1 otherwise."
                            }
                            # Update required fields if answer_relevance was required
                            if "required" in schema and "answer_relevance" not in schema["required"]:
                                schema["required"].append("answer_relevance")
                    else:
                        # If schema is flat, update type directly
                        schema["type"] = "string"
                        schema["enum"] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                        if "description" not in schema:
                            schema["description"] = "Score from 1 to 10. 10 if the output answer directly and effectively addresses the original input question based on the specified criteria, 1 otherwise."
                    
                    print("\n✅ Schema updated to use string type with enum 1-10")
                    print("\n" + "="*60)
                    print("📋 UPDATED SCHEMA:")
                    print("="*60)
                    print(json.dumps(schema, indent=2))
                    print("="*60)
    except Exception as e:
        print(f"⚠️  Could not extract/modify schema: {e}")
        import traceback
        traceback.print_exc()
    
    # Display current prompt
    print("\n" + "="*60)
    print("📝 CURRENT PROMPT:")
    print("="*60)
    
    # Try to get the prompt text from different possible structures
    prompt_text = None
    
    # Try nested structure: first.messages[0].prompt.template
    try:
        if hasattr(prompt, 'model_dump'):
            prompt_dict = prompt.model_dump()
        elif hasattr(prompt, 'dict'):
            prompt_dict = prompt.dict()
        else:
            prompt_dict = {}
        
        if "first" in prompt_dict:
            first = prompt_dict["first"]
            if "messages" in first and isinstance(first["messages"], list) and len(first["messages"]) > 0:
                msg = first["messages"][0]
                if isinstance(msg, dict) and "prompt" in msg:
                    prompt_obj = msg["prompt"]
                    if isinstance(prompt_obj, dict) and "template" in prompt_obj:
                        prompt_text = prompt_obj["template"]
    except Exception:
        pass
    
    # Fallback to other structures
    if not prompt_text:
        if hasattr(prompt, 'messages'):
            # If it's a chat prompt with messages
            for msg in prompt.messages:
                if hasattr(msg, 'content'):
                    prompt_text = msg.content
                    break
                elif isinstance(msg, dict) and 'content' in msg:
                    prompt_text = msg['content']
                    break
                elif isinstance(msg, dict) and 'prompt' in msg:
                    prompt_obj = msg['prompt']
                    if isinstance(prompt_obj, dict) and 'template' in prompt_obj:
                        prompt_text = prompt_obj['template']
                        break
        elif hasattr(prompt, 'template'):
            prompt_text = prompt.template
        elif hasattr(prompt, 'prompt'):
            prompt_text = prompt.prompt
        elif isinstance(prompt, str):
            prompt_text = prompt
    
    if prompt_text:
        print(prompt_text)
    else:
        print("⚠️  Could not extract prompt text. Full prompt object:")
        print(prompt)
    
    print("="*60)
    
    # Get edited prompt from user
    print("\n✏️  Enter your edited prompt (press Enter twice to finish, or 'cancel' to abort):")
    lines = []
    while True:
        try:
            line = input()
            if line.lower() == 'cancel':
                print("❌ Edit cancelled.")
                return
            if line == '' and lines and lines[-1] == '':
                # Two consecutive empty lines = done
                break
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Edit cancelled.")
            return
    
    # Join lines (remove the last empty line)
    edited_prompt = '\n'.join(lines[:-1]) if lines and lines[-1] == '' else '\n'.join(lines)
    
    if not edited_prompt.strip():
        print("❌ Empty prompt. Aborting.")
        return
    
    # Confirm before pushing
    print("\n" + "="*60)
    print("📤 EDITED PROMPT:")
    print("="*60)
    print(edited_prompt)
    print("="*60)
    
    confirm = input(f"\n✅ Push this prompt as '{push_name or prompt_name}'? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("❌ Push cancelled.")
        return
    
    # Push the prompt
    print(f"\n📤 Pushing prompt: {push_name or prompt_name}")
    try:
        from langchain_core.prompts import PromptTemplate
        
        # Get input variables and structure from original prompt
        input_variables = ["inputs", "outputs"]  # Default
        try:
            if hasattr(prompt, 'model_dump'):
                prompt_dict = prompt.model_dump()
            elif hasattr(prompt, 'dict'):
                prompt_dict = prompt.dict()
            else:
                prompt_dict = {}
            
            if "first" in prompt_dict:
                if "input_variables" in prompt_dict["first"]:
                    input_variables = prompt_dict["first"]["input_variables"]
        except Exception:
            pass
        
        # Create prompt template with mustache format
        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=edited_prompt,
            template_format="mustache"
        )
        
        # Try to update the original prompt structure in-place
        new_prompt = prompt
        try:
            # Access first.messages[0].prompt and update template
            if hasattr(prompt, 'first'):
                first = prompt.first
                if hasattr(first, 'messages') and len(first.messages) > 0:
                    msg = first.messages[0]
                    # Update the entire prompt object, not just template
                    if hasattr(msg, 'prompt'):
                        # Replace the prompt object entirely
                        msg.prompt = prompt_template
                    # If schema was modified, update it
                    if schema and hasattr(first, 'schema_'):
                        first.schema_ = schema
                # Also ensure input_variables are set on first
                if hasattr(first, 'input_variables'):
                    first.input_variables = input_variables
        except Exception as e:
            print(f"⚠️  Could not update original structure: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use original prompt
            new_prompt = prompt
        
        # Push to LangSmith (will use the modified prompt)
        client.push_prompt(
            push_name or prompt_name,
            object=new_prompt,
        )
        print(f"✅ Successfully pushed prompt: {push_name or prompt_name}")
        if schema:
            print("   (Schema modifications included)")
    except Exception as e:
        print(f"❌ Failed to push prompt: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prompt_editor.py <prompt_name> [push_name]")
        print("\nExample:")
        print("  python prompt_editor.py eval_default_more_relevant_score_dcab1c10:8eb5c6eb")
        print("  python prompt_editor.py eval_default_more_relevant_score_dcab1c10:8eb5c6eb my_edited_prompt")
        print("\nNote: Uses LANGSMITH_OWNER from environment variable for private prompts")
        sys.exit(1)
    
    prompt_name = sys.argv[1]
    push_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    edit_prompt(prompt_name, push_name)
