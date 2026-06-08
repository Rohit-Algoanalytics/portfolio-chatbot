import json
import re

def clean_json_response(response_text: str) -> dict:
    """
    Cleans an LLM JSON response by removing code fences and stray text.
    Returns a parsed JSON dictionary.
    """
    # Remove markdown code fences like ```json ... ``` or ``` ... ```
    cleaned_text = re.sub(r"^```(?:json)?|```$", "", response_text.strip(), flags=re.MULTILINE).strip()

    # Try parsing the cleaned text
    try:
        return cleaned_text
    except json.JSONDecodeError:
        # If still invalid, try to extract JSON substring
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return {"error": "Invalid JSON format", "raw_output": response_text}
