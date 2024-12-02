from openai_utils import get_completion
import json

FIX_JSON_PROMPT = """
Fix the following JSON format and return the correct JSON format. Do not output anything else:

{{json}}    
""".strip()


class JSONParser:
    """Utility class for extracting and fixing JSON from text."""

    @staticmethod
    def _extract_json(text: str, start_char: str, end_char: str) -> str:
        """Extract JSON from text, inclusive of start and end characters."""
        start_index = text.find(start_char)
        end_index = text.rfind(end_char)
        return text[start_index : end_index + 1]

    @staticmethod
    def fix_json(json_str: str) -> str:
        """Use an LLM to fix broken JSON."""
        prompt = FIX_JSON_PROMPT.replace("{{json}}", json_str)
        return get_completion(prompt)

    @staticmethod
    def extract_json_list(text: str, use_llm_to_fix: bool = False) -> list:
        """Extract a JSON list from text."""
        try:
            json_list_str = JSONParser._extract_json(text, "[", "]")
            return json.loads(json_list_str)
        except json.JSONDecodeError:
            if use_llm_to_fix:
                fixed_json_str = JSONParser.fix_json(text)
                return JSONParser.extract_json_list(
                    fixed_json_str, use_llm_to_fix=False
                )
            return None

    @staticmethod
    def extract_json_dict(text: str, use_llm_to_fix: bool = False) -> dict:
        """Extract a JSON dict from text."""
        try:
            json_dict_str = JSONParser._extract_json(text, "{", "}")
            return json.loads(json_dict_str)
        except json.JSONDecodeError:
            if use_llm_to_fix:
                fixed_json_str = JSONParser.fix_json(text)
                return JSONParser.extract_json_dict(
                    fixed_json_str, use_llm_to_fix=False
                )
            return None
