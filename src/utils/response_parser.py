# src/utils/response_parser.py
import json
from typing import Any

class ResponseParser:
    """Utility class for parsing LLM responses."""

    def extract_json(self, response: Any) -> str:
        """
        Extract a JSON string from an LLM response.
        Assumes the JSON is enclosed between the first '{' and the last '}'.
        """
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        start = content.find('{')
        end = content.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")
        json_str = content[start:end+1]
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON extracted from response") from e

    def extract_content(self, response: Any) -> str:
        """
        Extract and return the content string from an LLM response.
        Expected structure:
        {
            "choices": [
                {
                    "message": {
                        "content": "Generated text..."
                    }
                }
            ]
        }
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError("Cannot extract content from LLM response") from e
