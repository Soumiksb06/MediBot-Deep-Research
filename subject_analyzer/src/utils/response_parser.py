# src/utils/response_parser.py
import json
from typing import Any
from rich.console import Console

class ResponseParser:
    """Utility class for parsing LLM responses."""

    def __init__(self):
        """Initialize the response parser."""
        self.console = Console()

    def extract_json(self, response: Any) -> Any:
        """
        Extract and parse a JSON object from an LLM response.
        
        Assumes the JSON is enclosed between the first '{' and the last '}' in the content.
        """
        try:
            # Support both dict-style and attribute-style access
            if isinstance(response, dict):
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                # Fallback if response is not a dict (unlikely)
                content = str(response)
            self.console.print(f"[yellow]Raw response content:[/yellow] {content[:200]}...")
            start = content.find('{')
            end = content.rfind('}')
            if start == -1 or end == -1:
                self.console.print("[red]No JSON object found in response[/red]")
                raise ValueError("No JSON object found in response")
            json_str = content[start:end+1]
            return json.loads(json_str)
        except Exception as e:
            self.console.print(f"[red]Error extracting JSON: {str(e)}[/red]")
            raise

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
            # Try both dictionary and attribute access (depending on your LLM client)
            if isinstance(response, dict):
                return response["choices"][0]["message"]["content"]
            else:
                return response.choices[0].message.content
        except (KeyError, IndexError, AttributeError) as e:
            self.console.print("[red]Cannot extract content from LLM response[/red]")
            raise ValueError("Cannot extract content from LLM response") from e
