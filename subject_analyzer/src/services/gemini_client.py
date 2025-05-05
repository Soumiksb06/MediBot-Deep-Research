"""Gemini client implementation using google-genai library"""

from typing import Dict, List
import os
# Import the necessary components from the google-genai library
from google import genai
from google.genai import types
from rich.console import Console

from ..interfaces.llm_interface import LLMInterface
from ..models.analysis_models import AnalysisConfig

class GeminiClient(LLMInterface):
    """Client for Gemini API using google-genai"""

    def __init__(self, api_key: str, config: AnalysisConfig):
        # The google-genai client handles the base URL internally
        self.client = genai.Client(api_key=api_key)
        self.config = config
        self.console = Console()

    def chat(self, messages: List[Dict[str, str]], temperature: float = None) -> Dict:
        try:
            self.console.print("\n=== Gemini API Debug Info ===")
            for msg in messages:
                self.console.print(f"- Role: {msg['role']}")
                self.console.print(f"  Content: {msg['content'][:100]}...")

            # Convert the input messages list (dicts) to the google-genai format (list of types.Content)
            gemini_contents = []
            for msg in messages:
                # Map the roles. Gemini typically uses 'user' and 'model'.
                # Assuming 'system' messages are treated as 'user' input for now
                role = 'user' if msg['role'] in ['user', 'system'] else 'model'
                gemini_contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg['content'])],
                    )
                )

            # Create a GenerationConfig object
            generation_config = types.GenerateContentConfig(
                temperature=temperature or self.config.temperature,
                # Include other generation parameters from AnalysisConfig if applicable
                # e.g., top_p=self.config.top_p, top_k=self.config.top_k
                # Explicitly add automatic_function_calling to prevent AttributeError
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            # Use the generate_content method and pass the config object using the 'config' argument name
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=gemini_contents,
                config=generation_config, # Pass the config object using 'config'
            )

            self.console.print("\nResponse received from Gemini:")
            self.console.print(f"Response type: {type(response)}")

            # Extract the text from the response.
            generated_text = response.text if hasattr(response, 'text') else str(response)

            self.console.print(f"Generated content: {generated_text[:100]}...")

            # Format the response to match the expected dictionary structure
            formatted_response = {
                "choices": [
                    {
                        "message": {
                            "content": generated_text
                        }
                    }
                ]
            }

            return formatted_response

        except Exception as e:
            self.console.print(f"[red]Error in Gemini API call:[/red]")
            self.console.print(f"[red]Error type: {type(e)}[/red]")
            self.console.print(f"[red]Error message: {str(e)}[/red]")
            import traceback
            self.console.print("[red]Traceback:[/red]")
            self.console.print(traceback.format_exc())
            raise