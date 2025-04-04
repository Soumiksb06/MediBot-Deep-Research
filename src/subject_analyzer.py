from typing import Dict
import json
from rich.console import Console

def _extract_llm_response(self, response: Dict) -> str:
    """Safely extract content from LLM response"""
    try:
        # Get the raw content first
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
        elif isinstance(response, dict):
            if 'choices' in response and response['choices']:
                content = response['choices'][0]['message']['content']
        else:
            raise ValueError("Invalid response format from LLM")

        # Debugging: Print the raw content
        self.console.print(f"[yellow]Raw response content:[/yellow] {content}")

        # Clean the content - remove any non-JSON text
        start = content.find('{')
        end = content.rfind('}')
        
        if start == -1 or end == -1:
            self.console.print("[red]No JSON object found in response[/red]")
            raise ValueError("No JSON object found in response")
        
        json_str = content[start:end+1]

        # Validate that it's proper JSON
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            self.console.print("[red]Invalid JSON in response[/red]")
            self.console.print(f"Extracted JSON string: {json_str}")
            raise e

    except Exception as e:
        self.console.print(f"[red]Error extracting LLM response: {str(e)}[/red]")
        self.console.print(f"Response structure: {response}")
        raise 