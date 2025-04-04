"""DeepSeek client implementation"""

from typing import Dict, List
import requests
from rich.console import Console
from ..interfaces.llm_interface import LLMInterface
from ..models.analysis_models import AnalysisConfig

class DeepSeekClient(LLMInterface):
    """Client for DeepSeek API"""
    
    def __init__(self, api_key: str, base_url: str, config: AnalysisConfig):
        self.api_key = api_key
        self.base_url = base_url
        self.config = config
        self.console = Console()
        
    def chat(self, messages: List[Dict[str, str]], temperature: float = None) -> Dict:
        try:
            self.console.print("\n=== DeepSeek API Debug Info ===")
            for msg in messages:
                self.console.print(f"- Role: {msg['role']}")
                self.console.print(f"  Content: {msg['content'][:100]}...")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "stream": False
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            self.console.print("\nResponse received:")
            self.console.print(f"Status: {response.status_code}")
            self.console.print(f"Response type: {type(result)}")
            if "choices" in result and result["choices"]:
                self.console.print(f"First choice content: {result['choices'][0]['message']['content'][:100]}...")
            return result
        except requests.exceptions.RequestException as e:
            self.console.print(f"[red]Error in DeepSeek API call:[/red]")
            self.console.print(f"[red]Error type: {type(e)}[/red]")
            self.console.print(f"[red]Error message: {str(e)}[/red]")
            if hasattr(e.response, 'text'):
                self.console.print(f"[red]Response text: {e.response.text}[/red]")
            import traceback
            self.console.print("[red]Traceback:[/red]")
            self.console.print(traceback.format_exc())
            raise
