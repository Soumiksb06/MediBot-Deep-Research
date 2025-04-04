"""DeepSeek R1 client implementation"""

from openai import OpenAI
from typing import List, Dict
from src.interfaces.llm_client import LLMClientInterface
from rich.console import Console

console = Console()

class DeepSeekClient(LLMClientInterface):
    """Client for DeepSeek R1 API"""
    
    def __init__(self, api_key: str, base_url: str):
        """Initialize DeepSeek client"""
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.console = Console()

    def chat(self, messages: List[dict], temperature: float = 0.7) -> dict:
        """Execute chat completion with DeepSeek R1"""
        try:
            # Print debug information
            print("\n=== DeepSeek API Debug Info ===")
            print(f"Messages being sent:")
            for msg in messages:
                print(f"- Role: {msg['role']}")
                print(f"  Content: {msg['content'][:100]}...")
            
            # Make the API call
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=messages,
                temperature=temperature,
                stream=False  # We want the complete response
            )
            
            # Print response debug info
            print("\nResponse received:")
            print(f"Status: Success")
            print(f"Response type: {type(response)}")
            print(f"First choice content: {response.choices[0].message.content[:100]}...")
            print(f"Tokens used - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}")
            
            return response
            
        except Exception as e:
            self.console.print(f"[red]Error in DeepSeek API call:[/red]")
            self.console.print(f"[red]Error type: {type(e)}[/red]")
            self.console.print(f"[red]Error message: {str(e)}[/red]")
            import traceback
            self.console.print("[red]Traceback:[/red]")
            self.console.print(traceback.format_exc())
            raise 