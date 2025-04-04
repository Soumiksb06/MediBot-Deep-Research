"""Tavily content extraction client"""

import requests
from typing import Dict, List
from rich.console import Console
from ..interfaces.extractor_interface import ExtractorInterface

class TavilyExtractor(ExtractorInterface):
    """Client for Tavily content extraction API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/extract"
        self.console = Console()
        
    def extract(
        self,
        urls: List[str],
        extract_depth: str = "basic",
        include_images: bool = False
    ) -> Dict:
        if not urls:
            raise ValueError("No URLs provided")
        if extract_depth not in ["basic", "advanced"]:
            raise ValueError("Invalid extract_depth. Must be 'basic' or 'advanced'")
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "urls": urls,
                "extract_depth": extract_depth,
                "include_images": include_images
            }
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            if result.get("failed_results"):
                self.console.print(f"[yellow]Warning: Failed to extract {len(result['failed_results'])} URLs[/yellow]")
                for failed in result["failed_results"]:
                    self.console.print(f"[yellow]- {failed}[/yellow]")
            return result
        except requests.exceptions.RequestException as e:
            self.console.print(f"[red]Error making Tavily API request: {str(e)}[/red]")
            raise Exception(f"Tavily API request failed: {str(e)}")
        except Exception as e:
            self.console.print(f"[red]Error extracting content: {str(e)}[/red]")
            raise
