# subject_analyzer/src/services/tavily_client.py
from typing import Dict
from tavily import TavilyClient as TavilyAPI
from rich.console import Console
from ..interfaces.search_interface import SearchInterface

class TavilyClient(SearchInterface):
    """Client for Tavily search API"""

    def __init__(self, api_key: str):
        self.client = TavilyAPI(api_key=api_key)
        self.console = Console()

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "basic",
        topic: str = "general",
        include_answer: bool or str = True,
        include_raw_content: bool = False,
        include_images: bool = False,
        include_image_descriptions: bool = False,
        include_domains: list = None,
        exclude_domains: list = None,
        **kwargs
    ) -> Dict:
        if include_domains is None:
            include_domains = []
        if exclude_domains is None:
            exclude_domains = []

        try:
            self.console.print(f"\n[cyan]Searching for: {query}[/cyan] (Depth: {search_depth}, Topic: {topic})")
            params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "topic": topic,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
                "include_image_descriptions": include_image_descriptions,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
            }
            params.update(kwargs)
            response = self.client.search(**params)
            if "results" in response:
                self.console.print(f"[green]Found {len(response['results'])} results[/green]")
            return response
        except Exception as e:
            self.console.print(f"[red]Error in Tavily search:[/red] {str(e)}")
            raise
