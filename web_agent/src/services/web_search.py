"""Web search service"""

from typing import Dict, List
from rich.console import Console
from ..models.search_models import SearchConfig

class WebSearchService:
    """Service for performing web searches"""

    def __init__(self, search_client, config: SearchConfig):
        """
        Initialize web search service.

        Args:
            search_client: Implementation of the search client.
            config (SearchConfig): Configuration for search parameters.
        """
        self.search = search_client
        self.config = config
        self.console = Console()

    def search_subject(self, subject: str, domain: str, **kwargs) -> Dict:
        """
        Search for information about a subject.

        Args:
            subject (str): The main subject to search for.
            domain (str): The domain/field of the subject.
            **kwargs: Additional parameters to pass to the search client (e.g. search_depth, results).
        
        Returns:
            Dict: The search results.
        """
        try:
            self.console.print(f"\n[bold]Searching for subject: {subject}[/bold]")
            query = f"{subject} {domain}"
            max_results = kwargs.get("results", self.config.max_results)
            results = self.search.search(
                query=query,
                max_results=max_results,
                **kwargs
            )
            return results
        except Exception as e:
            self.console.print(f"[red]Error searching for subject: {str(e)}[/red]")
            raise

    def search_aspects(self, subject: str, aspects: List[str]) -> Dict[str, Dict]:
        """
        Search for information about specific aspects.

        Args:
            subject (str): The main subject context.
            aspects (List[str]): A list of aspects to search for.
        
        Returns:
            Dict: A mapping of each aspect to its search results.
        """
        try:
            self.console.print("\n[bold]Searching for aspects...[/bold]")
            results = {}
            for aspect in aspects:
                query = f"{subject} {aspect}"
                aspect_results = self.search.search(
                    query=query,
                    max_results=self.config.max_results_per_aspect
                )
                results[aspect] = aspect_results
            return results
        except Exception as e:
            self.console.print(f"[red]Error searching for aspects: {str(e)}[/red]")
            raise
