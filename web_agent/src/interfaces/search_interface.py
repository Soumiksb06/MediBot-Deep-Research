# Research_agent/web_agent/src/interfaces/search_interface.py
from typing import Dict

class SearchInterface:
    def search(self, query: str, max_results: int = 10, **kwargs) -> Dict:
        """Perform a search query."""
        raise NotImplementedError("This method should be overridden.")
