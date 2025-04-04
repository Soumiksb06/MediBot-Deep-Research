"""Interface for search clients"""

from abc import ABC, abstractmethod
from typing import Dict, List

class SearchInterface(ABC):
    """Abstract base class for search clients"""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> Dict:
        """Execute search query
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing search results
        """
        pass 