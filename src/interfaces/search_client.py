"""Interface for Search clients"""

from abc import ABC, abstractmethod
from typing import Dict, List

class SearchClientInterface(ABC):
    """Abstract base class for search clients"""
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> Dict:
        """Execute search query"""
        pass
    
    @abstractmethod
    def search_with_filters(self, query: str, site: str = None, max_results: int = 10) -> List[Dict]:
        """Execute search with site filters"""
        pass 