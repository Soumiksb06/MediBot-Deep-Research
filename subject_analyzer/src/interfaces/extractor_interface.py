"""Interface for content extraction clients"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class ExtractorInterface(ABC):
    """Abstract base class for content extraction clients"""
    
    @abstractmethod
    def extract(
        self,
        urls: List[str],
        extract_depth: str = "basic",
        include_images: bool = False
    ) -> Dict:
        """Extract content from URLs
        
        Args:
            urls: List of URLs to extract content from
            extract_depth: Extraction depth ('basic' or 'advanced')
            include_images: Whether to include images in extraction
            
        Returns:
            Dict containing extraction results and metadata
        """
        pass 