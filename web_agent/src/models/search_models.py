"""Data models for web search"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchConfig:
    """Configuration for web search"""
    max_results: int = 10
    max_results_per_aspect: int = 4
    include_snippets: bool = True
    min_relevance_score: float = 0.5

@dataclass
class SearchResult:
    """Individual search result"""
    url: str
    title: str
    content: str
    relevance_score: float
    source_type: str
    timestamp: Optional[str] = None

@dataclass
class AspectSearchResults:
    """Search results for a specific aspect"""
    aspect: str
    results: List[SearchResult]
    total_found: int
    average_relevance: float 
