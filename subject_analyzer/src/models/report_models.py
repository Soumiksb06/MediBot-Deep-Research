"""Models for research report generation"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

class ReportFormat(Enum):
    """Available report formats"""
    COMPARISON = "comparison"
    BLOG = "blog"
    ABSTRACT = "abstract"
    ARTICLE = "article"
    TECHNICAL = "technical"

@dataclass
class ReportRequest:
    """Research report request"""
    topic: str
    format: ReportFormat
    style_guide: Optional[Dict] = None
    max_sources: int = 10
    verification_rounds: int = 2

@dataclass
class ResearchFinding:
    """Individual research finding"""
    source_url: str
    title: str
    content: str
    relevance_score: float
    verified: bool = False

@dataclass
class ComparisonPoint:
    """Point of comparison between subjects"""
    aspect: str
    values: Dict[str, str]
    sources: List[str]
    confidence: float

@dataclass
class ResearchReport:
    """Final research report"""
    title: str
    format: ReportFormat
    content: str
    sources: List[str]
    comparison_points: Optional[List[ComparisonPoint]] = None
    metadata: Optional[Dict] = None 