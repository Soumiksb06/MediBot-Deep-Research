"""Data models for research"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

@dataclass
class ResearchConfig:
    """Configuration for research parameters"""
    max_iterations: int = 3
    max_related_topics: int = 5
    top_sites_count: int = 15
    temperature: float = 0.7
    verification_rounds: int = 2   # Added field
    max_sources: int = 10        

@dataclass
class ResearchSource:
    """Research source information"""
    url: str
    domain: str
    score: float
    is_academic: bool
    content: Optional[str] = None

@dataclass
class ResearchFinding:
    """Research finding information"""
    title: str
    content: str
    source: ResearchSource
    timestamp: datetime = datetime.now()

@dataclass
class ResearchPlan:
    """Research plan information"""
    research_goals: List[str]
    hypotheses: List[str]
    methodology: List[Dict[str, str]]
    data_collection: List[str]
    analysis_methods: List[str]
    validation_steps: List[str]
    expected_outcomes: List[str]

@dataclass
class ResearchState:
    """Research state information"""
    topic: str
    keywords: List[str]
    top_sites: List[ResearchSource]
    findings: List[ResearchFinding]
    related_topics: List[str]
    research_plan: Optional[ResearchPlan] = None
    validation_results: Optional[Dict] = None
    final_document: Optional[str] = None

# --- Add the missing definitions below ---

class ReportFormat(Enum):
    COMPARISON = "comparison"
    BLOG = "blog"
    ABSTRACT = "abstract"
    ARTICLE = "article"

@dataclass
class ReportRequest:
    """Structure for a report generation request"""
    topic: str
    format: ReportFormat
    verification_rounds: int = 2
    max_sources: int = 10
    include_sources: bool = True
    additional_notes: Optional[str] = None
