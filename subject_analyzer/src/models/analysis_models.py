"""Data models for subject analysis"""

from dataclasses import dataclass
from typing import List

@dataclass
class SubjectAnalysis:
    """Data class for subject analysis results"""
    main_subject: str
    What_needs_to_be_researched: List[str]
    temporal_context: List[str]

@dataclass
class AnalysisConfig:
    """Configuration for subject analysis"""
    temperature: float = 0.6
    model_name: str = "gemini-2.5-pro-exp-03-25"
    max_retries: int = 3
    timeout: int = 30
    tavily_api_key: str = None  # Tavily API key for web search

@dataclass
class AnalysisResult:
    """Data class for subject analysis result"""
    subject_analysis: SubjectAnalysis
    analysis_config: AnalysisConfig
    result: str
    confidence_score: float
    timestamp: str
    error: str = None
