"""Interface for Research services"""

from abc import ABC, abstractmethod
from typing import Dict, List

class ResearchServiceInterface(ABC):
    """Abstract base class for research services"""
    
    @abstractmethod
    async def create_research_plan(self, topic: str) -> Dict:
        """Create research plan"""
        pass
    
    @abstractmethod
    async def validate_research_plan(self, plan: Dict) -> Dict:
        """Validate research plan"""
        pass
    
    @abstractmethod
    async def find_sources(self, topic: str) -> List[str]:
        """Find research sources"""
        pass
    
    @abstractmethod
    async def analyze_content(self, content: str) -> Dict:
        """Analyze research content"""
        pass
    
    @abstractmethod
    async def synthesize_findings(self, findings: List[Dict]) -> Dict:
        """Synthesize research findings"""
        pass 