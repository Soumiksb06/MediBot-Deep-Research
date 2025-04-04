"""Interface for Language Model clients"""

from abc import ABC, abstractmethod
from typing import List, Dict

class LLMClientInterface(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def chat(self, messages: List[Dict], temperature: float = 0.7) -> Dict:
        """Execute chat completion"""
        pass 