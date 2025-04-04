"""Interface for LLM clients"""

from abc import ABC, abstractmethod
from typing import Dict, List

class LLMInterface(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict:
        """Execute chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            
        Returns:
            Dict containing the model's response
        """
        pass 