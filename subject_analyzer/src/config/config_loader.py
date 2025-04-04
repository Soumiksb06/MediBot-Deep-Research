"""Configuration loader"""

import os
from typing import Optional, Tuple
from dotenv import load_dotenv

from ..models.analysis_models import AnalysisConfig

class ConfigLoader:
    """Loader for application configuration"""
    
    @staticmethod
    def load_config() -> Tuple[AnalysisConfig, str, str]:
        """Load configuration from environment
        
        Returns:
            Tuple containing:
            - AnalysisConfig object
            - API key
            - Base URL
            
        Raises:
            ValueError: If required configuration is missing
        """
        # Load environment variables
        load_dotenv()
        
        # Get API configuration
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepinfra.com/v1/openai")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        # Create analysis config
        config = AnalysisConfig(
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            model_name=os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1"),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout=int(os.getenv("TIMEOUT", "30")),
            tavily_api_key=tavily_api_key
        )
        
        return config, api_key, base_url 