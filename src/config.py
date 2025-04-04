"""Configuration management for the Research AI Agent"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def validate_api_keys() -> Optional[str]:
    """Validate that all required API keys are present"""
    if not os.getenv("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY is not set"
    if not os.getenv("TAVILY_API_KEY"):
        return "TAVILY_API_KEY is not set"
    return None