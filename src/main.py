"""Main module for Research AI Agent"""

import os
from typing import Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from .config import validate_api_keys
from .models.research_models import ResearchConfig
from .clients.deepseek_client import DeepSeekClient
from .clients.tavily_client import TavilyClient
from .services.research_service import ResearchService

# Initialize console
console = Console()

def create_research_service() -> ResearchService:
    """Create and configure research service"""
    # Load environment variables
    load_dotenv()
    
    # Validate API keys
    if error := validate_api_keys():
        raise ValueError(f"Configuration error: {error}")
    
    # Initialize clients
    llm_client = DeepSeekClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    search_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Create config
    config = ResearchConfig(
        max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
        max_related_topics=5,
        top_sites_count=int(os.getenv("MAX_SOURCES", "6")),
        temperature=float(os.getenv("TEMPERATURE", "0.7"))
    )
    
    return ResearchService(
        llm_client=llm_client,
        search_client=search_client,
        config=config
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 