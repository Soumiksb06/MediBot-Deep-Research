"""Subject analyzer service"""

from typing import Dict, List
from rich.console import Console
from datetime import datetime
import re

from ..interfaces.llm_interface import LLMInterface
from ..models.analysis_models import SubjectAnalysis, AnalysisConfig
from ..utils.response_parser import ResponseParser

class SubjectAnalyzer:
    """Service for analyzing subjects in text"""
    
    def __init__(self, llm_client: LLMInterface, config: AnalysisConfig):
        """Initialize subject analyzer
        
        Args:
            llm_client: LLM client interface implementation
            config: Analysis configuration
        """
        self.llm = llm_client
        self.config = config
        self.parser = ResponseParser()
        self.console = Console()
    
    def _extract_temporal_context(self, text: str) -> Dict[str, str]:
        patterns = {
            'month_year': r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4}',
            'year': r'\b\d{4}\b',
            'relative': r'\b(?:current|upcoming|future|next|last|previous)\b'
        }
        temporal_info = {}
        for context_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                temporal_info[context_type] = matches[0]
        return temporal_info
    
    def _create_analysis_prompt(self, text: str) -> str:
        temporal_info = self._extract_temporal_context(text)
        prompt = f"""You are a subject analysis expert skilled in extracting the core subject from any given text. Your task is to analyze the following text and determine its primary subject. The text may contain a website URL, company name, product name, personal name, or any other entity. You must identify the most relevant subject based on context. 

Text to analyze: {text}

"""
        if temporal_info:
            prompt += """Note: The text contains temporal context. Include any relevant timeframe details in your analysis.
"""
        prompt += """Your response must be a JSON object with exactly the following keys and no additional text:
{
    "main_subject": "The primary subject or entity. If the text includes a URL, company, product, or individual name, use that. If multiple entities are present, choose the most relevant one.",
    "temporal_context": {
        "timeframe": "Extracted timeframe if mentioned, else leave blank",
        "relevance": "Explanation of why this timeframe is significant for the subject"
    },
    "What_needs_to_be_researched": ["A list of key research areas tailored to the identified main subject"]
}
Ensure that your entire response is ONLY the JSON object with no extra commentary.
"""
        return prompt


    
    def analyze(self, text: str) -> Dict:
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a subject analysis expert. You must respond ONLY with valid JSON objects."
                },
                {
                    "role": "user",
                    "content": self._create_analysis_prompt(text)
                }
            ]
            response = self.llm.chat(messages)
            result = self.parser.extract_json(response)
            required_fields = ["main_subject", "temporal_context", "What_needs_to_be_researched"]
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                raise ValueError(f"Missing required fields in response: {missing_fields}")
            return result
        except Exception as e:
            self.console.print(f"[red]Error analyzing subject:[/red]")
            self.console.print(f"[red]Error type: {type(e)}[/red]")
            self.console.print(f"[red]Error message: {str(e)}[/red]")
            raise
