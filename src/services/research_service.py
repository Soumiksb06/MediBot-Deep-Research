"""Research service implementation"""

import json
from typing import Dict, List
from src.interfaces.research_service import ResearchServiceInterface
from src.interfaces.llm_client import LLMClientInterface
from src.interfaces.search_client import SearchClientInterface
from src.models.research_models import ResearchConfig, ResearchState, ResearchSource, ResearchFinding
from src.utils.text_utils import clean_text
from src.utils.web_utils import extract_domain, validate_source
from rich.console import Console

console = Console()

class ResearchService(ResearchServiceInterface):
    """Service for conducting research"""
    
    def __init__(
        self,
        llm_client: LLMClientInterface,
        search_client: SearchClientInterface,
        config: ResearchConfig
    ):
        """Initialize research service"""
        self.llm = llm_client
        self.search = search_client
        self.config = config
        self.state = ResearchState(
            topic="",
            keywords=[],
            top_sites=[],
            findings=[],
            related_topics=[]
        )

    def _extract_llm_response(self, response: Dict) -> str:
        """Safely extract content from LLM response"""
        try:
            # Get the raw content
            content = response.choices[0].message.content
            
            # Clean the content - remove any non-JSON text
            # Find the first { and last }
            start = content.find('{')
            end = content.rfind('}')
            
            if start == -1 or end == -1:
                self.console.print("[red]No JSON object found in response[/red]")
                self.console.print(f"Raw response: {content}")
                raise ValueError("No JSON object found in response")
            
            json_str = content[start:end+1]
            
            # Validate that it's proper JSON
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                self.console.print("[red]Invalid JSON in response[/red]")
                self.console.print(f"Extracted JSON string: {json_str}")
                raise
            
        except Exception as e:
            self.console.print(f"[red]Error extracting LLM response: {str(e)}[/red]")
            self.console.print(f"Response structure: {response}")
            raise

    async def create_research_plan(self, topic: str) -> Dict:
        """Create research plan"""
        prompt = f"""As a PhD-level research planner, create a detailed research plan for the topic: {topic}
        
        The plan should follow these research best practices:
        1. Define clear research goals and hypotheses
        2. Conduct thorough literature review
        3. Design robust methodology
        
        Format the response as a JSON with the following structure:
        {{
            "research_goals": [],
            "hypotheses": [],
            "methodology": [],
            "data_collection": [],
            "analysis_methods": [],
            "validation_steps": [],
            "expected_outcomes": []
        }}
        """
        
        response = self.llm.chat([
            {"role": "system", "content": "You are a PhD-level research planner."},
            {"role": "user", "content": prompt}
        ])
        
        content = self._extract_llm_response(response)
        return json.loads(content)

    async def validate_research_plan(self, plan: Dict) -> Dict:
        """Validate research plan"""
        validation_prompt = f"""As a research methodology expert, validate the following research plan:
        {json.dumps(plan, indent=2)}
        
        Evaluate against research best practices and provide detailed feedback.
        Return response as JSON with:
        {{
            "is_valid": bool,
            "feedback": [],
            "improvements": [],
            "compliance_score": float
        }}
        """
        
        response = self.llm.chat([
            {"role": "system", "content": "You are a research methodology expert."},
            {"role": "user", "content": validation_prompt}
        ])
        
        content = self._extract_llm_response(response)
        return json.loads(content)

    async def find_sources(self, topic: str) -> List[str]:
        """Find research sources"""
        query = f"Best academic and research websites for {topic}"
        results = self.search.search(
            query=query,
            max_results=self.config.top_sites_count,  # ✅ Correct parameter
            search_depth="advanced",  # ✅ Explicitly pass search_depth
            topic="general",  # ✅ Optional, can be "news" if needed
            include_answer=True,  # ✅ Get an AI-generated summary if available
            include_raw_content=False  # ✅ Set to True if full HTML content is needed
        )

        
        sources = []
        for result in results.get("results", []):
            url = result.get("url", "")
            domain = extract_domain(url)
            validation = validate_source(url)
            
            source = ResearchSource(
                url=url,
                domain=domain,
                score=validation["score"],
                is_academic=validation["is_academic"],
                content=clean_text(result.get("content", ""))
            )
            sources.append(source)
        
        return sources

    async def analyze_content(self, content: str) -> Dict:
        """Analyze research content"""
        analysis_prompt = f"""Analyze this research content and provide:
        1. Key findings
        2. Methodology used
        3. Limitations
        4. Future research directions
        5. Practical implications
        
        Content: {content[:2000]}  # Limit content length for API
        
        Format response as JSON with these keys."""
        
        response = self.llm.chat([
            {"role": "system", "content": "You are a research analysis expert."},
            {"role": "user", "content": analysis_prompt}
        ])
        
        content = self._extract_llm_response(response)
        return json.loads(content)

    async def synthesize_findings(self, findings: List[Dict]) -> Dict:
        """Synthesize research findings"""
        synthesis_prompt = f"""As a research synthesis expert, analyze these findings:
        {json.dumps(findings, indent=2)}
        
        Provide:
        1. Key themes
        2. Consensus points
        3. Contradictions
        4. Research gaps
        5. Recommendations
        
        Format as JSON with these keys."""
        
        response = self.llm.chat([
            {"role": "system", "content": "You are a research synthesis expert."},
            {"role": "user", "content": synthesis_prompt}
        ])
        
        content = self._extract_llm_response(response)
        return json.loads(content) 