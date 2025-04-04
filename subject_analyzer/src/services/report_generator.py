# subject_analyzer/src/services/report_generator.py
"""Research report generator service"""

import json
from datetime import datetime
from typing import List, Dict, Union
from rich.console import Console

from ..models.report_models import (
    ReportFormat,
    ReportRequest,
    ResearchFinding,
    ComparisonPoint,
    ResearchReport
)
from ..services.subject_analyzer import SubjectAnalyzer
from ..interfaces.search_interface import SearchInterface
from ..interfaces.extractor_interface import ExtractorInterface

class ReportGenerator:
    """Service for generating research reports"""
    
    def __init__(
        self,
        analyzer: SubjectAnalyzer,
        search_client: SearchInterface,
        extractor: ExtractorInterface
    ):
        """Initialize report generator
        
        Args:
            analyzer: Subject analyzer service
            search_client: Search client for web research
            extractor: Content extractor client
        """
        self.analyzer = analyzer
        self.search = search_client
        self.extractor = extractor
        self.console = Console()
        
    def _create_format_prompt(self, format: ReportFormat, findings: List[ResearchFinding]) -> str:
        """Create format-specific prompt
        
        Args:
            format: Desired report format
            findings: Research findings to include
        
        Returns:
            Formatted prompt string
        """
        sources_text = "\n\n".join([
            f"Source: {f.source_url}\nTitle: {f.title}\nContent: {f.content}"
            for f in findings if getattr(f, "verified", False)
        ])
        
        if format == ReportFormat.COMPARISON:
            return f"""Create a detailed comparison report using ONLY the information from these verified sources:

{sources_text}

Format the response as a comprehensive comparison with:
1. Introduction explaining what's being compared
2. A markdown comparison table with key features
3. Detailed analysis of each major difference
4. Conclusion with key takeaways

Use proper markdown formatting. Include ONLY factual information from the sources but adhere to user query."""
        elif format == ReportFormat.BLOG:
            return f"""Write an engaging blog post using ONLY the information from these verified sources while answering user's question:

{sources_text}

Make it engaging and conversational while maintaining accuracy. Include:
1. Attention-grabbing introduction
2. Clear sections with descriptive headings
3. Real examples and insights from the sources
4. Conclusion with key takeaways

Use proper markdown formatting. Include ONLY factual information from the sources."""
        elif format == ReportFormat.ABSTRACT:
            return f"""Write a research paper abstract using ONLY the information from these verified sources:

{sources_text}

Follow standard abstract structure:
1. Background/Context
2. Objective/Purpose
3. Methods/Approach
4. Results/Findings
5. Conclusion/Implications

Keep it concise and academic. Use proper markdown formatting. Include ONLY factual information from the sources to answer user's question."""
        else:
            return f"""Create a detailed article using ONLY the information from these verified sources and answer user's question:

{sources_text}

Format as a comprehensive article with:
1. Clear introduction
2. Well-organized sections
3. Supporting evidence from sources
4. Logical conclusions

Use proper markdown formatting. Include ONLY factual information from the sources to answer user's question.
Never divert from the original question or query user asked!"""
        
    def _verify_findings(self, findings: List[ResearchFinding], rounds: int = 2) -> List[ResearchFinding]:
        """Verify research findings through multiple search rounds
        
        Args:
            findings: Initial research findings
            rounds: Number of verification rounds
            
        Returns:
            List of verified findings
        """
        verified_findings = []
        
        # First extract full content from URLs
        try:
            urls = [f.source_url for f in findings if f.source_url]
            if urls:
                extracted = self.extractor.extract(
                    urls=urls,
                    extract_depth="advanced",
                    include_images=False
                )
                
                # Update findings with extracted content
                for finding in findings:
                    for result in extracted.get("results", []):
                        if result.get("url") == finding.source_url:
                            finding.content = result.get("text", finding.content)
                            break
        except Exception as e:
            self.console.print(f"[yellow]Warning: Content extraction failed: {str(e)}[/yellow]")
        
        for finding in findings:
            verification_count = 0
            verification_query = f"verify: {finding.title} {finding.content[:100]}"
            for _ in range(rounds):
                try:
                    verification_results = self.search.search(
                        query=verification_query,
                        max_results=10
                    )
                    for result in verification_results.get("results", []):
                        if any(keyword in result.get("content", "").lower() 
                               for keyword in finding.content.lower().split()):
                            verification_count += 1
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Verification failed for {finding.title}: {str(e)}[/yellow]")
            finding.verified = verification_count >= rounds
            if finding.verified:
                verified_findings.append(finding)
                
        return verified_findings

    def _extract_comparison_points(self, findings: List[ResearchFinding]) -> List[ComparisonPoint]:
        """Extract comparison points from findings
        
        Args:
            findings: Verified research findings
            
        Returns:
            List of comparison points
        """
        analysis = self.analyzer.analyze(" ".join([f.title for f in findings]))
        subjects = analysis["key_aspects"]
        comparison_points = []
        aspects = ["features", "performance", "use cases", "limitations"]
        for aspect in aspects:
            values = {}
            sources = []
            confidence = 0.0
            for subject in subjects:
                query = f"{subject} {aspect}"
                try:
                    results = self.search.search(query=query, max_results=3)
                    if results.get("results"):
                        result = results["results"][0]
                        values[subject] = result.get("answer", "No information found")
                        sources.append(result.get("url"))
                        confidence += float(result.get("relevance_score", 0.5))
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Could not find {aspect} for {subject}: {str(e)}[/yellow]")
            if values:
                comparison_points.append(ComparisonPoint(
                    aspect=aspect,
                    values=values,
                    sources=sources,
                    confidence=confidence / len(subjects)
                ))
        return comparison_points

    def generate_report(self, request: ReportRequest) -> ResearchReport:
        """Generate research report
        
        Args:
            request: Report generation request
            
        Returns:
            Generated research report
        """
        try:
            # Analyze the topic using the subject analyzer
            analysis_results = self.analyzer.analyze(request.topic)
            
            # Collect initial findings from search results if available
            findings = []
            search_results = analysis_results.get("search_results", {})  # Safely get search_results
            
            if search_results:
                # Process main subject search results if available
                main_results = search_results.get("main_subject", {}).get("results", [])
                for result in main_results:
                    if len(findings) < request.max_sources:
                        findings.append(ResearchFinding(
                            source_url=result.get("url", ""),
                            title=result.get("title", ""),
                            content=result.get("answer", result.get("content", "")),
                            relevance_score=float(result.get("relevance_score", 0.5))
                        ))
                # Process aspect results if available
                for aspect_results in search_results.get("aspects", {}).values():
                    for result in aspect_results.get("results", []):
                        if len(findings) < request.max_sources:
                            findings.append(ResearchFinding(
                                source_url=result.get("url", ""),
                                title=result.get("title", ""),
                                content=result.get("answer", result.get("content", "")),
                                relevance_score=float(result.get("relevance_score", 0.5))
                            ))
            else:
                self.console.print("[yellow]No search_results found in analysis; proceeding without additional source findings.[/yellow]")
            
            # Verify findings
            verified_findings = self._verify_findings(findings, request.verification_rounds)
            
            # Extract comparison points if needed
            comparison_points = None
            if request.format == ReportFormat.COMPARISON:
                comparison_points = self._extract_comparison_points(verified_findings)
            
            # Generate formatted content using the provided format prompt
            prompt = self._create_format_prompt(request.format, verified_findings)
            response = self.analyzer.llm.chat([
                {"role": "system", "content": "You are an expert research writer."},
                {"role": "user", "content": prompt}
            ])
            
            # Extract content from the LLM response
            content = self.analyzer.parser.extract_content(response)
            
            # Create the final report using subject analysis for title and confidence score
            report = ResearchReport(
                title=analysis_results.get("main_subject", "Untitled Report"),
                format=request.format,
                content=content,
                sources=[f.source_url for f in verified_findings],
                comparison_points=comparison_points,
                metadata={
                    "generated_at": datetime.now().isoformat(),
                    "confidence_score": analysis_results.get("confidence_score", 0),
                    "verified_sources_count": len(verified_findings)
                }
            )
            
            return report
            
        except Exception as e:
            self.console.print(f"[red]Error generating report:[/red]")
            self.console.print(f"[red]Error type: {type(e)}[/red]")
            self.console.print(f"[red]Error message: {str(e)}[/red]")
            raise
