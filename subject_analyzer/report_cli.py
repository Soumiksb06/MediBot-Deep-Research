"""Research Report Generator CLI"""

import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config.config_loader import ConfigLoader
from src.services.deepseek_client import DeepSeekClient
from src.services.tavily_client import TavilyClient
from src.services.tavily_extractor import TavilyExtractor
from src.services.subject_analyzer import SubjectAnalyzer
from src.services.report_generator import ReportGenerator
from src.models.report_models import ReportFormat, ReportRequest

def detect_format(topic: str) -> ReportFormat:
    """Detect requested format from topic
    
    Args:
        topic: Research topic/request
        
    Returns:
        Detected report format
    """
    topic_lower = topic.lower()
    
    if any(word in topic_lower for word in ["compare", "comparison", "versus", "vs"]):
        return ReportFormat.COMPARISON
    elif "blog" in topic_lower:
        return ReportFormat.BLOG
    elif any(word in topic_lower for word in ["abstract", "research paper"]):
        return ReportFormat.ABSTRACT
    else:
        return ReportFormat.ARTICLE

def save_report(report_content: str, title: str):
    """Save report to file
    
    Args:
        report_content: Report content in markdown format
        title: Report title for filename
    """
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Create filename from title
    filename = f"reports/{title.lower().replace(' ', '_')[:30]}.md"
    
    # Save report
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    return filename

def main():
    """Main entry point"""
    console = Console()
    
    try:
        # Load configuration
        config, api_key, base_url = ConfigLoader.load_config()
        
        # Initialize services
        llm_client = DeepSeekClient(api_key, base_url, config)
        search_client = TavilyClient(config.tavily_api_key)
        extractor = TavilyExtractor(config.tavily_api_key)
        analyzer = SubjectAnalyzer(llm_client, search_client, config)
        generator = ReportGenerator(analyzer, search_client, extractor)
        
        console.print("\n[bold blue]Research Report Generator[/bold blue]")
        console.print("Enter your research request. Format will be auto-detected, or you can specify:")
        console.print("- 'vs' or 'compare' for comparison")
        console.print("- 'blog' for blog post")
        console.print("- 'abstract' for research paper abstract")
        console.print("\nPress Ctrl+C to exit.\n")
        
        while True:
            try:
                topic = input("Research request: ").strip()
                if topic:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        # Create request
                        format = detect_format(topic)
                        request = ReportRequest(
                            topic=topic,
                            format=format,
                            verification_rounds=2
                        )
                        
                        # Generate report
                        task = progress.add_task(f"[cyan]Generating {format.value} report...", total=None)
                        report = generator.generate_report(request)
                        progress.remove_task(task)
                        
                        # Save report
                        filename = save_report(report.content, report.title)
                        
                        # Print summary
                        console.print(f"\n[green]Report generated successfully![/green]")
                        console.print(f"Format: {report.format.value}")
                        console.print(f"Title: {report.title}")
                        console.print(f"Sources: {len(report.sources)}")
                        console.print(f"Confidence: {report.metadata['confidence_score']:.2%}")
                        console.print(f"\nSaved to: {filename}")
                    
                console.print("\nEnter another request or press Ctrl+C to exit.\n")
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Exiting Report Generator...[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                
    except ValueError as e:
        console.print(f"\n[red]Configuration Error: {str(e)}[/red]")
        console.print("[yellow]Please check your API keys in the .env file[/yellow]")
        return

if __name__ == "__main__":
    main() 