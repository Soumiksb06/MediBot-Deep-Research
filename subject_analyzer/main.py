"""Subject Analyzer Main Module"""

import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from src.config.config_loader import ConfigLoader
from src.models.report_models import ReportFormat, ReportRequest
from src.services.deepseek_client import DeepSeekClient
from src.services.subject_analyzer import SubjectAnalyzer

def analyze_subject(analyzer: SubjectAnalyzer, topic: str, console: Console) -> dict:
    """Analyze a subject using the analyzer service"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Analyzing subject...", total=None)
        try:
            # Use the correct method: analyze() instead of analyze_subject()
            analysis = analyzer.analyze(topic)
            progress.remove_task(task)
            return analysis
        except Exception as e:
            progress.remove_task(task)
            raise e

def save_analysis(analysis: dict, topic: str) -> str:
    """Save analysis results to file"""
    os.makedirs("analysis", exist_ok=True)
    filename = f"analysis/{topic.lower().replace(' ', '_')[:30]}.md"
    
    # Format analysis content for Markdown
    content = f"# Analysis Report for: {topic}\n\n"
    # Include the main subject, temporal context, and research areas
    main_subject = analysis.get("main_subject", "Unknown Subject")
    temporal_context = analysis.get("temporal_context", {})
    research_areas = analysis.get("What_needs_to_be_researched", [])
    
    content += f"## Main Subject\n{main_subject}\n\n"
    content += "## Temporal Context\n"
    if temporal_context:
        for key, value in temporal_context.items():
            content += f"- **{key.capitalize()}:** {value}\n"
    else:
        content += "None\n"
    content += "\n"
    
    content += "## What Needs to be Researched\n"
    if research_areas:
        for area in research_areas:
            content += f"- {area}\n"
    else:
        content += "None\n"
    content += "\n"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filename

def main():
    """Main entry point for subject analyzer"""
    console = Console()
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config, api_key, base_url = ConfigLoader.load_config()
        
        # Initialize services
        llm_client = DeepSeekClient(api_key, base_url, config)
        analyzer = SubjectAnalyzer(llm_client, config)
        
        console.print("\n[bold blue]Subject Analyzer[/bold blue]")
        console.print("Enter a subject to analyze. Analysis will include:")
        console.print("- Main Subject of Query")
        console.print("- Temporal Context")
        console.print("- What needs to be researched (research areas)")
        console.print("\nPress Ctrl+C to exit.\n")
        
        while True:
            try:
                topic = input("Subject to analyze: ").strip()
                if topic:
                    # Create a report request
                    request = ReportRequest(
                        topic=topic,
                        format=ReportFormat.BLOG if "blog" in topic.lower() else ReportFormat.ARTICLE,
                        verification_rounds=2
                    )
                    
                    # Perform analysis
                    analysis = analyze_subject(analyzer, topic, console)
                    main_subject = analysis.get("main_subject", "Unknown Subject")
                    # Now using the new key for research areas
                    research_areas = analysis.get("What_needs_to_be_researched", [])
                    
                    console.print("\n[bold green]Agent's Understanding:[/bold green]")
                    console.print(f"Main Subject: [cyan]{main_subject}[/cyan]")
                    console.print(f"What needs to be researched: [cyan]{', '.join(research_areas) if research_areas else 'None'}[/cyan]\n")
                    
                    # Ask user if the agent's understanding is correct
                    user_confirm = input("Did the agent understand your query perfectly? (yes/no): ").strip().lower()
                    if user_confirm != "yes":
                        clarification = input("Please provide clarification: ").strip()
                        if clarification:
                            topic = clarification
                            analysis = analyze_subject(analyzer, topic, console)
                            main_subject = analysis.get("main_subject", "Unknown Subject")
                            research_areas = analysis.get("What_needs_to_be_researched", [])
                            console.print("\n[bold green]Revised Analysis Results:[/bold green]")
                            console.print(f"Main Subject: [cyan]{main_subject}[/cyan]")
                            console.print(f"What needs to be researched: [cyan]{', '.join(research_areas) if research_areas else 'None'}[/cyan]\n")
                        else:
                            console.print("[yellow]No clarification provided. Proceeding with initial analysis.[/yellow]")
                    
                    # Save results in Markdown format
                    filename = save_analysis(analysis, topic)
                    
                    console.print(f"\n[green]Analysis completed successfully![/green]")
                    console.print(f"Query: {topic}")
                    console.print(f"\nSaved to: {filename}")
                
                console.print("\nEnter another subject or press Ctrl+C to exit.\n")
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Exiting Subject Analyzer...[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                continue
                
    except ValueError as e:
        console.print(f"\n[red]Configuration Error: {str(e)}[/red]")
        console.print("[yellow]Please check your API keys in the .env file[/yellow]")
        return

if __name__ == "__main__":
    main()
