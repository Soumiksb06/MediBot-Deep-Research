"""Subject Analyzer Main Module - Interactive Refinement with Revert Option"""

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
            analysis = analyzer.analyze(topic)
            progress.remove_task(task)
            return analysis
        except Exception as e:
            progress.remove_task(task)
            raise e

def save_analysis(analysis: dict, topic: str, version: int = 1) -> str:
    """Save analysis results to a versioned file"""
    os.makedirs("analysis", exist_ok=True)
    filename = f"analysis/{topic.lower().replace(' ', '_')[:30]}_v{version}.md"
    
    # Format analysis content for Markdown
    content = f"# Analysis Report for: {topic}\n\n"
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
                if not topic:
                    continue

                version = 1
                saved_analyses = {}  # Dictionary to store versions
                
                while True:  # Loop until user confirms correctness
                    # Create a report request
                    request = ReportRequest(
                        topic=topic,
                        format=ReportFormat.BLOG if "blog" in topic.lower() else ReportFormat.ARTICLE,
                        verification_rounds=2
                    )
                    
                    # Perform analysis
                    analysis = analyze_subject(analyzer, topic, console)
                    main_subject = analysis.get("main_subject", "Unknown Subject")
                    research_areas = analysis.get("What_needs_to_be_researched", [])

                    console.print("\n[bold green]Agent's Understanding:[/bold green]")
                    console.print(f"Main Subject: [cyan]{main_subject}[/cyan]")
                    console.print(f"What needs to be researched: [cyan]{', '.join(research_areas) if research_areas else 'None'}[/cyan]\n")

                    # Ask user if the agent's understanding is correct
                    user_confirm = input("Did the agent understand your query perfectly? (yes/no/revert): ").strip().lower()

                    if user_confirm == "yes":
                        break  # Exit the loop if the user confirms

                    elif user_confirm == "revert":
                        if saved_analyses:
                            console.print("\n[bold cyan]Available Versions:[/bold cyan]")
                            for ver, details in saved_analyses.items():
                                console.print(f"Version {ver}: {details['topic']}")

                            revert_to = input("Enter the version number to revert to: ").strip()
                            if revert_to.isdigit() and int(revert_to) in saved_analyses:
                                topic = saved_analyses[int(revert_to)]["topic"]
                                analysis = saved_analyses[int(revert_to)]["analysis"]
                                console.print(f"[yellow]Reverted to Version {revert_to}[/yellow]\n")
                                continue
                            else:
                                console.print("[red]Invalid version number. Continuing with the latest version.[/red]")

                    else:  # User said "no" (or provided additional instruction)
                        clarification = input("Please provide clarification or refinements: ").strip()
                        if clarification:
                            topic = clarification  # Keep existing knowledge but refine
                        else:
                            console.print("[yellow]No clarification provided. Proceeding with existing analysis.[/yellow]")

                    # Save analysis before next iteration
                    saved_analyses[version] = {"topic": topic, "analysis": analysis}
                    version += 1

                # Save final version in Markdown format
                filename = save_analysis(analysis, topic, version)
                
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
