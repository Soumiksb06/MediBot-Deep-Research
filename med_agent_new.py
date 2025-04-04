#!/usr/bin/env python3
"""
Medical Health Help System using Interactive Patient Engagement,
Subject Analysis, Web Content Extraction, RAG with Supabase, and
Comprehensive Diagnostic Reporting. In addition to generating a
detailed diagnostic report, a simplified patient-friendly summary is created.
This version now includes clearer, actionable recommendations for the patient.
Supports local file references (PDF, DOCX, CSV, Excel, TXT), robust logging,
asynchronous operations, and user-defined search parameters.
New enhancements include ML/NLP-based diagnosis prediction, sentiment analysis,
NER for medical terms, visualization of query trends, FAISS-based vector search optimization,
explainability via SHAP, and wearable data integration.
Required packages:
  - PyPDF2, python-docx, pandas, rich, python-dotenv, openai, supabase,
    scikit-learn, nltk, spacy, shap, faiss-cpu, matplotlib, seaborn
"""

import os
import sys
import asyncio
import logging
import csv
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console

# Configure logging
logging.basicConfig(
    filename="medical_agent.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Add project root to sys.path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import agent modules
from web_agent.src.services.web_search import WebSearchService
from web_agent.src.models.search_models import SearchConfig
from subject_analyzer.src.services.tavily_client import TavilyClient
from subject_analyzer.src.services.tavily_extractor import TavilyExtractor
from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer
from subject_analyzer.src.services.deepseek_client import DeepSeekClient
from subject_analyzer.src.models.analysis_models import AnalysisConfig

# ==============================
# New Data Science / Analytics & Visualization Functions
# ==============================

def train_symptom_classifier():
    """
    Trains a simple symptom-to-diagnosis classifier using a dummy dataset.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    # Dummy training data
    data = [
        ("fever cough sore throat", "Flu"),
        ("headache nausea sensitivity to light", "Migraine"),
        ("chest pain shortness of breath", "Heart Attack"),
        ("joint pain stiffness", "Arthritis"),
        ("abdominal pain diarrhea vomiting", "Gastroenteritis")
    ]
    texts, labels = zip(*data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, labels)
    return vectorizer, model

def predict_diagnosis(query, vectorizer, model):
    """
    Predicts a diagnosis from the patient query and returns the label with probability scores.
    """
    X_query = vectorizer.transform([query])
    pred = model.predict(X_query)[0]
    proba = model.predict_proba(X_query)[0]
    prob_dict = dict(zip(model.classes_, proba))
    return pred, prob_dict

def analyze_sentiment(query):
    """
    Performs sentiment analysis on the query using NLTK's VADER.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(query)
        return score
    except Exception as e:
        return {"compound": 0.0, "neg": 0.0, "neu": 0.0, "pos": 0.0}

def extract_medical_entities(query):
    """
    Extracts medical entities from the query using spaCy.
    Only returns entities matching a predefined list of common medical terms.
    """
    try:
        import spacy
        # Try to load the English model
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        common_med_terms = {"flu", "migraine", "heart attack", "arthritis", "gastroenteritis",
                            "fever", "cough", "pain", "diarrhea", "vomiting", "headache",
                            "nausea", "chest pain", "shortness of breath", "sore throat"}
        entities = set()
        for ent in doc.ents:
            if ent.text.lower() in common_med_terms:
                entities.add(ent.text)
        return list(entities)
    except Exception as e:
        return []

def save_query_history(query, diagnosis, sentiment, entities):
    """
    Saves the query details to a CSV file for trend analysis.
    """
    filename = "query_history.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "query", "predicted_diagnosis", "sentiment_compound", "entities"])
        writer.writerow([datetime.now().isoformat(), query, diagnosis, sentiment.get("compound", 0), ", ".join(entities)])

def visualize_query_trends():
    """
    Generates visualizations from the query history:
      - Bar chart for frequency of predicted diagnoses.
      - Line chart for sentiment compound score over time.
      - Pie chart for distribution of extracted medical entities.
    The plots are saved as PNG files.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter

    filename = "query_history.csv"
    if not os.path.exists(filename):
        print("No query history available for visualization.")
        return
    df = pd.read_csv(filename)
    if df.empty:
        print("Query history is empty.")
        return

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Bar plot for predicted diagnosis frequency
    diag_counts = df['predicted_diagnosis'].value_counts()
    plt.figure(figsize=(8,6))
    sns.barplot(x=diag_counts.index, y=diag_counts.values, palette="viridis")
    plt.title("Frequency of Predicted Diagnoses")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.tight_layout()
    diag_plot_file = "diagnosis_frequency.png"
    plt.savefig(diag_plot_file)
    plt.close()

    # Line plot for sentiment compound over time
    df = df.sort_values("timestamp")
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x="timestamp", y="sentiment_compound", marker="o")
    plt.title("Sentiment Compound Score Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Sentiment Compound Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    sentiment_plot_file = "sentiment_over_time.png"
    plt.savefig(sentiment_plot_file)
    plt.close()

    # Pie chart for common medical entities
    entity_list = []
    for entities_str in df['entities']:
        if pd.isna(entities_str) or entities_str.strip() == "":
            continue
        entities = [e.strip() for e in entities_str.split(",") if e.strip()]
        entity_list.extend(entities)
    if entity_list:
        entity_counts = Counter(entity_list)
        labels = list(entity_counts.keys())
        sizes = list(entity_counts.values())
        plt.figure(figsize=(8,8))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.title("Distribution of Extracted Medical Entities")
        plt.tight_layout()
        entities_plot_file = "entities_distribution.png"
        plt.savefig(entities_plot_file)
        plt.close()
    else:
        entities_plot_file = None

    print(f"Diagnosis frequency plot saved as: {diag_plot_file}")
    print(f"Sentiment over time plot saved as: {sentiment_plot_file}")
    if entities_plot_file:
        print(f"Medical entities distribution plot saved as: {entities_plot_file}")

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from a list of embeddings.
    """
    try:
        import faiss
        import numpy as np
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        embeddings_array = np.array(embeddings).astype('float32')
        index.add(embeddings_array)
        return index
    except Exception as e:
        return None

def search_faiss(index, query_embedding, chunks, k):
    """
    Searches the FAISS index with the query embedding and returns matched chunks.
    """
    import numpy as np
    query_vec = np.array(query_embedding).reshape(1, -1).astype('float32')
    distances, indices = index.search(query_vec, k)
    matched_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return matched_chunks

def read_wearable_data():
    """
    Reads wearable data from 'wearable_data.csv' (if available) and returns a summary string.
    """
    import pandas as pd
    filename = "wearable_data.csv"
    if not os.path.exists(filename):
        return None
    df = pd.read_csv(filename)
    summary = ""
    if "heart_rate" in df.columns:
        avg_hr = df["heart_rate"].mean()
        summary += f"Average Heart Rate: {avg_hr:.1f}\n"
    if "steps" in df.columns:
        total_steps = df["steps"].sum()
        summary += f"Total Steps: {total_steps}\n"
    return summary

def explain_diagnosis(query, vectorizer, model):
    """
    Provides a simple explanation for the predicted diagnosis using SHAP.
    """
    try:
        import shap
        X_query = vectorizer.transform([query]).toarray()
        # Create an explainer using a sample of the training data
        sample_texts = vectorizer.transform(vectorizer.get_feature_names_out()).toarray()
        explainer = shap.LinearExplainer(model, sample_texts, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_query)
        feature_names = vectorizer.get_feature_names_out()
        import numpy as np
        abs_shap = np.abs(shap_values[0])
        top_indices = np.argsort(abs_shap)[-5:][::-1]
        explanation = "Top contributing features:\n"
        for idx in top_indices:
            explanation += f"- {feature_names[idx]}: {shap_values[0][idx]:.4f}\n"
        return explanation
    except Exception as e:
        return "Explanation not available."

# ==============================
# Existing Code (with modifications)
# ==============================

def extract_text_from_file(file_path, console):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            try:
                import PyPDF2
            except ImportError:
                console.print("[red]PyPDF2 is required for PDF extraction. Install with 'pip install PyPDF2'.[/red]")
                return ""
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        elif ext == '.docx':
            try:
                import docx
            except ImportError:
                console.print("[red]python-docx is required for DOCX extraction. Install with 'pip install python-docx'.[/red]")
                return ""
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        elif ext in ['.csv']:
            try:
                import pandas as pd
            except ImportError:
                console.print("[red]pandas is required for CSV extraction. Install with 'pip install pandas'.[/red]")
                return ""
            df = pd.read_csv(file_path)
            return df.to_csv(index=False)
        elif ext in ['.xls', '.xlsx']:
            try:
                import pandas as pd
            except ImportError:
                console.print("[red]pandas is required for Excel extraction. Install with 'pip install pandas'.[/red]")
                return ""
            df = pd.read_excel(file_path)
            return df.to_csv(index=False)
        else:
            with open(file_path, 'r', encoding="utf-8") as f:
                return f.read()
    except Exception as ex:
        console.print(f"[yellow]Warning: Could not extract file {file_path}: {ex}[/yellow]")
        logging.error(f"File extraction error for {file_path}: {ex}")
        return ""

class MedicalTask:
    def __init__(self, query, console: Console):
        self.original_query = query
        self.current_query = query
        self.analysis = {}
        self.search_results = {}      # topic -> list of search result dicts
        self.extracted_content = {}   # topic -> list of extraction result dicts
        self.console = console
        self.feedback_history = []

    def update_feedback(self, feedback):
        self.feedback_history.append({
            "query": self.current_query,
            "feedback": feedback,
            "time": datetime.now().isoformat()
        })

    async def analyze(self, subject_analyzer, current_date: str):
        self.console.print(f"Analyzing patient query for diagnosis (as of {current_date})...")
        try:
            self.analysis = subject_analyzer.analyze(f"{self.current_query} (as of {current_date})")
            logging.info("Subject analysis successful.")
        except Exception as e:
            logging.error(f"Subject analysis failed: {e}")
            raise e
        self.console.print("\nAgent's Understanding:")
        self.console.print(f"Patient Query: {self.original_query}")
        self.console.print(f"Identified Medical Issue: {self.analysis.get('main_subject', 'Unknown Issue')}")
        temporal = self.analysis.get("temporal_context", {})
        if temporal:
            for key, value in temporal.items():
                self.console.print(f"- {key.capitalize()}: {value}")
        else:
            self.console.print("No temporal context provided.")
        needs = self.analysis.get("What_needs_to_be_researched", [])
        self.console.print("Key aspects to investigate:")
        self.console.print(f"{', '.join(needs) if needs else 'None'}")
        self.console.print("")

    async def search_and_extract(self, search_service, extractor, omit_urls, search_depth, search_breadth):
        topics = [self.analysis.get("main_subject", self.current_query)]
        topics += self.analysis.get("What_needs_to_be_researched", [])
        for topic in topics:
            if not topic:
                continue
            self.console.print(f"Searching for medical information on: {topic}")
            detailed_query = f"{topic} medical diagnosis (Depth: {search_depth}, Breadth: {search_breadth})"
            self.console.print(f"Executing search for: {detailed_query}")
            try:
                response = search_service.search_subject(
                    topic, "medical", search_depth=search_depth, results=search_breadth
                )
                results = response.get("results", [])
                results = [res for res in results if res.get("url") and not any(
                    omit.lower() in res.get("url").lower() for omit in omit_urls)]
                self.search_results[topic] = results
                for res in results:
                    relevance = res.get("score", "N/A")
                    self.console.print(f"Found: {res.get('title', 'No Title')} (Relevance: {relevance})")
            except Exception as e:
                self.console.print(f"[red]Search failed for {topic}: {e}[/red]")
                self.search_results[topic] = []
            self.console.print(f"Extracting content for: {topic}...")
            try:
                urls = [res.get("url") for res in self.search_results[topic] if res.get("url")]
                extraction_response = extractor.extract(
                    urls=urls,
                    extract_depth="advanced",
                    include_images=False
                )
                extracted = extraction_response.get("results", [])
                for item in extracted:
                    if "text" in item and item["text"]:
                        text = item["text"]
                        if len(text) > 300:
                            item["text"] = text[150:len(text)-150]
                    if "raw_content" in item and item["raw_content"]:
                        raw_content = item["raw_content"]
                        if len(raw_content) > 300:
                            item["raw_content"] = raw_content[150:len(raw_content)-150]
                self.extracted_content[topic] = extracted
                failed = [res for res in extracted if res.get("error")]
                if failed:
                    self.console.print(f"Warning: Failed to extract {len(failed)} URLs")
            except Exception as e:
                self.console.print(f"[red]Extraction failed for {topic}: {e}[/red]")

    async def analyze_full_content_rag(self, deepseek_client):
        self.console.print("Aggregating extracted content for comprehensive diagnostic analysis...")
        full_content = ""
        citations = []
        for topic, items in self.extracted_content.items():
            for item in items:
                url = item.get("url", "No URL")
                title = item.get("title", "No Title")
                content = item.get("text") or item.get("raw_content", "")
                full_content += f"\n\n=== Content from {url} ===\n{content}\n"
                citations.append(f"{title}: {url}")
        def chunk_text(text, chunk_size=1200):
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        chunks = chunk_text(full_content, chunk_size=1000)
        self.console.print(f"Total chunks generated: {len(chunks)}")
        semaphore = asyncio.Semaphore(20)
        async def get_embedding(text):
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            for attempt in range(3):
                try:
                    async with semaphore:
                        response = await openai.Embedding.acreate(
                            model="text-embedding-3-small",
                            input=text
                        )
                    return response["data"][0]["embedding"]
                except Exception as e:
                    self.console.print(f"[yellow]Embedding error (attempt {attempt+1}): {e}[/yellow]")
                    await asyncio.sleep(2 ** attempt)
            raise Exception("Failed to obtain embedding after 3 attempts.")
        chunk_embeddings = await asyncio.gather(*[get_embedding(chunk) for chunk in chunks])
        from supabase import create_client, Client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        try:
            supabase.table("embeddings").delete().eq("source", "Aggregated content").execute()
            self.console.print("Cleared previous embeddings.")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not clear previous embeddings: {e}[/yellow]")
        data_to_insert = []
        for i, chunk in enumerate(chunks):
            data_to_insert.append({
                "chunk": chunk,
                "embedding": chunk_embeddings[i],
                "source": "Aggregated content"
            })
        batch_size = 200
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i+batch_size]
            supabase.table("embeddings").upsert(batch).execute()
            self.console.print(f"Inserted batch {i // batch_size + 1} of {((len(data_to_insert)-1)//batch_size)+1}.")
        # Revised prompt: instruct the LLM to answer the patient's main query directly.
        summarization_prompt = f"""Generate a comprehensive diagnostic report that directly addresses the patient's query.
Patient Query: {self.current_query}
Based on the aggregated content below, provide a detailed medical analysis with actionable recommendations.
Include specific answers to the query and cite each source.

Aggregated Relevant Content:
"""
        # Obtain embedding for summarization prompt
        query_embedding = await get_embedding(summarization_prompt)
        USE_FAISS = os.getenv("USE_FAISS", "False").lower() == "true"
        if USE_FAISS:
            # Use FAISS for matching
            faiss_index = build_faiss_index(chunk_embeddings)
            if faiss_index is not None:
                k = min(155, len(chunks))
                matched_chunks = search_faiss(faiss_index, query_embedding, chunks, k)
            else:
                self.console.print("[yellow]FAISS index could not be built; falling back to Supabase.[/yellow]")
                match_response = supabase.rpc("match_chunks", {"query_embedding": query_embedding, "match_count": len(chunks)}).execute()
                matched_chunks = [row["chunk"] for row in match_response.data] if match_response.data else []
        else:
            match_response = supabase.rpc("match_chunks", {"query_embedding": query_embedding, "match_count": 155}).execute()
            matched_chunks = [row["chunk"] for row in match_response.data] if match_response.data else []
        self.console.print(f"Retrieved {len(matched_chunks)} relevant chunks.")
        aggregated_relevant = "\n\n".join(matched_chunks)
        prompt = f"""You are an expert diagnostic report generator.
Based on the following aggregated content, generate a comprehensive diagnostic report that directly addresses the patient's query:
"{self.current_query}"
Provide detailed medical analysis, actionable recommendations, and include citations for each source.

Aggregated Relevant Content:
{aggregated_relevant}

Citations:
{chr(10).join(citations)}

Respond with a detailed Markdown-formatted report.
"""
        messages = [
            {"role": "system", "content": "You are an expert diagnostic report generator."},
            {"role": "user", "content": prompt}
        ]
        self.console.print("Performing secondary analysis via RAG...")
        try:
            response = deepseek_client.chat(messages)
            comprehensive_report = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                supabase.table("embeddings").delete().eq("source", "Aggregated content").execute()
                self.console.print("[green]Cleared Supabase embeddings after report generation.[/green]")
            except Exception as e:
                self.console.print(f"[red]Error cleaning up Supabase: {e}[/red]")
            return comprehensive_report, citations
        except Exception as e:
            self.console.print(f"[red]Secondary analysis via RAG failed: {e}[/red]")
            try:
                supabase.table("embeddings").delete().eq("source", "Aggregated content").execute()
                self.console.print("[green]Cleared Supabase embeddings after failed attempt.[/green]")
            except Exception as cleanup_err:
                self.console.print(f"[red]Error during cleanup: {cleanup_err}[/red]")
            return "Secondary analysis failed.", citations

    def generate_report(self, additional_files, comprehensive_report, citations):
        report = f"# Medical Diagnostic Report for: {self.original_query} (Generated on {datetime.today().strftime('%Y-%m-%d')})\n\n"
        report += f"**Refined Query:** {self.current_query}\n\n"
        report += "## Agent's Understanding\n"
        for key, value in self.analysis.items():
            report += f"- **{key.capitalize()}**: {value}\n"
        report += "\n## Search Results\n"
        for topic, results in self.search_results.items():
            report += f"### Search: {topic}\n"
            if results:
                for res in results:
                    title = res.get("title", "No Title")
                    url = res.get("url", "No URL")
                    relevance = res.get("score", "N/A")
                    report += f"- **Title:** {title}\n  - **URL:** {url}\n  - **Relevance Score:** {relevance}\n\n"
            else:
                report += "No search results found.\n\n"
        report += "## Extracted Full Content\n"
        for topic, items in self.extracted_content.items():
            report += f"### Extraction for: {topic}\n"
            if items:
                for item in items:
                    ext_url = item.get("url", "No URL")
                    text = item.get("text") or item.get("raw_content", "")
                    report += f"- **URL:** {ext_url}\n  - **Content:**\n{text}\n\n"
            else:
                report += "No content extracted.\n\n"
        if additional_files:
            report += "## Additional Reference Files\n"
            for path, content in additional_files.items():
                report += f"- **{path}**:\n{content}\n\n"
        wearable_summary = read_wearable_data()
        if wearable_summary:
            report += "## Wearable Data Summary\n"
            report += wearable_summary + "\n"
        report += "\n---\n\n"
        report += "## Comprehensive Diagnostic Report (via Secondary Analysis)\n"
        report += comprehensive_report + "\n\n"
        report += "## Citations\n"
        for citation in citations:
            report += f"- {citation}\n"
        return report

    def generate_summary_report(self, comprehensive_report, citations):
        summary = f"# Summary Diagnostic Report for: {self.original_query} (Generated on {datetime.today().strftime('%Y-%m-%d')})\n\n"
        summary += f"**Refined Query:** {self.current_query}\n\n"
        summary += "## Comprehensive Diagnostic Report\n"
        summary += comprehensive_report + "\n\n"
        summary += "## Citations\n"
        for citation in citations:
            summary += f"- {citation}\n"
        return summary

    async def generate_patient_summary_report(self, deepseek_client, comprehensive_report, citations):
        self.console.print("Generating patient-friendly summary report with clear action steps...")
        prompt = f"""You are a medical assistant who explains complex diagnostic reports in simple, clear language for patients.
Based on the following comprehensive diagnostic report, produce a short summary that a non-medical professional can easily understand.
This summary must:
- Clearly state the main findings.
- Provide specific, actionable recommendations (including any suggested medications, diagnostic tests, and referrals).
- Tell the patient exactly what to do next.
Comprehensive Diagnostic Report:
{comprehensive_report}
"""
        messages = [
            {"role": "system", "content": "You are a patient-friendly medical summarization assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = deepseek_client.chat(messages)
            simplified_report = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return simplified_report
        except Exception as e:
            self.console.print(f"[red]Failed to generate patient-friendly summary: {e}[/red]")
            return "Patient-friendly summary generation failed."

    def save_reports(self, additional_files, comprehensive_report, citations):
        full_report = self.generate_report(additional_files, comprehensive_report, citations)
        summary_report = self.generate_summary_report(comprehensive_report, citations)
        base_filename = ''.join(c if c.isalnum() or c.isspace() else '_' for c in self.original_query.split('.')[0])
        full_report_filename = f"{base_filename}_diagnostic_full.md"
        try:
            with open(full_report_filename, "w", encoding="utf-8") as f:
                f.write(full_report)
            self.console.print(f"\n[bold green]Full diagnostic report saved to: {full_report_filename}[/bold green]")
            logging.info(f"Full report saved to {full_report_filename}")
        except Exception as ex:
            self.console.print(f"[red]Failed to save full report: {ex}[/red]")
            logging.error(f"Full report save error: {ex}")
        summary_report_filename = f"{base_filename}_diagnostic_summary.md"
        try:
            with open(summary_report_filename, "w", encoding="utf-8") as f:
                f.write(summary_report)
            self.console.print(f"[bold green]Summary diagnostic report saved to: {summary_report_filename}[/bold green]")
            logging.info(f"Summary diagnostic report saved to {summary_report_filename}")
        except Exception as ex:
            self.console.print(f"[red]Failed to save summary report: {ex}[/red]")
            logging.error(f"Summary report save error: {ex}")
        return full_report_filename, summary_report_filename

    def save_patient_report(self, patient_report):
        base_filename = ''.join(c if c.isalnum() or c.isspace() else '_' for c in self.original_query.split('.')[0])
        patient_report_filename = f"{base_filename}_patient_summary.md"
        try:
            with open(patient_report_filename, "w", encoding="utf-8") as f:
                f.write(patient_report)
            self.console.print(f"[bold green]Patient-friendly summary saved to: {patient_report_filename}[/bold green]")
            logging.info(f"Patient report saved to {patient_report_filename}")
        except Exception as ex:
            self.console.print(f"[red]Failed to save patient-friendly summary: {ex}[/red]")
            logging.error(f"Patient report save error: {ex}")
        return patient_report_filename

async def main():
    load_dotenv()
    console = Console()
    current_date = datetime.today().strftime("%Y-%m-%d")
    
    # New: Analyze query sentiment, extract medical entities, and predict diagnosis.
    original_query = input("Describe your current medical symptoms or issue: ").strip()
    if not original_query:
        console.print("[red]No query provided. Exiting.[/red]")
        return

    sentiment_score = analyze_sentiment(original_query)
    entities = extract_medical_entities(original_query)
    console.print(f"\nSentiment Analysis: {sentiment_score}")
    console.print(f"Extracted Medical Entities: {entities}\n")
    
    vectorizer, clf_model = train_symptom_classifier()
    predicted_diag, diag_proba = predict_diagnosis(original_query, vectorizer, clf_model)
    console.print(f"Predicted Diagnosis: {predicted_diag} with probabilities: {diag_proba}\n")
    explanation = explain_diagnosis(original_query, vectorizer, clf_model)
    console.print(f"Diagnosis Explanation:\n{explanation}\n")
    
    # Save query history for trend analysis.
    save_query_history(original_query, predicted_diag, sentiment_score, entities)
    
    search_config = SearchConfig()
    analysis_config = AnalysisConfig(
        model_name=os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1"),
        temperature= 0.3
    )
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    deepseek_client = DeepSeekClient(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, config=analysis_config)
    search_client = TavilyClient(api_key=TAVILY_API_KEY)
    extractor = TavilyExtractor(api_key=TAVILY_API_KEY)
    search_service = WebSearchService(search_client, search_config)
    subject_analyzer = SubjectAnalyzer(llm_client=deepseek_client, config=analysis_config)
    
    include_urls_input = input("Enter URL(s) to include in the search (comma-separated, leave blank to search the web): ").strip()
    include_urls = [url.strip() for url in include_urls_input.split(',')] if include_urls_input else []
    omit_urls_input = input("Enter URL(s) to omit from search results (comma-separated, leave blank if none): ").strip()
    omit_urls = [url.strip() for url in omit_urls_input.split(',')] if omit_urls_input else []
    own_files_input = input("Enter local file path(s) as additional reference (comma-separated, leave blank if none): ").strip()
    own_files = [f.strip() for f in own_files_input.split(',')] if own_files_input else []
    additional_files = {}
    for file_path in own_files:
        if os.path.exists(file_path):
            content = extract_text_from_file(file_path, console)
            if content:
                additional_files[file_path] = content
        else:
            console.print(f"[yellow]Warning: File {file_path} not found.[/yellow]")
    
    search_depth = input("Enter search depth (e.g., 'basic', 'advanced'; default is 'advanced'): ").strip() or "advanced"
    search_breadth_input = input("Enter search breadth (number of results per query; default is 10): ").strip()
    try:
        search_breadth = int(search_breadth_input) if search_breadth_input else 10
    except ValueError:
        search_breadth = 10

    task = MedicalTask(original_query, console)
    
    # Feedback loop for subject analysis.
    while True:
        await task.analyze(subject_analyzer, current_date)
        feedback = input("Provide feedback on the agent's analysis (leave blank if correct): ").strip()
        if feedback:
            task.update_feedback(feedback)
            task.current_query = f"{original_query} - Additional context: {feedback}"
            console.print("[yellow]Reanalyzing query with your feedback...[/yellow]")
        else:
            break

    if not include_urls:
        await task.search_and_extract(search_service, extractor, omit_urls, search_depth, search_breadth)
    else:
        filtered_urls = [url for url in include_urls if not any(omit.lower() in url.lower() for omit in omit_urls)]
        task.search_results = {"User Provided": [{"title": "User Provided", "url": url, "content": ""} for url in filtered_urls]}
        console.print(f"[green]Using user provided URLs: {filtered_urls}[/green]")
        try:
            extraction_response = extractor.extract(
                urls=filtered_urls,
                extract_depth="advanced",
                include_images=False
            )
            task.extracted_content["User Provided"] = extraction_response.get("results", [])
        except Exception as e:
            console.print(f"[red]Extraction failed for user provided URLs: {e}[/red]")
    
    comprehensive_report, citations = await task.analyze_full_content_rag(deepseek_client)
    full_report_filename, summary_report_filename = task.save_reports(additional_files, comprehensive_report, citations)
    console.print(f"\n[bold green]Diagnostic process complete. Full and summary reports have been saved.[/bold green]")
    
    patient_report = await task.generate_patient_summary_report(deepseek_client, comprehensive_report, citations)
    patient_report_filename = task.save_patient_report(patient_report)
    console.print(f"\n[bold green]Patient-friendly summary report saved as: {patient_report_filename}[/bold green]")
    
    # Ask the user if they want to view visualizations of past query trends.
    viz_choice = input("Would you like to generate visualizations for query trends? (y/n): ").strip().lower()
    if viz_choice == 'y':
        visualize_query_trends()

if __name__ == "__main__":
    asyncio.run(main())
