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
from subject_analyzer.src.services.gemini_client import GeminiClient # <--- Changed line
from subject_analyzer.src.models.analysis_models import AnalysisConfig
# ==============================
# New Data Science / Analytics & Visualization Functions
# ==============================

def train_symptom_classifier():
    """
    Trains a simple symptom-to-diagnosis classifier using the survey data.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import pandas as pd # Import pandas

    # Load data from the survey report
    try:
        df = pd.read_csv("Survey report_Sheet1.csv")
        # Extract symptoms and medical history
        # Ensure these column names match exactly your CSV headers
        symptoms = df['What are the current symptoms or health issues you are facing'].fillna("").tolist()
        # Use Medical Health History as labels, handle potential missing values and multiple conditions
        labels = df['Medical Health History'].fillna("None").tolist()

        # You might need more sophisticated processing if Medical Health History contains multiple conditions
        # For a simple classifier, you could use the first mentioned condition or treat each as a separate label
        # Example: Splitting multiple conditions
        processed_labels = [label.split(',')[0].strip() if isinstance(label, str) else 'None' for label in labels]

        # Filter out entries with no symptoms or no relevant medical history if desired
        # Example: combined_data = [(s, l) for s, l in zip(symptoms, processed_labels) if s and l != 'None']
        # if not combined_data:
        #     print("Warning: No valid data for training found in the survey report.")
        #     return None, None
        # texts, labels = zip(*combined_data)

        texts = symptoms
        labels = processed_labels

        if not texts or not labels:
             print("Warning: No data loaded from survey report for training.")
             return None, None


    except FileNotFoundError:
        print("Error: Survey report_Sheet1.txt not found.")
        # Fallback to dummy data or handle appropriately
        data = [
            ("fever cough sore throat", "Flu"),
            ("headache nausea sensitivity to light", "Migraine"),
            ("chest pain shortness of breath", "Heart Attack"),
            ("joint pain stiffness", "Arthritis"),
            ("abdominal pain diarrhea vomiting", "Gastroenteritis")
        ]
        texts, labels = zip(*data)
    except KeyError as e:
        print(f"Error: Missing expected column in Survey report_Sheet1.txt: {e}")
        # Fallback to dummy data or handle appropriately
        data = [
            ("fever cough sore throat", "Flu"),
            ("headache nausea sensitivity to light", "Migraine"),
            ("chest pain shortness of breath", "Heart Attack"),
            ("joint pain stiffness", "Arthritis"),
            ("abdominal pain diarrhea vomiting", "Gastroenteritis")
        ]
        texts, labels = zip(*data)
    except Exception as e:
        print(f"An error occurred while reading the survey report: {e}")
        # Fallback to dummy data or handle appropriately
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

import os
import asyncio
import logging
from datetime import datetime
from rich.console import Console
from typing import Dict, List # Ensure List and Dict are imported if not at top level
# Import types for google-genai if needed within methods, or ensure it's at top level
try:
    from google.genai import types
except ImportError:
    # Handle the case where google-genai is not installed or types are not available
    # You might want to log a warning or raise an error here
    types = None
    logging.warning("Could not import google.genai.types. Embedding functionality may be affected.")


# Assume these functions are defined elsewhere in your med_agent_new.py
# from .your_other_modules import read_wearable_data, build_faiss_index

# Placeholder implementations if they are not provided elsewhere
def read_wearable_data():
    """Placeholder for reading wearable device data."""
    # Implement your logic to read and summarize wearable data
    return "Wearable data summary: [Data not available in this example]"

def build_faiss_index(embeddings):
    """Placeholder for building FAISS index."""
    # Implement your FAISS index building logic
    # Requires faiss-cpu and numpy
    try:
        import faiss
        import numpy as np
        if not embeddings:
            return None
        # Assuming embeddings are numpy arrays or can be converted
        embedding_dim = len(embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(embeddings).astype('float32'))
        return index
    except ImportError:
        logging.error("FAISS or Numpy not installed. Cannot build FAISS index.")
        return None
    except Exception as e:
        logging.error(f"Error building FAISS index: {e}")
        return None


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
            # The subject_analyzer uses the LLM client (now GeminiClient)
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
                            # Keep a central portion - adjust indices as needed
                            start_index = max(0, len(text) // 2 - 150)
                            end_index = min(len(text), len(text) // 2 + 150)
                            item["text"] = text[start_index:end_index]
                    if "raw_content" in item and item["raw_content"]:
                         raw_content = item["raw_content"]
                         if len(raw_content) > 300:
                            # Keep a central portion - adjust indices as needed
                             start_index = max(0, len(raw_content) // 2 - 150)
                             end_index = min(len(raw_content), len(raw_content) // 2 + 150)
                             item["raw_content"] = raw_content[start_index:end_index]
                self.extracted_content[topic] = extracted
                failed = [res for res in extracted if res.get("error")]
                if failed:
                    self.console.print(f"Warning: Failed to extract {len(failed)} URLs")
            except Exception as e:
                self.console.print(f"[red]Extraction failed for {topic}: {e}[/red]")


    # Modified analyze_full_content_rag to use the gemini_client for embeddings
    async def analyze_full_content_rag(self, gemini_client): # Argument name changed for clarity
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

        def chunk_text(text, chunk_size=765):
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        chunks = chunk_text(full_content, chunk_size=765)
        self.console.print(f"Total chunks generated: {len(chunks)}")

        # === Modified Embedding Section to use Google Embeddings ===
        # Removed OpenAI import and API key setting here

        # Changed get_embedding to a synchronous function
        def get_embedding(text):
            # Use the gemini_client (which uses google-generativeai) for embeddings
            # The embedding model name might be different, 'embedding-001' or similar is common
            GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004") # Use a Google embedding model

            # Removed attempt loop here for simplicity in demonstrating the fix,
            # but you might want to re-add retry logic within this synchronous function
            try:
                # Use the embed_content method from the google-generativeai client
                # Assuming the gemini_client instance passed to this method has the .client attribute
                # Ensure types is accessible (imported at top or locally if needed)
                if types is None:
                     raise ImportError("google.genai.types not available. Cannot perform embedding.")

                embedding_response = gemini_client.client.models.embed_content( # Removed await
                    model=GOOGLE_EMBEDDING_MODEL,
                    contents=[types.Part.from_text(text=text)] # Corrected line: pass text as keyword argument
                )
                # The structure of the embedding response from google-generativeai might differ from OpenAI.
                # Based on the documentation and common patterns, the embedding vector might be in a 'embedding' attribute
                # or nested within a structure like response.embeddings[0].values.
                # We need to inspect the actual response structure if this doesn't work.
                # Assuming a structure like response.embedding.values for now, common in some Google APIs.
                # **NOTE:** You might still need to adjust how the embedding is extracted based on the actual response structure.
                embedding_vector = embedding_response.embeddings[0].values # Access the first embedding's values

                return embedding_vector
            except Exception as e:
                # Log the error or handle it as needed within the synchronous function
                self.console.print(f"[yellow]Embedding error: {e}[/yellow]")
                # Re-raise the exception if you want it to be caught by the executor
                raise

        # === End Modified Embedding Section ===

        # Get the current running loop
        loop = asyncio.get_event_loop()

        # Run synchronous get_embedding calls in a thread pool executor
        chunk_embeddings = await asyncio.gather(*[loop.run_in_executor(None, get_embedding, chunk) for chunk in chunks])


        # Ensure numpy is imported for FAISS if needed
        try:
            import numpy as np
        except ImportError:
             self.console.print("[red]Numpy is required for FAISS. Install with 'pip install numpy'.[/red]")
             raise
        except Exception as e:
             self.console.print(f"[red]Error importing numpy: {e}[/red]")
             raise


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
            # Ensure embedding_vector is a list or compatible format for Supabase
            embedding_vector = chunk_embeddings[i]
            if isinstance(embedding_vector, (list, tuple)):
                 # If it's already a list/tuple, use directly
                 pass
            elif hasattr(embedding_vector, 'tolist'):
                 # If it's a numpy array or similar, convert to list
                 embedding_vector = embedding_vector.tolist()
            else:
                 # Attempt a direct conversion if needed
                 try:
                     embedding_vector = list(embedding_vector) # Adjust if necessary based on actual type
                 except TypeError:
                     self.console.print(f"[red]Error converting embedding vector to list for chunk {i}. Type: {type(embedding_vector)}[/red]")
                     # Decide how to handle chunks that fail embedding conversion
                     continue # Skip inserting this chunk

            data_to_insert.append({
                "chunk": chunk,
                "embedding": embedding_vector, # Use the Google embedding vector
                "source": "Aggregated content"
            })

        batch_size = 200
        # Use upsert with on_conflict for robustness
        # on_conflict_keys = "chunk" # Assuming chunk is unique or you have another unique key
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i+batch_size]
            try:
                supabase.table("embeddings").insert(batch).execute()
                self.console.print(f"Inserted batch {i // batch_size + 1} of {((len(data_to_insert) + batch_size - 1) // batch_size)}.") # Corrected total batch calculation
            except Exception as e:
                 self.console.print(f"[red]Error inserting batch {i // batch_size + 1}: {e}[/red]")
                 # Decide whether to continue or stop on batch insertion errors

        # Revised prompt: instruct the LLM to answer the patient's main query directly.
        summarization_prompt = f"""Generate a comprehensive diagnostic report that directly addresses the patient's query.
Patient Query: {self.current_query}
Based on the aggregated content below, provide a detailed medical analysis with actionable recommendations.
Include specific answers to the query and cite each source.

Aggregated Relevant Content:
"""
        # Obtain embedding for summarization prompt using the updated get_embedding function
        # No need for await here as get_embedding is synchronous
        # Ensure get_embedding doesn't return None or raise an unexpected error
        try:
            query_embedding = get_embedding(summarization_prompt) # This now uses the Google embedding model
        except Exception as e:
            self.console.print(f"[red]Failed to get embedding for summarization prompt: {e}[/red]")
            # Decide how to proceed if query embedding fails - likely cannot perform RAG
            return "Error: Could not generate query embedding for RAG.", citations


        USE_FAISS = os.getenv("USE_FAISS", "False").lower() == "true"
        if USE_FAISS:
            # Use FAISS for matching
            # build_faiss_index requires numpy
            faiss_index = build_faiss_index(chunk_embeddings)
            if faiss_index is not None:
                k = min(155, len(chunks))
                # Ensure query_embedding is in the correct format for FAISS search (numpy array float32)
                try:
                    query_vec = np.array(query_embedding).reshape(1, -1).astype('float32')
                    distances, indices = faiss_index.search(query_vec, k)
                    matched_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
                except Exception as e:
                     self.console.print(f"[red]Error during FAISS search: {e}[/red]")
                     self.console.print("[yellow]Falling back to Supabase for RAG.[/yellow]")
                     # Ensure query_embedding is a list for Supabase RPC
                     match_response = supabase.rpc("match_chunks", {"query_embedding": list(query_embedding), "match_count": 155}).execute() # Increased match_count for fallback
                     matched_chunks = [row["chunk"] for row in match_response.data] if match_response.data else []
            else:
                self.console.print("[yellow]FAISS index could not be built; falling back to Supabase.[/yellow]")
                # Ensure query_embedding is a list for Supabase RPC
                match_response = supabase.rpc("match_chunks", {"query_embedding": list(query_embedding), "match_count": 155}).execute()
                matched_chunks = [row["chunk"] for row in match_response.data] if match_response.data else []
        else:
            # Ensure query_embedding is a list for Supabase RPC
            # Use Supabase RPC for matching
            match_response = supabase.rpc("match_chunks", {"query_embedding": list(query_embedding), "match_count": 200}).execute()
            matched_chunks = [row["chunk"] for row in match_response.data] if match_response.data else []

        self.console.print(f"Retrieved {len(matched_chunks)} relevant chunks.")
        aggregated_relevant = "\n\n".join(matched_chunks)

        prompt = f"""You are an expert diagnostic report generator. Based on the following aggregated content, generate a comprehensive diagnostic report that directly addresses the patient's query:
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
            # Use the gemini_client (LLM client) for the final report generation
            response = gemini_client.chat(messages) # This calls the chat method of the GeminiClient
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
            # Handle potentially non-string values in analysis
            report += f"- **{key.capitalize()}**: {str(value) if not isinstance(value, (list, dict)) else value}\n"

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

        wearable_summary = read_wearable_data() # Assuming read_wearable_data is defined elsewhere
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

    async def generate_patient_summary_report(self, gemini_client, comprehensive_report, citations): # Argument name changed
        self.console.print("Generating patient-friendly summary report with clear action steps...")
        prompt = f"""You are a medical assistant who explains complex diagnostic reports in simple, clear language for patients. Based on the following comprehensive diagnostic report, produce a short summary that a non-medical professional can easily understand.
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
            # Use the gemini_client (LLM client) for patient summary generation
            response = gemini_client.chat(messages) # This calls the chat method of the GeminiClient
            simplified_report = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return simplified_report
        except Exception as e:
            self.console.print(f"[red]Failed to generate patient-friendly summary: {e}[/red]")
            return "Patient-friendly summary generation failed."


    def save_reports(self, additional_files, comprehensive_report, citations):
        full_report = self.generate_report(additional_files, comprehensive_report, citations)
        summary_report = self.generate_summary_report(comprehensive_report, citations)

        # Sanitize original_query for filename
        base_filename = ''.join(c if c.isalnum() or c.isspace() else '_' for c in self.original_query.split('.')[0])
        base_filename = base_filename.strip().replace(' ', '_') # Replace spaces with underscores and strip
        if not base_filename:
             base_filename = "report" # Default name if query is empty or invalid chars


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
        # Sanitize original_query for filename
        base_filename = ''.join(c if c.isalnum() or c.isspace() else '_' for c in self.original_query.split('.')[0])
        base_filename = base_filename.strip().replace(' ', '_') # Replace spaces with underscores and strip
        if not base_filename:
             base_filename = "report" # Default name if query is empty or invalid chars

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
    # Use the Gemini 2.5 Pro model name. Based on search, "gemini-2.5-pro-exp-03-25" is a possible model ID. or "gemini-2.5-flash-exp-04-17"
    # Confirm the exact model name from the Gemini API documentation you are using (e.g., Google AI Studio or Vertex AI).
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-exp-04-17") # <--- Changed line
    analysis_config = AnalysisConfig(
        model_name=GEMINI_MODEL_NAME, # <--- Changed line
        temperature= 0.3
    )

    # Obtain Gemini API Key and Base URL from environment variables
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # The base URL depends on whether you are using Google AI Studio API or Vertex AI API.
    # Example for Google AI Studio: "https://generativelanguage.googleapis.com"
    # Example for Vertex AI: "https://us-central1-aiplatform.googleapis.com" (adjust region as needed)
    # **IMPORTANT:** Set the correct base URL based on your Gemini API access method.
    # GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com") # <--- Changed line

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Instantiate the GeminiClient instead of DeepSeekClient
    gemini_client = GeminiClient(api_key=GEMINI_API_KEY,  config=analysis_config) # <--- Changed line

    search_client = TavilyClient(api_key=TAVILY_API_KEY)
    extractor = TavilyExtractor(api_key=TAVILY_API_KEY)
    search_service = WebSearchService(search_client, search_config)

    # Update the subject_analyzer to use the gemini_client
    subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config) # <--- Changed line

    # Continue using gemini_client for RAG and patient summary generation
    # task.analyze_full_content_rag(deepseek_client) should be changed to task.analyze_full_content_rag(gemini_client)
    # task.generate_patient_summary_report(deepseek_client, ...) should be changed to task.generate_patient_summary_report(gemini_client, ...)
    # These calls are further down in the main function and should be updated accordingly.

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
    
    comprehensive_report, citations = await task.analyze_full_content_rag(gemini_client)
    full_report_filename, summary_report_filename = task.save_reports(additional_files, comprehensive_report, citations)
    console.print(f"\n[bold green]Diagnostic process complete. Full and summary reports have been saved.[/bold green]")
    
    patient_report = await task.generate_patient_summary_report(gemini_client, comprehensive_report, citations)
    patient_report_filename = task.save_patient_report(patient_report)
    console.print(f"\n[bold green]Patient-friendly summary report saved as: {patient_report_filename}[/bold green]")
    
    # Ask the user if they want to view visualizations of past query trends.
    viz_choice = input("Would you like to generate visualizations for query trends? (y/n): ").strip().lower()
    if viz_choice == 'y':
        visualize_query_trends()

if __name__ == "__main__":
    asyncio.run(main())
