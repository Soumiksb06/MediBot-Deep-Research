#!/usr/bin/env python3
"""
Streamlit Application for the Medical Health Help System.

Features:
- Interactive Patient Engagement via Streamlit UI.
- Subject Analysis using Gemini.
- Web Content Extraction via Tavily.
- RAG with Supabase/FAISS using Gemini Embeddings.
- Comprehensive Diagnostic Reporting (Full, Summary, Patient-Friendly) using Gemini.
- Sentiment Analysis and NER using Gemini (replaces NLTK/spaCy).
- ML-based Diagnosis Prediction (scikit-learn) & Explanation (SHAP) - Requires survey data upload.
- Visualization of Query Trends (Matplotlib/Seaborn).
- Optional Wearable Data Integration (CSV upload).
- Local File Reference Upload (PDF, DOCX, CSV, Excel, TXT).
- Asynchronous operations handled within Streamlit using asyncio.run().
- API Key management via st.secrets or environment variables.

Modified to display all analysis results persistently using tabs.
"""

import os
import sys
import asyncio
import logging
import csv
import io
import traceback
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from rich.console import Console
from rich.logging import RichHandler
import PyPDF2
import docx
import pandas as pd
from supabase import create_client, Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import shap
import faiss
import google.generativeai as genai
from google.generativeai import types as genai_types

# Assuming custom modules are correctly installed or in the same directory
try:
    from web_agent.src.services.web_search import WebSearchService
    from web_agent.src.models.search_models import SearchConfig
    from subject_analyzer.src.services.tavily_client import TavilyClient
    from subject_analyzer.src.services.tavily_extractor import TavilyExtractor
    from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer
    from subject_analyzer.src.services.gemini_client import GeminiClient
    from subject_analyzer.src.models.analysis_models import AnalysisConfig
except ImportError as e:
    logging.basicConfig(level=logging.ERROR, filename="streamlit_error.log", filemode="a")
    logger = logging.getLogger(__name__)
    logger.error(f"ImportError for custom modules: {e}\n{traceback.format_exc()}")
    st.error(f"Required custom agent modules (web_agent, subject_analyzer) not found: {e}. Please ensure they are installed or in the correct Python path.")
    st.stop()

# --- Configuration & Initialization ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="streamlit_medical_agent.log",
    filemode="a"
)
logger = logging.getLogger(__name__)

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide", page_title="Medical Diagnostic Agent")
st.title("🩺 Medical Diagnostic Agent")
st.markdown("Enter patient symptoms and configure options to generate a diagnostic report.")

# --- Load API Keys and Config ---
# Use st.secrets with fallback to environment variables for local development
GEMINI_API_KEY = st.secrets.get("api_keys", {}).get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = st.secrets.get("api_keys", {}).get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
SUPABASE_URL = st.secrets.get("supabase", {}).get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("supabase", {}).get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
GEMINI_MODEL_NAME = st.secrets.get("models", {}).get("GEMINI_MODEL_NAME") or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-04-17")
GOOGLE_EMBEDDING_MODEL = st.secrets.get("models", {}).get("GOOGLE_EMBEDDING_MODEL") or os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")

if not all([GEMINI_API_KEY, TAVILY_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    st.error("Missing required API keys or configurations. Please set them in Streamlit secrets or environment variables.")
    st.stop()

# --- Global Clients and Resources (Cached) ---

@st.cache_resource
def get_gemini_client():
    """Initializes and returns the Gemini client."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
        client_wrapper = GeminiClient(api_key=GEMINI_API_KEY, config=analysis_config)
        logger.info("Custom GeminiClient wrapper initialized successfully.")
        return client_wrapper
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        logger.error(f"Gemini client initialization failed: {e}\n{traceback.format_exc()}")
        st.stop()

@st.cache_resource
def get_supabase_client():
    """Initializes and returns the Supabase client."""
    try:
        client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        logger.error(f"Supabase client initialization failed: {e}\n{traceback.format_exc()}")
        st.stop()

@st.cache_resource
def get_tavily_clients():
    """Initializes and returns Tavily search and extractor clients."""
    try:
        search_client = TavilyClient(api_key=TAVILY_API_KEY)
        extractor = TavilyExtractor(api_key=TAVILY_API_KEY)
        search_config = SearchConfig()
        search_service = WebSearchService(search_client, search_config)
        logger.info("Tavily clients initialized successfully.")
        return search_service, extractor
    except Exception as e:
        st.error(f"Failed to initialize Tavily clients: {e}")
        logger.error(f"Tavily client initialization failed: {e}\n{traceback.format_exc()}")
        st.stop()

# Initialize clients
gemini_client = get_gemini_client()
supabase = get_supabase_client()
search_service, extractor = get_tavily_clients()

# --- Helper Functions ---

def safe_filename(query, suffix):
    """Creates a safe filename from a query string."""
    base = ''.join(c if c.isalnum() or c.isspace() else '_' for c in query[:50])
    base = base.strip().replace(' ', '_')
    return f"{base or 'report'}_{suffix}"

async def analyze_sentiment_gemini(query, client):
    """Performs sentiment analysis using Gemini."""
    prompt = f"""Analyze the sentiment of the following patient query. Respond with a simple classification (Positive, Negative, Neutral) and a brief explanation.

Patient Query: "{query}"

Sentiment Classification:
Explanation:"""
    try:
        messages = [{"role": "user", "content": prompt}]
        if asyncio.iscoroutinefunction(client.chat):
            response_data = await client.chat(messages)
        else:
            loop = asyncio.get_running_loop()
            response_data = await loop.run_in_executor(None, client.chat, messages)
        sentiment_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse sentiment response.")
        return sentiment_text
    except Exception as e:
        logger.error(f"Gemini sentiment analysis failed: {e}\n{traceback.format_exc()}")
        return f"Error during sentiment analysis: {e}"

async def extract_medical_entities_gemini(query, client):
    """Extracts medical entities using Gemini."""
    prompt = f"""Extract the key medical entities (like symptoms, conditions, medications mentioned) from the following patient query. List them clearly using bullet points.

Patient Query: "{query}"

Medical Entities:
- [Entity 1]
- [Entity 2]
..."""
    try:
        messages = [{"role": "user", "content": prompt}]
        if asyncio.iscoroutinefunction(client.chat):
            response_data = await client.chat(messages)
        else:
            loop = asyncio.get_running_loop()
            response_data = await loop.run_in_executor(None, client.chat, messages)
        entities_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse entities response.")
        return entities_text
    except Exception as e:
        logger.error(f"Gemini entity extraction failed: {e}\n{traceback.format_exc()}")
        return f"Error during entity extraction: {e}"

def extract_text_from_file(uploaded_file):
    """Extracts text from various uploaded file types with memory efficiency."""
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        bytes_data = uploaded_file.getvalue()
        if ext == '.pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            return text[:100000]  # Limit to prevent memory issues
        elif ext == '.docx':
            doc = docx.Document(io.BytesIO(bytes_data))
            return "\n".join(para.text for para in doc.paragraphs)[:100000]
        elif ext == '.csv':
            df = pd.read_csv(io.BytesIO(bytes_data), nrows=1000)  # Limit rows
            return df.to_csv(index=False)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(io.BytesIO(bytes_data), nrows=1000)
            return df.to_csv(index=False)
        elif ext == '.txt':
            return bytes_data.decode('utf-8')[:100000]
        else:
            return f"[Unsupported file type: {ext}]"
    except Exception as e:
        logger.error(f"Failed to extract text from {uploaded_file.name}: {e}\n{traceback.format_exc()}")
        return f"Error processing {uploaded_file.name}: {e}"

@st.cache_data
def train_symptom_classifier(survey_data_df):
    """Trains a symptom classifier from uploaded survey data with validation."""
    if survey_data_df is None or survey_data_df.empty:
        return None, None
    try:
        symptom_col = 'What are the current symptoms or health issues you are facing'
        history_col = 'Medical Health History'
        if symptom_col not in survey_data_df.columns or history_col not in survey_data_df.columns:
            st.error(f"Survey data must contain '{symptom_col}' and '{history_col}' columns. Please check the file format.")
            return None, None
        df = survey_data_df.copy()
        symptoms = df[symptom_col].astype(str).fillna("").tolist()
        labels = df[history_col].astype(str).fillna("None").tolist()
        processed_labels = [label.split(',')[0].strip() if isinstance(label, str) else 'None' for label in labels]
        valid_data = [(s, l) for s, l in zip(symptoms, processed_labels) if s.strip() and l != 'None']
        if not valid_data:
            st.warning("No valid symptom/diagnosis pairs found for training.")
            return None, None
        texts, final_labels = zip(*valid_data)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=2)
        X = vectorizer.fit_transform(texts)
        if not vectorizer.vocabulary_:
            st.warning("Insufficient vocabulary for training. Please provide more detailed survey data.")
            return None, None
        model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
        model.fit(X, final_labels)
        logger.info("Symptom classifier trained successfully.")
        return vectorizer, model
    except Exception as e:
        st.error(f"Error training classifier: {e}")
        logger.error(f"Classifier training failed: {e}\n{traceback.format_exc()}")
        return None, None

def predict_diagnosis(query, vectorizer, model):
    """Predicts diagnosis using the trained model."""
    if not vectorizer or not model:
        return "Model not trained", {}
    try:
        X_query = vectorizer.transform([query])
        pred = model.predict(X_query)[0]
        proba = model.predict_proba(X_query)[0]
        prob_dict = dict(zip(model.classes_, proba))
        sorted_proba = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        return pred, sorted_proba
    except Exception as e:
        logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
        return f"Prediction error: {e}", {}

def explain_diagnosis_shap(query, vectorizer, model, background_texts=None):
    """Generates SHAP explanation for the prediction."""
    if not vectorizer or not model:
        return "Model not trained, cannot explain."
    try:
        X_query = vectorizer.transform([query]).toarray()
        if X_query.nnz == 0:
            return "Explanation not available (query contains no known terms)."
        predict_proba_dense = lambda x: model.predict_proba(x)
        if background_texts and len(background_texts) > 10:
            background_data_sparse = vectorizer.transform(background_texts)
            num_clusters = min(10, background_data_sparse.shape[0])
            # Use a smaller sample for background data if needed to prevent memory issues
            if background_data_sparse.shape[0] > 100:
                 background_data_sparse = background_data_sparse[:100]
            background_summary = shap.kmeans(background_data_sparse.toarray(), num_clusters)
        else:
            # Use a small zero array if no suitable background texts are available
            background_summary = np.zeros((1, X_query.shape[1]))
        explainer = shap.KernelExplainer(predict_proba_dense, background_summary)
        shap_values = explainer.shap_values(X_query, nsamples=100) # Limit nsamples for performance
        predicted_class = model.predict(X_query)[0]
        predicted_class_index = list(model.classes_).index(predicted_class)
        shap_values_instance = shap_values[predicted_class_index][0]
        feature_names = vectorizer.get_feature_names_out()
        # Ensure indices are within bounds
        sorted_indices = np.argsort(np.abs(shap_values_instance))
        sorted_indices = sorted_indices[sorted_indices < len(feature_names)][::-1]

        explanation = f"Top contributing features for '{predicted_class}' (SHAP values):\n"
        count = 0
        for idx in sorted_indices:
            if count >= 5: # Limit to top 5
                break
            if not np.isclose(shap_values_instance[idx], 0):
                explanation += f"- {feature_names[idx]}: {shap_values_instance[idx]:.4f}\n"
                count += 1
        return explanation if "Top" in explanation else "No significant features identified."
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}\n{traceback.format_exc()}")
        return f"SHAP explanation failed: {e}"


def save_query_history(query, diagnosis, sentiment, entities):
    """Saves query details to a CSV file."""
    filename = "query_history.csv"
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writerow(["timestamp", "query", "predicted_diagnosis", "sentiment_summary", "entities_summary"])
            # Limit the length of sentiment and entities summaries for CSV
            sentiment_summary = str(sentiment).split('\n')[0][:200] if sentiment else "N/A"
            entities_summary = str(entities).split('\n')[0][:200] if entities else "N/A"
            writer.writerow([datetime.now().isoformat(), query, diagnosis, sentiment_summary, entities_summary])
        logger.info("Query history saved successfully.")
        return True
    except Exception as e:
        st.warning(f"Could not save query history: {e}")
        logger.error(f"Query history save failed: {e}\n{traceback.format_exc()}")
        return False

def visualize_query_trends():
    """Generates and returns matplotlib figures for query trends."""
    filename = "query_history.csv"
    figs = {}
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        st.info("No query history available for visualization.")
        return figs
    try:
        df = pd.read_csv(filename)
        if df.empty:
            st.info("Query history is empty.")
            return figs
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        if df.empty:
            st.info("No valid timestamps in query history.")
            return figs
        df = df.sort_values("timestamp")
        if 'predicted_diagnosis' in df.columns:
            valid_diagnoses = df[~df['predicted_diagnosis'].isin(["N/A - Model not trained", "Model not trained", "Prediction error", "N/A"])]['predicted_diagnosis']
            if not valid_diagnoses.empty:
                # Handle potential non-string entries by converting to string
                diag_counts = valid_diagnoses.astype(str).value_counts().nlargest(10)
                if not diag_counts.empty:
                    fig_diag, ax_diag = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=diag_counts.index, y=diag_counts.values, ax=ax_diag, palette="viridis")
                    ax_diag.set_title("Top 10 Predicted Diagnoses")
                    ax_diag.set_xlabel("Diagnosis")
                    ax_diag.set_ylabel("Count")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    figs['diagnosis_frequency'] = fig_diag
        if 'entities_summary' in df.columns:
            entity_list = []
            # Process entities_summary column, handling potential non-string values
            for entities_str in df['entities_summary'].fillna("").astype(str):
                if "Medical Entities:" in entities_str:
                    entities = entities_str.split("Medical Entities:", 1)[1]
                elif entities_str.startswith("- "):
                    entities = entities_str
                else:
                    entities = "" # Ignore entries that don't match expected formats
                if entities:
                    # Split by lines, remove leading/trailing spaces and hyphens, remove brackets
                    entity_list.extend([e.strip().lstrip('-').replace('[', '').replace(']', '') for e in entities.split('\n') if e.strip()])

            if entity_list:
                # Convert to lowercase before counting to group similar entities
                entity_counts = Counter([e.lower() for e in entity_list if e]).most_common(15)
                if entity_counts:
                    labels, sizes = zip(*entity_counts)
                    fig_ent, ax_ent = plt.subplots(figsize=(8, 8))
                    wedges, texts, autotexts = ax_ent.pie(sizes, autopct="%1.1f%%", startangle=140, pctdistance=0.85)
                    ax_ent.legend(wedges, labels, title="Entities", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    plt.setp(autotexts, size=8, weight="bold")
                    ax_ent.set_title("Top 15 Medical Entities")
                    plt.tight_layout()
                    figs['entities_distribution'] = fig_ent

        return figs
    except Exception as e:
        st.error(f"Error generating visualizations: {e}")
        logger.error(f"Visualization failed: {e}\n{traceback.format_exc()}")
        return figs


def build_faiss_index(embeddings):
    """Builds a FAISS index."""
    # Filter out None embeddings and ensure they are in a suitable format
    valid_embeddings = [
        np.array(emb).astype('float32') for emb in embeddings
        if emb is not None and isinstance(emb, (np.ndarray, list)) and len(emb) > 0
    ]

    if not valid_embeddings:
        logger.warning("No valid embeddings provided for FAISS index.")
        return None
    try:
        # Stack valid embeddings into a single numpy array
        embeddings_array = np.vstack(valid_embeddings)

        # Ensure the array is 2D
        if embeddings_array.ndim != 2:
            logger.error(f"Embeddings array has incorrect dimensions for FAISS: {embeddings_array.ndim}")
            st.error("Embeddings format incorrect for FAISS.")
            return None

        dim = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dim) # Using L2 distance
        index.add(embeddings_array)
        logger.info(f"FAISS index built successfully with {index.ntotal} vectors.")
        return index
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        logger.error(f"FAISS index failed: {e}\n{traceback.format_exc()}")
        return None

def search_faiss(index, query_embedding, k):
    """Searches the FAISS index."""
    if index is None:
        logger.warning("FAISS index is None, cannot search.")
        return np.array([]), np.array([]) # Return empty arrays on failure

    # Ensure query_embedding is in a suitable format and convert to float32 numpy array
    if not isinstance(query_embedding, (np.ndarray, list)) or len(query_embedding) == 0:
         logger.error("Invalid or empty query embedding format for FAISS search.")
         return np.array([]), np.array([]) # Return empty arrays on failure

    try:
        query_vec = np.array(query_embedding).reshape(1, -1).astype('float32')
        # Ensure k is not greater than the number of vectors in the index
        k_adjusted = min(k, index.ntotal)
        if k_adjusted <= 0:
            logger.warning("Adjusted k for FAISS search is zero or negative.")
            return np.array([]), np.array([])

        distances, indices = index.search(query_vec, k_adjusted)
        logger.info(f"FAISS search completed, found {len(indices[0])} results.")
        return distances[0], indices[0]
    except Exception as e:
        st.error(f"Error searching FAISS index: {e}")
        logger.error(f"FAISS search failed: {e}\n{traceback.format_exc()}")
        return np.array([]), np.array([]) # Return empty arrays on failure


def read_wearable_data(uploaded_file):
    """Reads and summarizes wearable data from an uploaded CSV."""
    if uploaded_file is None:
        return None
    try:
        # Read the file into a DataFrame, limiting rows to prevent memory issues
        df = pd.read_csv(uploaded_file, nrows=1000)
        summary = "Wearable Data Summary:\n"
        processed_columns = []

        # Normalize column names: convert to lowercase and replace spaces/special chars with underscore
        df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]+', '_', regex=True).str.strip('_')

        # Check for specific columns and add summary if found
        if "heart_rate" in df.columns:
            # Ensure column is numeric, coercing errors to NaN, then drop NaN
            hr_numeric = pd.to_numeric(df["heart_rate"], errors='coerce').dropna()
            if not hr_numeric.empty:
                summary += f"- Heart Rate: Avg={hr_numeric.mean():.1f}, Min={hr_numeric.min():.1f}, Max={hr_numeric.max():.1f} bpm\n"
                processed_columns.append("heart_rate")
        if "steps" in df.columns:
            steps_numeric = pd.to_numeric(df["steps"], errors='coerce').dropna()
            if not steps_numeric.empty:
                summary += f"- Steps: Total={steps_numeric.sum():.0f}, Avg={steps_numeric.mean():.0f}\n"
                processed_columns.append("steps")

        # Add a message if no recognized columns were found
        if not processed_columns:
            summary += "- No recognized wearable data columns (e.g., heart_rate, steps) found in the file."

        logger.info("Wearable data processed successfully.")
        return summary
    except Exception as e:
        st.warning(f"Could not process wearable data ({uploaded_file.name}): {e}")
        logger.error(f"Wearable data processing failed: {e}\n{traceback.format_exc()}")
        return f"Error processing wearable data: {e}"


async def run_analysis_pipeline(
    original_query,
    include_urls,
    omit_urls,
    additional_files_content,
    wearable_data_summary,
    search_depth,
    search_breadth,
    use_faiss,
    gemini_client_instance,
    supabase_client_instance,
    search_service_instance,
    extractor_instance,
    subject_analyzer_instance
):
    """Runs the full medical analysis pipeline asynchronously."""
    st.info("Starting analysis pipeline...")
    # Initialize results dictionary to store all outputs
    analysis_results = {
        'original_query': original_query,
        'comprehensive_report': "Analysis did not complete.",
        'patient_summary_report': "Analysis did not complete.",
        'citations': [],
        'sentiment': "N/A",
        'entities': "N/A",
        'subject_analysis': {},
        'subject_analysis_error': None,
        'search_results': {}, # Store full search results here
        'extracted_content': {}, # Store full extracted content here
        'additional_files_content': additional_files_content, # Store content from uploaded files
        'wearable_data_summary': wearable_data_summary, # Store wearable data summary
        'diagnosis_prediction': {"diagnosis": "N/A", "probabilities": {}},
        'prediction_explanation': "N/A"
    }
    current_date = datetime.today().strftime("%Y-%m-%d")
    current_query = original_query
    progress_bar = st.progress(0, text="Initializing...")

    # --- Preliminary Analysis (Sentiment & Entities) ---
    st.markdown("### Preliminary Analysis")
    prelim_status = st.empty()
    progress_bar.progress(2, text="Analyzing sentiment & entities...")
    prelim_status.info("Analyzing sentiment and entities...")
    try:
        sentiment_result, entities_result = await asyncio.gather(
            analyze_sentiment_gemini(current_query, gemini_client_instance),
            extract_medical_entities_gemini(current_query, gemini_client_instance)
        )
        analysis_results['sentiment'] = sentiment_result
        analysis_results['entities'] = entities_result
        prelim_status.success("Preliminary analysis complete.")
        # Removed direct display here, will display later in results section
    except Exception as e:
        st.error(f"Preliminary analysis failed: {e}")
        analysis_results['sentiment'] = f"Error: {e}"
        analysis_results['entities'] = f"Error: {e}"
        prelim_status.error("Preliminary analysis failed.")
    progress_bar.progress(5, text="Preliminary analysis complete.")

    # --- Subject Analysis ---
    st.markdown("### 1. Analyzing Query Subject")
    progress_bar.progress(7, text="Analyzing query context...")
    try:
        with st.spinner("Analyzing query context..."):
            if asyncio.iscoroutinefunction(subject_analyzer_instance.analyze):
                analysis = await subject_analyzer_instance.analyze(f"{current_query} (as of {current_date})")
            else:
                # Assuming subject_analyzer.analyze is synchronous and needs to run in an executor
                loop = asyncio.get_running_loop()
                analysis = await loop.run_in_executor(None, subject_analyzer_instance.analyze, f"{current_query} (as of {current_date})")
            analysis_results['subject_analysis'] = analysis
            st.success("Query analysis complete.")
            # Removed direct display here, will display later in results section
    except Exception as e:
        st.error(f"Subject analysis failed: {e}")
        analysis_results['subject_analysis_error'] = str(e)
        st.warning("Proceeding without full subject analysis.")
    progress_bar.progress(15, text="Query analysis complete.")

    # --- Web Search and Extraction ---
    st.markdown("### 2. Searching and Extracting Information")
    search_results_dict = {}
    extracted_content_dict = {}
    search_status = st.empty()

    # Determine topics for search
    topics = list(set([analysis_results.get("subject_analysis", {}).get("main_subject", current_query)] + analysis_results.get("subject_analysis", {}).get("What_needs_to_be_researched", [])))
    topics = [topic for topic in topics if topic and topic.strip()] # Filter out empty topics

    if not include_urls and search_service_instance and topics:
        search_status.info(f"Searching: {', '.join(topics)}")
        for i, topic in enumerate(topics):
            progress_val = 15 + int(35 * (i / len(topics)))
            progress_bar.progress(progress_val, text=f"Searching: {topic}...")
            search_status.info(f"Processing: {topic}...")
            try:
                loop = asyncio.get_running_loop()
                search_kwargs = {"search_depth": search_depth, "results": search_breadth}
                # Ensure search_service_instance.search_subject is compatible with run_in_executor if synchronous
                # If search_subject is async, await it directly
                if asyncio.iscoroutinefunction(search_service_instance.search_subject):
                    response = await search_service_instance.search_subject(topic, "medical", **search_kwargs)
                else:
                    response = await loop.run_in_executor(
                        None,
                        lambda srv, t, cat, kw: srv.search_subject(t, cat, **kw),
                        search_service_instance,
                        topic,
                        "medical",
                        search_kwargs
                    )

                results = response.get("results", [])
                # Filter results based on omit_urls
                filtered_results = [r for r in results if r.get("url") and not any(o.lower() in r["url"].lower() for o in omit_urls)]
                search_results_dict[topic] = filtered_results

                urls = [r["url"] for r in filtered_results if r.get("url")]
                if urls and extractor_instance:
                    extraction_progress = st.progress(0, text=f"Extracting for {topic}...")
                    extracted_items = []
                    batch_size = 5 # Extract URLs in smaller batches
                    for j in range(0, len(urls), batch_size):
                        batch_urls = urls[j:j+batch_size]
                        extraction_progress_val = int(((j + len(batch_urls)) / len(urls)) * 100)
                        extraction_progress.progress(extraction_progress_val, text=f"Extracting batch {j//batch_size + 1}/{len(urls)//batch_size + (1 if len(urls)%batch_size > 0 else 0)} for {topic}...")
                        try:
                            # Ensure extractor_instance.extract is compatible with run_in_executor if synchronous
                            # If extract is async, await it directly
                            if asyncio.iscoroutinefunction(extractor_instance.extract):
                                extraction_response = await extractor_instance.extract(urls=batch_urls, extract_depth="advanced", include_images=False)
                            else:
                                extraction_response = await loop.run_in_executor(
                                    None,
                                    lambda ext, u, d, i: ext.extract(urls=u, extract_depth=d, include_images=i),
                                    extractor_instance,
                                    batch_urls,
                                    "advanced",
                                    False
                                )
                            extracted_items.extend(extraction_response.get("results", []))
                        except Exception as e:
                            logger.error(f"Batch extraction failed for {topic}, batch {j//batch_size + 1}: {e}\n{traceback.format_exc()}")
                            st.warning(f"Batch extraction failed for {topic}, batch {j//batch_size + 1}: {e}")

                    extracted_content_dict[topic] = extracted_items
                    failed_count = sum(1 for item in extracted_content_dict[topic] if item.get("error"))
                    if failed_count:
                        st.warning(f"Failed to extract {failed_count} URLs for '{topic}'.")
                    extraction_progress.empty() # Clear batch extraction progress bar

            except Exception as e:
                st.error(f"Search failed for '{topic}': {e}")
                logger.error(f"Search failed for '{topic}': {e}\n{traceback.format_exc()}")
                search_results_dict[topic] = []
                extracted_content_dict[topic] = []

    elif include_urls and extractor_instance:
        search_status.info(f"Using {len(include_urls)} provided URLs.")
        filtered_urls = [url for url in include_urls if not any(o.lower() in url.lower() for o in omit_urls)]
        search_results_dict["User Provided"] = [{"title": "User Provided", "url": url, "score": "N/A"} for url in filtered_urls]

        if filtered_urls:
            progress_bar.progress(30, text="Extracting provided URLs...")
            with st.spinner("Extracting provided URLs..."):
                try:
                    loop = asyncio.get_running_loop()
                    # Ensure extractor_instance.extract is compatible with run_in_executor if synchronous
                    # If extract is async, await it directly
                    if asyncio.iscoroutinefunction(extractor_instance.extract):
                         extraction_response = await extractor_instance.extract(urls=filtered_urls, extract_depth="advanced", include_images=False)
                    else:
                        extraction_response = await loop.run_in_executor(
                            None,
                            lambda ext, u, d, i: ext.extract(urls=u, extract_depth=d, include_images=i),
                            extractor_instance,
                            filtered_urls,
                            "advanced",
                            False
                        )
                    extracted_content_dict["User Provided"] = extraction_response.get("results", [])
                    failed_count = sum(1 for item in extracted_content_dict["User Provided"] if item.get("error"))
                    if failed_count:
                        st.warning(f"Failed to extract {failed_count} provided URLs.")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    logger.error(f"Extraction failed: {e}\n{traceback.format_exc()}")
                    extracted_content_dict["User Provided"] = []
    else:
        search_status.warning("Search or URL extraction skipped (no topics, no URLs, or service unavailable).")

    analysis_results['search_results'] = search_results_dict
    analysis_results['extracted_content'] = extracted_content_dict
    progress_bar.progress(50, text="Search and extraction complete.")

    # --- RAG Analysis ---
    st.markdown("### 3. Performing RAG Analysis")
    rag_status = st.empty()
    comprehensive_report = "RAG analysis failed."
    citations = []
    full_content_for_rag = ""
    citations_map = {} # Map URL/filename to title for citation generation

    # Aggregate content from extracted sources
    for topic, items in extracted_content_dict.items():
        for item in items:
            url = item.get("url", "No URL")
            title = item.get("title", "No Title")
            content = item.get("text") or item.get("raw_content", "")
            if content and content.strip():
                # Add a clear separator and source identifier
                full_content_for_rag += f"\n\n=== START_SOURCE: {title} ({url}) ===\n{content}\n=== END_SOURCE ===\n"
                if url != "No URL":
                    citations_map[url] = title

    # Aggregate content from additional files
    for filename, content in additional_files_content.items():
        if content and content.strip():
            full_content_for_rag += f"\n\n=== START_SOURCE: Uploaded File: {filename} ===\n{content}\n=== END_SOURCE ===\n"
            citations_map[filename] = f"Uploaded File: {filename}"

    # Aggregate wearable data summary
    if wearable_data_summary and wearable_data_summary.strip():
        full_content_for_rag += f"\n\n=== START_SOURCE: Wearable Data ===\n{wearable_data_summary}\n=== END_SOURCE ===\n"
        citations_map["Wearable Data"] = "Wearable Data Summary"


    if not full_content_for_rag.strip():
        rag_status.warning("No content available for RAG. Skipping RAG analysis.")
        progress_bar.progress(85, text="RAG skipped.")
        comprehensive_report = "No relevant information found from search or provided files."
        citations = list(citations_map.values()) # Still include provided URLs/files as citations even if content wasn't used in RAG

    else:
        rag_status.info("Starting RAG pipeline...")
        with st.spinner("Performing RAG analysis..."):
            try:
                # --- Chunking ---
                progress_bar.progress(55, text="Chunking content...")
                def chunk_text(text, chunk_size=1000, overlap=100): # Adjusted chunk size for potentially better RAG
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = min(start + chunk_size, len(text))
                        chunk = text[start:end]
                        chunks.append(chunk)
                        start += chunk_size - overlap if chunk_size > overlap else chunk_size # Ensure progress

                    # Filter out very short chunks that might be noise
                    return [c for c in chunks if len(c.strip()) > 50]

                all_chunks = chunk_text(full_content_for_rag)

                if not all_chunks:
                    rag_status.warning("No usable chunks generated. Skipping RAG.")
                    progress_bar.progress(85, text="RAG skipped.")
                    comprehensive_report = "Could not process content for RAG."
                    citations = list(citations_map.values())
                else:
                    rag_status.info(f"Generated {len(all_chunks)} chunks for RAG.")

                    # --- Embedding ---
                    progress_bar.progress(60, text="Generating embeddings...")
                    rag_status.info(f"Embedding {len(all_chunks)} chunks...")

                    async def get_embedding_batch(texts, model_name):
                        try:
                            # google-generativeai embed_content can take a list of strings
                            result = await genai.embed_content_async(model=model_name, content=texts, task_type="RETRIEVAL_DOCUMENT")
                            # Safely access embeddings, check if result and 'embeddings' key exist
                            if result and 'embeddings' in result and result['embeddings']:
                                return result['embeddings']
                            else:
                                logger.warning(f"Embedding batch returned no embeddings or unexpected structure.")
                                return [None] * len(texts) # Return Nones for this batch
                        except Exception as e:
                            logger.error(f"Embedding batch failed: {e}\n{traceback.format_exc()}")
                            # Return a list of Nones for this batch to indicate failure
                            return [None] * len(texts)

                    batch_size = 100
                    num_batches = (len(all_chunks) + batch_size - 1) // batch_size
                    embedding_tasks = [
                        get_embedding_batch(all_chunks[i:i+batch_size], GOOGLE_EMBEDDING_MODEL)
                        for i in range(0, len(all_chunks), batch_size)
                    ]

                    batch_results = await asyncio.gather(*embedding_tasks, return_exceptions=True)

                    all_embeddings = []
                    valid_chunks = []
                    for i, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"Batch {i} embedding task raised exception: {result}")
                            # Skip this batch
                            continue
                        if result is not None:
                            # Add only valid embeddings and their corresponding chunks
                            for j, emb in enumerate(result):
                                if emb is not None and len(emb) > 0:
                                    all_embeddings.append(emb)
                                    # Calculate the correct index in the original all_chunks list
                                    original_chunk_index = i * batch_size + j
                                    if original_chunk_index < len(all_chunks):
                                         valid_chunks.append(all_chunks[original_chunk_index])
                                else:
                                    logger.warning(f"Skipping invalid embedding for chunk {i*batch_size + j}")


                    if not all_embeddings:
                        rag_status.error("Embedding failed for all chunks.")
                        progress_bar.progress(85, text="RAG failed.")
                        comprehensive_report = "Embedding failed, could not perform RAG."
                        citations = list(citations_map.values())
                    else:
                        rag_status.info(f"Successfully embedded {len(all_embeddings)} chunks.")
                        progress_bar.progress(75, text="Matching relevant content...")

                        # --- Matching (FAISS or Supabase) ---
                        matched_chunks = []
                        query_embedding = None
                        try:
                            # Get embedding for the query
                            query_embedding_response = await genai.embed_content_async(model=GOOGLE_EMBEDDING_MODEL, content=current_query, task_type="RETRIEVAL_QUERY")
                             # Safely access the query embedding
                            if query_embedding_response and 'embeddings' in query_embedding_response and query_embedding_response['embeddings']:
                                query_embedding = query_embedding_response['embeddings'][0] # Get the first embedding
                            else:
                                raise ValueError("Query embedding response was empty or in unexpected format.")

                        except Exception as e:
                            logger.error(f"Failed to embed query for RAG: {e}\n{traceback.format_exc()}")
                            st.error(f"Failed to embed query for RAG: {e}")
                            rag_status.error("Failed to embed query. Cannot perform RAG matching.")
                            progress_bar.progress(85, text="RAG failed.")
                            comprehensive_report = "Failed to embed query, could not perform RAG."
                            citations = list(citations_map.values())

                        if query_embedding is not None:
                            k = min(20, len(valid_chunks)) # Retrieve top N relevant chunks

                            if use_faiss and all_embeddings:
                                rag_status.info("Attempting to use FAISS for matching...")
                                loop = asyncio.get_running_loop()
                                # Pass valid_embeddings to build FAISS index
                                faiss_index = await loop.run_in_executor(None, build_faiss_index, all_embeddings)
                                if faiss_index:
                                    try:
                                        distances, indices = await loop.run_in_executor(None, search_faiss, faiss_index, query_embedding, k)
                                        if indices is not None and indices.size > 0:
                                            # Retrieve chunks using indices from the valid_chunks list
                                            matched_chunks = [valid_chunks[i] for i in indices if 0 <= i < len(valid_chunks)]
                                            rag_status.info(f"FAISS matching found {len(matched_chunks)} relevant chunks.")
                                        else:
                                            rag_status.warning("FAISS search returned no indices. Falling back to Supabase.")
                                            use_faiss = False # Fallback to Supabase
                                    except Exception as e:
                                        logger.error(f"FAISS search execution failed: {e}\n{traceback.format_exc()}")
                                        rag_status.warning(f"FAISS search execution failed: {e}. Falling back to Supabase.")
                                        use_faiss = False # Fallback to Supabase
                                else:
                                    rag_status.warning("FAISS index failed to build. Falling back to Supabase.")
                                    use_faiss = False # Fallback to Supabase

                            if not use_faiss and supabase_client_instance:
                                rag_status.info("Using Supabase for matching...")
                                try:
                                    # Supabase RPC expects embedding as a list
                                    match_params = {"query_embedding": query_embedding, "match_threshold": 0.5, "match_count": k} # Adjusted match_threshold
                                    loop = asyncio.get_running_loop()
                                    # Ensure supabase_client_instance.rpc is compatible with run_in_executor if synchronous
                                    match_response = await loop.run_in_executor(None, lambda: supabase_client_instance.rpc("match_chunks", match_params).execute())

                                    if match_response and match_response.data:
                                        # Extract chunks from Supabase response, handling potential 'chunk' or 'content' keys
                                        matched_chunks = [row.get("chunk") or row.get("content") for row in match_response.data if row.get("chunk") or row.get("content")]
                                        rag_status.info(f"Supabase matching found {len(matched_chunks)} relevant chunks.")
                                    else:
                                        rag_status.warning("Supabase matching found no relevant chunks.")
                                        matched_chunks = [] # Ensure matched_chunks is an empty list if no data

                                except Exception as e:
                                     logger.error(f"Supabase RAG matching failed: {e}\n{traceback.format_exc()}")
                                     rag_status.error(f"Supabase RAG matching failed: {e}")
                                     matched_chunks = [] # Ensure matched_chunks is an empty list on error

                            if not matched_chunks:
                                rag_status.warning("No relevant content chunks found after matching.")
                                comprehensive_report = "No relevant information found from search or provided files."
                                citations = list(citations_map.values())
                            else:
                                # --- Synthesis ---
                                rag_status.info(f"Synthesizing report from {len(matched_chunks)} relevant chunks...")
                                # Join the matched chunks to provide context to the LLM
                                aggregated_relevant_content = "\n\n".join(matched_chunks)

                                # Generate citations based on the sources that contributed chunks
                                # This is a simplification; a more robust approach would map chunks back to sources.
                                # For now, we'll use the initial citations_map based on all extracted/provided sources.
                                final_citations = list(citations_map.values())

                                # Construct the prompt for the language model
                                synthesis_prompt = f"""You are an expert diagnostic report generator. Synthesize a comprehensive medical diagnostic report based on the patient's query and the provided relevant context.

Patient Query: {current_query}

Relevant Context:
---
{aggregated_relevant_content}
---

Instructions:
1. Identify key symptoms, medical history, and relevant details from the Patient Query and Relevant Context.
2. Analyze the information from the Relevant Context, noting any potential diagnoses, contributing factors, or conflicting information related to the patient's query.
3. Integrate temporal context if available in the subject analysis.
4. Incorporate wearable data summary if provided.
5. Structure the report in Markdown with sections: Patient Query, Analysis (incorporating findings from context), Wearable Data (if applicable), Considerations (differential diagnoses, recommended investigations, treatment considerations), Disclaimer.
6. Maintain a professional and medically informative tone. Avoid giving a definitive diagnosis; instead, discuss possibilities and necessary steps for a healthcare professional.
7. If the provided context is insufficient to address aspects of the query, state this clearly.
8. Do NOT include a separate "Citations" section within this report text itself. Citations will be listed separately.
"""
                                progress_bar.progress(80, text="Synthesizing comprehensive report...")
                                messages = [{"role": "user", "content": synthesis_prompt}]

                                try:
                                    # Use the gemini_client to generate the comprehensive report
                                    # Ensure gemini_client.chat is compatible with run_in_executor if synchronous
                                    if asyncio.iscoroutinefunction(gemini_client_instance.chat):
                                        response_data = await gemini_client_instance.chat(messages)
                                    else:
                                         response_data = await loop.run_in_executor(None, gemini_client_instance.chat, messages)

                                    comprehensive_report = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Synthesis error.")
                                    citations = final_citations # Assign the generated citations
                                    rag_status.success("Comprehensive report synthesized.")

                                except Exception as e:
                                    logger.error(f"Comprehensive report synthesis failed: {e}\n{traceback.format_exc()}")
                                    rag_status.error(f"Comprehensive report synthesis failed: {e}")
                                    comprehensive_report = f"Comprehensive report synthesis failed: {e}"
                                    citations = final_citations # Still include potential citations


            except Exception as e:
                logger.error(f"RAG pipeline error during embedding or matching: {e}\n{traceback.format_exc()}")
                rag_status.error(f"RAG pipeline error: {e}")
                comprehensive_report = f"RAG pipeline failed: {e}"
                citations = list(citations_map.values())


    analysis_results['comprehensive_report'] = comprehensive_report
    analysis_results['citations'] = citations
    progress_bar.progress(85, text="RAG complete.")


    # --- Patient-Friendly Summary ---
    st.markdown("### 4. Generating Patient Summary")
    summary_status = st.empty()
    patient_summary_report = "Summary generation failed."

    # Only generate summary if the comprehensive report had some content
    if comprehensive_report and "failed" not in comprehensive_report.lower() and "error" not in comprehensive_report.lower() and "not found" not in comprehensive_report.lower():
        progress_bar.progress(90, text="Generating patient-friendly summary...")
        summary_status.info("Generating patient-friendly summary...")
        with st.spinner("Generating patient-friendly summary..."):
            try:
                # Prompt for generating a patient-friendly summary with actionable steps
                summary_prompt = f"""As a medical assistant, create a patient-friendly summary from the following comprehensive medical report.
The summary should be easy for a non-medical person to understand.
It must clearly state:
- The main findings or possible issues discussed in the report.
- Any suggested next steps, tests, or consultations mentioned.
- Any important considerations or disclaimers from the report.
Keep it concise and actionable.

Comprehensive Medical Report:
{comprehensive_report}
"""
                messages = [{"role": "user", "content": summary_prompt}]
                loop = asyncio.get_running_loop()

                # Use the gemini_client to generate the patient summary
                # Ensure gemini_client.chat is compatible with run_in_executor if synchronous
                if asyncio.iscoroutinefunction(gemini_client_instance.chat):
                    response_data = await gemini_client_instance.chat(messages)
                else:
                    response_data = await loop.run_in_executor(None, gemini_client_instance.chat, messages)

                patient_summary_report = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Summary error.")
                summary_status.success("Patient-friendly summary generated.")
            except Exception as e:
                logger.error(f"Patient summary generation failed: {e}\n{traceback.format_exc()}")
                summary_status.error(f"Patient summary failed: {e}")
                patient_summary_report = f"Patient summary generation failed: {e}"
    else:
        summary_status.warning("Skipping patient summary generation due to issues with the comprehensive report.")

    analysis_results['patient_summary_report'] = patient_summary_report
    progress_bar.progress(100, text="Analysis complete!")
    st.success("Analysis complete!")

    # Return the full results dictionary
    return analysis_results


# --- Streamlit UI Layout ---

with st.sidebar:
    st.header("Configuration")
    query = st.text_area("1. Enter Patient Symptoms/Issue:", height=150, key="query_input")
    st.subheader("Web Search Options")
    include_urls_str = st.text_area("Include URLs (one per line, optional):", height=150, key="include_urls")
    omit_urls_str = st.text_area("Omit URLs Containing (one per line, optional):", height=150, key="omit_urls")
    search_depth = st.selectbox("Search Depth:", ["basic", "advanced"], index=1, key="search_depth")
    search_breadth = st.number_input("Search Breadth:", min_value=1, max_value=20, value=7, key="search_breadth") # Min results changed to 1
    st.subheader("Reference Files")
    uploaded_files = st.file_uploader("Upload Files (PDF, DOCX, CSV, XLSX, TXT):", accept_multiple_files=True, type=['pdf', 'docx', 'csv', 'xlsx', 'xls', 'txt'], key="file_uploader")
    uploaded_wearable_file = st.file_uploader("Wearable Data CSV:", accept_multiple_files=False, type=['csv'], key="wearable_uploader")
    uploaded_survey_file = st.file_uploader("Survey Data CSV:", accept_multiple_files=False, type=['csv'], key="survey_uploader")
    st.subheader("Advanced Options")
    use_faiss = st.checkbox("Use FAISS for RAG (Recommended)", value=True, key="use_faiss") # Default to True and add recommendation
    st.divider()
    submit_button = st.button("Run Analysis", type="primary", key="submit_button", use_container_width=True)

st.header("Analysis Results")
st.markdown("**Disclaimer:** This tool provides information only and is not a substitute for professional medical advice.")

# Initialize session state variables
default_state = {
    'analysis_complete': False,
    'results': {}, # This will store the full analysis_results dictionary
    'classifier_trained': False,
    'vectorizer': None,
    'model': None,
    'background_texts_sample': None,
    'last_survey_file': None, # To track the uploaded survey file and prevent re-training
    'prediction_displayed': False # To ensure prediction is displayed only once per run
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Train Classifier (if survey data is uploaded) ---
survey_df = None
if uploaded_survey_file:
    file_details = (uploaded_survey_file.name, uploaded_wearable_file.size) # Corrected to use uploaded_survey_file.size
    # Check if a new file has been uploaded or a different file is selected
    if 'last_survey_file' not in st.session_state or st.session_state.last_survey_file != file_details:
        st.session_state.classifier_trained = False
        st.session_state.vectorizer = None
        st.session_state.model = None
        st.session_state.background_texts_sample = None
        st.session_state.last_survey_file = file_details # Update the last processed file

        try:
            # Read the survey data
            survey_df = pd.read_csv(uploaded_survey_file)
            st.sidebar.success(f"Loaded survey data: {uploaded_survey_file.name}")

            # Train the classifier
            with st.spinner("Training symptom classifier..."):
                vectorizer, model = train_symptom_classifier(survey_df)

            if vectorizer and model:
                st.session_state.classifier_trained = True
                st.session_state.vectorizer = vectorizer
                st.session_state.model = model

                # Prepare background texts sample for SHAP explanation
                symptom_col = 'What are the current symptoms or health issues you are facing'
                if symptom_col in survey_df.columns:
                    background_texts = survey_df[symptom_col].astype(str).fillna("").tolist()
                    # Select a reasonable sample size, ensuring no duplicates if sample_size > len(background_texts)
                    sample_size = min(100, len(background_texts))
                    if sample_size > 0:
                         st.session_state.background_texts_sample = np.random.choice(background_texts, sample_size, replace=False).tolist()
                    else:
                         st.session_state.background_texts_sample = []
                else:
                    st.session_state.background_texts_sample = []


                st.sidebar.success("Classifier trained successfully.")
            else:
                st.session_state.classifier_trained = False
                st.sidebar.warning("Classifier training failed. Check data format.")

        except Exception as e:
            st.sidebar.error(f"Error processing survey data or training classifier: {e}")
            logger.error(f"Survey data processing or training error: {e}\n{traceback.format_exc()}")
            st.session_state.classifier_trained = False
            st.session_state.last_survey_file = None # Reset if processing fails

# Reset classifier state if the file uploader is cleared
elif 'last_survey_file' in st.session_state and uploaded_survey_file is None:
    st.session_state.classifier_trained = False
    st.session_state.vectorizer = None
    st.session_state.model = None
    st.session_state.background_texts_sample = None
    del st.session_state.last_survey_file
    st.sidebar.info("Survey data removed. Classifier reset.")


# --- Handle Analysis Submission ---
if submit_button and query:
    # Reset analysis state for a new run
    st.session_state.analysis_complete = False
    st.session_state.results = {}
    st.session_state.prediction_displayed = False # Reset prediction display flag

    # Process URLs and files
    include_urls_list = [url.strip() for url in include_urls_str.split('\n') if url.strip()]
    omit_urls_list = [url.strip() for url in omit_urls_str.split('\n') if url.strip()]
    additional_files_content = {}
    if uploaded_files:
        st.markdown("### Processing Reference Files")
        with st.expander("File Processing Status", expanded=True):
            for file in uploaded_files:
                file_status = st.empty() # Create a placeholder for each file's status
                file_status.info(f"Processing {file.name}...")
                content = extract_text_from_file(file)
                if content and "Error" not in content and "[Unsupported" not in content:
                    additional_files_content[file.name] = content
                    file_status.success(f"✅ Processed {file.name}")
                else:
                    # Display the specific error or warning message
                    file_status.warning(f"⚠️ Could not process {file.name}: {content}")
    st.divider() # Add a divider after file processing section

    # Process Wearable Data
    wearable_summary = read_wearable_data(uploaded_wearable_file)
    if wearable_summary and "Error" not in wearable_summary:
        st.markdown("### Wearable Data Summary")
        st.text(wearable_summary) # Display the summary directly or in an expander
        st.divider()


    # --- Diagnosis Prediction (if classifier is trained) ---
    if st.session_state.classifier_trained and not st.session_state.prediction_displayed:
        st.markdown("### AI Diagnosis Prediction (from Survey Data)")
        with st.spinner("Generating diagnosis prediction and explanation..."):
            # Ensure query is a string
            query_str = str(query)
            pred_diag, pred_proba = predict_diagnosis(query_str, st.session_state.vectorizer, st.session_state.model)
            explanation = explain_diagnosis_shap(query_str, st.session_state.vectorizer, st.session_state.model, st.session_state.background_texts_sample)

        st.write(f"**Predicted Diagnosis:** {pred_diag}")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Prediction Probabilities", expanded=False):
                # Display probabilities as a dictionary
                st.json(pred_proba)
        with col2:
            with st.expander("Explanation (SHAP Analysis)", expanded=False):
                 st.text(explanation)

        # Store prediction results in session state.results before running the main pipeline
        st.session_state.results['diagnosis_prediction'] = {"diagnosis": pred_diag, "probabilities": pred_proba}
        st.session_state.results['prediction_explanation'] = explanation
        st.session_state.prediction_displayed = True # Set flag to True

        st.divider() # Add a divider after the prediction section
    elif not st.session_state.classifier_trained and not st.session_state.prediction_displayed:
        st.info("AI Diagnosis Prediction skipped. Upload survey data to enable this feature.")
        st.session_state.prediction_displayed = True # Set flag even if skipped to prevent repeated message
        st.divider()


    # --- Run Main Analysis Pipeline ---
    st.header("Main Analysis Pipeline")
    try:
        analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
        subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config) # Assuming GeminiClient is the correct client

        # Run the asynchronous pipeline
        analysis_results = asyncio.run(run_analysis_pipeline(
            original_query=query,
            include_urls=include_urls_list,
            omit_urls=omit_urls_list,
            additional_files_content=additional_files_content,
            wearable_data_summary=wearable_summary,
            search_depth=search_depth,
            search_breadth=search_breadth,
            use_faiss=use_faiss,
            gemini_client_instance=gemini_client,
            supabase_client_instance=supabase,
            search_service_instance=search_service,
            extractor_instance=extractor,
            subject_analyzer_instance=subject_analyzer
        ))

        # Store the full analysis results in session state
        st.session_state.results.update(analysis_results) # Update with results from pipeline

        # Save query history after successful pipeline completion
        save_query_history(
            query,
            st.session_state.results.get('diagnosis_prediction', {}).get('diagnosis', 'N/A'), # Use prediction diagnosis if available
            st.session_state.results.get('sentiment', 'N/A'),
            st.session_state.results.get('entities', 'N/A')
        )

        st.session_state.analysis_complete = True # Set flag to display results

    except Exception as e:
        st.error(f"Error during analysis pipeline execution: {e}")
        logger.error(f"Analysis pipeline execution error: {e}\n{traceback.format_exc()}")
        st.session_state.analysis_complete = False # Ensure analysis_complete is False on error


# --- Display Persistent Results (after analysis is complete) ---
if st.session_state.analysis_complete:
    st.divider() # Adds a horizontal line to separate sections
    st.header("Complete Analysis Results")

    # Use tabs to organize the results
    tab_summary, tab_comprehensive, tab_preliminary, tab_subject, tab_search, tab_extracted, tab_prediction, tab_trends = st.tabs([
        "Patient Summary",
        "Comprehensive Report",
        "Preliminary Analysis",
        "Subject Analysis",
        "Search Results",
        "Extracted Content",
        "AI Prediction",
        "Query Trends"
    ])

    with tab_summary:
        st.markdown("### Patient-Friendly Summary")
        st.write(st.session_state.results.get('patient_summary_report', 'No patient summary available.'))
        if st.session_state.results.get('wearable_data_summary'):
            st.markdown("#### Wearable Data Summary")
            st.text(st.session_state.results['wearable_data_summary'])


    with tab_comprehensive:
        st.markdown("### Comprehensive Report")
        st.write(st.session_state.results.get('comprehensive_report', 'No comprehensive report available.'))
        if st.session_state.results.get('citations'):
            st.markdown("#### Citations")
            for citation in st.session_state.results['citations']:
                st.write(f"- {citation}")


    with tab_preliminary:
        st.markdown("### Preliminary Analysis")
        st.markdown("#### Sentiment Analysis")
        st.write(st.session_state.results.get('sentiment', 'Sentiment analysis not available.'))
        st.markdown("#### Medical Entities")
        st.write(st.session_state.results.get('entities', 'Medical entities not available.'))

    with tab_subject:
        st.markdown("### Subject Analysis (Agent's Understanding)")
        subject_analysis = st.session_state.results.get('subject_analysis', {})
        if subject_analysis:
            st.write(f"**Original Query:** {st.session_state.results.get('original_query', 'N/A')}")
            st.write(f"**Identified Issue:** {subject_analysis.get('main_subject', 'Unknown')}")
            temporal = subject_analysis.get("temporal_context", {})
            if temporal:
                st.write("**Temporal Context:**")
                for k, v in temporal.items():
                    st.write(f"- {k.capitalize()}: {v}")
            needs = subject_analysis.get("What_needs_to_be_researched", [])
            st.write("**Key Aspects to Investigate:**")
            st.write(f"{', '.join(needs) if needs else 'None'}")
        elif st.session_state.results.get('subject_analysis_error'):
            st.error(f"Subject analysis failed: {st.session_state.results['subject_analysis_error']}")
        else:
             st.info("Subject analysis not available.")


    with tab_search:
        st.markdown("### Search Results")
        search_results = st.session_state.results.get('search_results', {})
        if search_results:
            for topic, results in search_results.items():
                st.markdown(f"#### Results for '{topic}'")
                if results:
                    for i, res in enumerate(results):
                        title = res.get('title', 'No Title')
                        url = res.get('url', '#')
                        snippet = res.get('snippet', 'No snippet available.')
                        st.write(f"{i+1}. [{title}]({url})")
                        st.markdown(f"   *{snippet}*") # Display snippet italicized
                else:
                    st.info(f"No search results found for '{topic}'.")
        else:
            st.info("No web search was performed or no results were found.")

    with tab_extracted:
        st.markdown("### Extracted Content")
        extracted_content = st.session_state.results.get('extracted_content', {})
        additional_files_content = st.session_state.results.get('additional_files_content', {})

        if extracted_content:
            st.markdown("#### Content from Web Extraction")
            for topic, items in extracted_content.items():
                st.markdown(f"##### Extracted for '{topic}'")
                if items:
                    for i, item in enumerate(items):
                        url = item.get('url', 'No URL')
                        content = item.get('text') or item.get('raw_content', 'No content extracted.')
                        st.write(f"{i+1}. **URL:** {url}")
                        with st.expander("View Extracted Text"):
                            st.text(content) # Use st.text for preformatted text
                else:
                    st.info(f"No content extracted for '{topic}'.")
        else:
            st.info("No web content was extracted.")

        if additional_files_content:
            st.markdown("#### Content from Uploaded Files")
            for filename, content in additional_files_content.items():
                 st.markdown(f"##### File: {filename}")
                 with st.expander("View File Content"):
                      st.text(content) # Display file content

        if not extracted_content and not additional_files_content:
             st.info("No web content extracted and no additional files were processed.")


    with tab_prediction:
        st.markdown("### AI Diagnosis Prediction (from Survey Data)")
        prediction_results = st.session_state.results.get('diagnosis_prediction', {})
        explanation = st.session_state.results.get('prediction_explanation', 'Explanation not available.')
        if prediction_results and prediction_results.get('diagnosis') != 'N/A':
            st.write(f"**Predicted Diagnosis:** {prediction_results.get('diagnosis', 'N/A')}")
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("Prediction Probabilities", expanded=False):
                    st.json(prediction_results.get('probabilities', {}))
            with col2:
                 with st.expander("Explanation (SHAP Analysis)", expanded=False):
                    st.text(explanation)
        else:
            st.info("AI Diagnosis Prediction was not performed (no survey data uploaded or training failed).")

    with tab_trends:
        st.markdown("### Query Trends Visualizations")
        # Generate and display visualizations when this tab is active
        figs = visualize_query_trends()
        if figs:
            for fig_name, fig in figs.items():
                st.pyplot(fig)
                plt.close(fig) # Close the figure to free up memory
        else:
            st.info("No query history available to generate visualizations.")
