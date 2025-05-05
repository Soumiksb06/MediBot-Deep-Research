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
GEMINI_MODEL_NAME = st.secrets.get("models", {}).get("GEMINI_MODEL_NAME") or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
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
            background_summary = shap.kmeans(background_data_sparse.toarray(), num_clusters)
        else:
            background_summary = np.zeros((1, X_query.shape[1]))
        explainer = shap.KernelExplainer(predict_proba_dense, background_summary)
        shap_values = explainer.shap_values(X_query, nsamples='auto')
        predicted_class = model.predict(X_query)[0]
        predicted_class_index = list(model.classes_).index(predicted_class)
        shap_values_instance = shap_values[predicted_class_index][0]
        feature_names = vectorizer.get_feature_names_out()
        sorted_indices = np.argsort(np.abs(shap_values_instance))[::-1]
        explanation = f"Top contributing features for '{predicted_class}' (SHAP values):\n"
        for idx in sorted_indices[:5]:
            if idx < len(feature_names) and not np.isclose(shap_values_instance[idx], 0):
                explanation += f"- {feature_names[idx]}: {shap_values_instance[idx]:.4f}\n"
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
            sentiment_summary = str(sentiment).split('\n')[0]
            entities_summary = str(entities).split('\n')[0]
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
            valid_diagnoses = df[~df['predicted_diagnosis'].isin(["N/A - Model not trained", "Model not trained", "Prediction error"])]['predicted_diagnosis']
            diag_counts = valid_diagnoses.value_counts().nlargest(10)
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
            for entities_str in df['entities_summary'].fillna("").astype(str):
                if "Medical Entities:" in entities_str:
                    entities = entities_str.split("Medical Entities:", 1)[1]
                elif entities_str.startswith("- "):
                    entities = entities_str
                else:
                    entities = ""
                if entities:
                    entity_list.extend([e.strip().lstrip('-').replace('[', '').replace(']', '') for e in entities.split('\n') if e.strip()])
            if entity_list:
                entity_counts = Counter([e.lower() for e in entity_list]).most_common(15)
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
    if not embeddings or not isinstance(embeddings[0], (np.ndarray, list)):
        return None
    try:
        embeddings_array = np.array(embeddings).astype('float32')
        if embeddings_array.ndim != 2:
            st.error("Embeddings format incorrect for FAISS.")
            return None
        dim = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_array)
        return index
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        logger.error(f"FAISS index failed: {e}\n{traceback.format_exc()}")
        return None

def search_faiss(index, query_embedding, k):
    """Searches the FAISS index."""
    if index is None:
        return None, None
    try:
        query_vec = np.array(query_embedding).reshape(1, -1).astype('float32')
        k_adjusted = min(k, index.ntotal)
        if k_adjusted <= 0:
            return np.array([]), np.array([])
        distances, indices = index.search(query_vec, k_adjusted)
        return distances[0], indices[0]
    except Exception as e:
        st.error(f"Error searching FAISS index: {e}")
        logger.error(f"FAISS search failed: {e}\n{traceback.format_exc()}")
        return None, None

def read_wearable_data(uploaded_file):
    """Reads and summarizes wearable data from an uploaded CSV."""
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file, nrows=1000)  # Limit rows for efficiency
        summary = "Wearable Data Summary:\n"
        processed_columns = []
        df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]+', '_', regex=True).str.strip('_')
        if "heart_rate" in df.columns:
            hr_numeric = pd.to_numeric(df["heart_rate"], errors='coerce').dropna()
            if not hr_numeric.empty:
                summary += f"- Heart Rate: Avg={hr_numeric.mean():.1f}, Min={hr_numeric.min():.1f}, Max={hr_numeric.max():.1f} bpm\n"
                processed_columns.append("heart_rate")
        if "steps" in df.columns:
            steps_numeric = pd.to_numeric(df["steps"], errors='coerce').dropna()
            if not steps_numeric.empty:
                summary += f"- Steps: Total={steps_numeric.sum():.0f}, Avg={steps_numeric.mean():.0f}\n"
                processed_columns.append("steps")
        if not processed_columns:
            summary += "- No recognized columns found."
        return summary
    except Exception as e:
        st.warning(f"Could not process wearable data ({uploaded_file.name}): {e}")
        logger.error(f"Wearable data failed: {e}\n{traceback.format_exc()}")
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
    analysis_results = {
        'comprehensive_report': "Analysis did not complete.",
        'patient_summary_report': "Analysis did not complete.",
        'citations': [],
        'sentiment': "N/A",
        'entities': "N/A",
        'subject_analysis': {},
        'subject_analysis_error': None,
        'search_results': {},
        'extracted_content': {}
    }
    current_date = datetime.today().strftime("%Y-%m-%d")
    current_query = original_query
    progress_bar = st.progress(0, text="Initializing...")

    # Preliminary Analysis
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
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Sentiment Analysis", expanded=False):
                st.write(sentiment_result)
        with col2:
            with st.expander("Medical Entities", expanded=False):
                st.write(entities_result)
    except Exception as e:
        st.error(f"Preliminary analysis failed: {e}")
        analysis_results['sentiment'] = f"Error: {e}"
        analysis_results['entities'] = f"Error: {e}"
        prelim_status.error("Preliminary analysis failed.")
    progress_bar.progress(5, text="Preliminary analysis complete.")

    # Subject Analysis
    st.markdown("### 1. Analyzing Query Subject")
    progress_bar.progress(7, text="Analyzing query context...")
    try:
        with st.spinner("Analyzing query context..."):
            if asyncio.iscoroutinefunction(subject_analyzer_instance.analyze):
                analysis = await subject_analyzer_instance.analyze(f"{current_query} (as of {current_date})")
            else:
                loop = asyncio.get_running_loop()
                analysis = await loop.run_in_executor(None, subject_analyzer_instance.analyze, f"{current_query} (as of {current_date})")
            analysis_results['subject_analysis'] = analysis
            st.success("Query analysis complete.")
            with st.expander("Agent's Understanding", expanded=True):
                st.write(f"**Query:** {original_query}")
                st.write(f"**Issue:** {analysis.get('main_subject', 'Unknown')}")
                temporal = analysis.get("temporal_context", {})
                if temporal:
                    st.write("**Temporal Context:**")
                    for k, v in temporal.items():
                        st.write(f"- {k.capitalize()}: {v}")
                needs = analysis.get("What_needs_to_be_researched", [])
                st.write("**To Investigate:**")
                st.write(f"{', '.join(needs) if needs else 'None'}")
    except Exception as e:
        st.error(f"Subject analysis failed: {e}")
        analysis_results['subject_analysis_error'] = str(e)
        st.warning("Proceeding without full subject analysis.")
    progress_bar.progress(15, text="Query analysis complete.")

    # Web Search and Extraction
    st.markdown("### 2. Searching and Extracting Information")
    search_results_dict = {}
    extracted_content_dict = {}
    search_status = st.empty()
    if not include_urls and search_service_instance:
        topics = list(set([analysis.get("main_subject", current_query)] + analysis.get("What_needs_to_be_researched", [])))
        if not topics:
            search_status.warning("No topics identified for search.")
        else:
            search_status.info(f"Searching: {', '.join(topics)}")
            for i, topic in enumerate(topics):
                progress_val = 15 + int(35 * (i / len(topics)))
                progress_bar.progress(progress_val, text=f"Searching: {topic}...")
                search_status.info(f"Processing: {topic}...")
                try:
                    loop = asyncio.get_running_loop()
                    search_kwargs = {"search_depth": search_depth, "results": search_breadth}
                    response = await loop.run_in_executor(
                        None,
                        lambda srv, t, cat, kw: srv.search_subject(t, cat, **kw),
                        search_service_instance,
                        topic,
                        "medical",
                        search_kwargs
                    )
                    results = response.get("results", [])
                    filtered_results = [r for r in results if r.get("url") and not any(o.lower() in r["url"].lower() for o in omit_urls)]
                    search_results_dict[topic] = filtered_results
                    st.subheader(f"Results for '{topic}'")
                    for j, res in enumerate(filtered_results):
                        st.write(f"{j+1}. [{res.get('title', 'No Title')}]({res.get('url', '#')})")
                    urls = [r["url"] for r in filtered_results]
                    if urls:
                        extraction_response = await loop.run_in_executor(
                            None,
                            lambda ext, u, d, i: ext.extract(urls=u, extract_depth=d, include_images=i),
                            extractor_instance,
                            urls,
                            "advanced",
                            False
                        )
                        extracted_content_dict[topic] = extraction_response.get("results", [])
                        failed = sum(1 for item in extracted_content_dict[topic] if item.get("error"))
                        if failed:
                            st.warning(f"Failed to extract {failed} URLs for '{topic}'.")
                except Exception as e:
                    st.error(f"Search failed for '{topic}': {e}")
                    search_results_dict[topic] = []
                    extracted_content_dict[topic] = []
    elif include_urls and extractor_instance:
        search_status.info(f"Using {len(include_urls)} provided URLs.")
        filtered_urls = [url for url in include_urls if not any(o.lower() in url.lower() for o in omit_urls)]
        search_results_dict["User Provided"] = [{"title": "User Provided", "url": url, "score": "N/A"} for url in filtered_urls]
        st.subheader("Provided URLs")
        for j, url in enumerate(filtered_urls):
            st.write(f"{j+1}. [{url}]({url})")
        if filtered_urls:
            progress_bar.progress(30, text="Extracting provided URLs...")
            with st.spinner("Extracting provided URLs..."):
                try:
                    loop = asyncio.get_running_loop()
                    extraction_response = await loop.run_in_executor(
                        None,
                        lambda ext, u, d, i: ext.extract(urls=u, extract_depth=d, include_images=i),
                        extractor_instance,
                        filtered_urls,
                        "advanced",
                        False
                    )
                    extracted_content_dict["User Provided"] = extraction_response.get("results", [])
                    failed = sum(1 for item in extracted_content_dict["User Provided"] if item.get("error"))
                    if failed:
                        st.warning(f"Failed to extract {failed} provided URLs.")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    extracted_content_dict["User Provided"] = []
    else:
        search_status.warning("Search skipped (no URLs or service unavailable).")
    analysis_results['search_results'] = search_results_dict
    analysis_results['extracted_content'] = extracted_content_dict
    progress_bar.progress(50, text="Search complete.")

    # RAG Analysis
    st.markdown("### 3. Performing RAG Analysis")
    rag_status = st.empty()
    comprehensive_report = "RAG analysis failed."
    citations = []
    full_content_for_rag = ""
    citations_map = {}
    for topic, items in extracted_content_dict.items():
        for item in items:
            url = item.get("url", "No URL")
            title = item.get("title", "No Title")
            content = item.get("text") or item.get("raw_content", "")
            if content.strip():
                full_content_for_rag += f"\n\n=== {title} ({url}) ===\n{content}\n"
                if url != "No URL":
                    citations_map[url] = title
    for filename, content in additional_files_content.items():
        if content.strip():
            full_content_for_rag += f"\n\n=== Uploaded File: {filename} ===\n{content}\n"
            citations_map[filename] = f"Uploaded File: {filename}"
    if wearable_data_summary and wearable_data_summary.strip():
        full_content_for_rag += f"\n\n=== Wearable Data ===\n{wearable_data_summary}\n"
        citations_map["Wearable Data"] = "Wearable Data Summary"
    if not full_content_for_rag.strip():
        rag_status.warning("No content for RAG. Skipping.")
        progress_bar.progress(85, text="RAG skipped.")
    else:
        rag_status.info("Starting RAG pipeline...")
        with st.spinner("Performing RAG analysis..."):
            try:
                progress_bar.progress(55, text="Chunking content...")
                def chunk_text(text, chunk_size=700, overlap=50):
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = min(start + chunk_size, len(text))
                        chunks.append(text[start:end])
                        start += chunk_size - overlap
                    return [c for c in chunks if len(c.strip()) > 20]
                all_chunks = chunk_text(full_content_for_rag)
                if not all_chunks:
                    rag_status.warning("No usable chunks. Skipping RAG.")
                    progress_bar.progress(85, text="RAG skipped.")
                else:
                    progress_bar.progress(60, text="Generating embeddings...")
                    rag_status.info(f"Embedding {len(all_chunks)} chunks...")
                    async def get_embedding_batch(texts, model_name, batch_num, total_batches):
                        progress_bar.progress(60 + int(15 * (batch_num / total_batches)), text=f"Embedding batch {batch_num}/{total_batches}...")
                        try:
                            result = await genai.embed_content_async(model=model_name, content=texts, task_type="RETRIEVAL_DOCUMENT")
                            return result['embedding']
                        except Exception as e:
                            st.warning(f"Embedding batch {batch_num} failed: {e}")
                            return [None] * len(texts)
                    batch_size = 100
                    num_batches = (len(all_chunks) + batch_size - 1) // batch_size
                    embedding_tasks = [
                        get_embedding_batch(all_chunks[i:i+batch_size], GOOGLE_EMBEDDING_MODEL, (i // batch_size) + 1, num_batches)
                        for i in range(0, len(all_chunks), batch_size)
                    ]
                    batch_results = await asyncio.gather(*embedding_tasks, return_exceptions=True)
                    all_embeddings = []
                    for result in batch_results:
                        if not isinstance(result, Exception) and result:
                            all_embeddings.extend(result)
                    valid_indices = [i for i, emb in enumerate(all_embeddings) if emb is not None and len(emb) > 0]
                    if not valid_indices:
                        rag_status.error("Embedding failed for all chunks.")
                        progress_bar.progress(85, text="RAG failed.")
                    else:
                        all_chunks = [all_chunks[i] for i in valid_indices]
                        all_embeddings = [all_embeddings[i] for i in valid_indices]
                        rag_status.info(f"Generated {len(all_embeddings)} embeddings. Matching content...")
                        progress_bar.progress(75, text="Matching content...")
                        matched_chunks = []
                        try:
                            query_embedding = (await genai.embed_content_async(model=GOOGLE_EMBEDDING_MODEL, content=current_query, task_type="RETRIEVAL_QUERY"))['embedding']
                            k = min(15, len(all_chunks))
                            if use_faiss:
                                loop = asyncio.get_running_loop()
                                faiss_index = await loop.run_in_executor(None, build_faiss_index, all_embeddings)
                                if faiss_index:
                                    rag_status.info("Using FAISS for matching...")
                                    distances, indices = await loop.run_in_executor(None, search_faiss, faiss_index, query_embedding, k)
                                    if indices is not None:
                                        matched_chunks = [all_chunks[i] for i in indices if 0 <= i < len(all_chunks)]
                                    else:
                                        rag_status.warning("FAISS failed, using Supabase.")
                                        use_faiss = False
                                else:
                                    rag_status.warning("FAISS index failed, using Supabase.")
                                    use_faiss = False
                            if not use_faiss:
                                rag_status.info("Using Supabase for matching...")
                                if supabase_client_instance:
                                    match_params = {"query_embedding": query_embedding, "match_threshold": 0.75, "match_count": k}
                                    loop = asyncio.get_running_loop()
                                    match_response = await loop.run_in_executor(None, lambda: supabase_client_instance.rpc("match_chunks", match_params).execute())
                                    if match_response.data:
                                        matched_chunks = [row.get("chunk", row.get("content")) for row in match_response.data if row.get("chunk", row.get("content"))]
                            if not matched_chunks:
                                rag_status.warning("No relevant chunks found.")
                                comprehensive_report = "No relevant information found."
                            else:
                                rag_status.info(f"Synthesizing from {len(matched_chunks)} chunks...")
                                aggregated_relevant = "\n\n".join(matched_chunks)
                                citations = [f"{title}: {url_or_file}" for url_or_file, title in citations_map.items()]
                                synthesis_prompt = f"""You are an expert diagnostic report generator. Synthesize a comprehensive medical diagnostic report based on the patient's query and context.

Patient Query: {current_query}
Context:
---
{aggregated_relevant}
---
Instructions:
1. Identify key symptoms/issues from the query and context.
2. Analyze context information, noting any conflicts or gaps.
3. Incorporate temporal context if available.
4. Integrate wearable data if provided.
5. Structure the report in Markdown with sections: Patient Query, Analysis, Wearable Data (if applicable), Considerations, Disclaimer.
6. Use a professional tone, avoiding definitive diagnoses.
7. If information is lacking, state it clearly.
"""
                                progress_bar.progress(80, text="Synthesizing report...")
                                messages = [{"role": "user", "content": synthesis_prompt}]
                                loop = asyncio.get_running_loop()
                                response_data = await loop.run_in_executor(None, gemini_client_instance.chat, messages)
                                comprehensive_report = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Synthesis error.")
                                rag_status.success("RAG complete.")
                        except Exception as e:
                            rag_status.error(f"RAG failed: {e}")
                            comprehensive_report = f"RAG error: {e}"
            except Exception as e:
                rag_status.error(f"RAG pipeline error: {e}")
                comprehensive_report = f"RAG pipeline failed: {e}"
    progress_bar.progress(85, text="RAG complete.")

    # Patient-Friendly Summary
    st.markdown("### 4. Generating Patient Summary")
    summary_status = st.empty()
    patient_summary_report = "Summary generation failed."
    if "failed" not in comprehensive_report.lower() and "not found" not in comprehensive_report.lower():
        progress_bar.progress(90, text="Generating summary...")
        summary_status.info("Generating summary...")
        with st.spinner("Generating summary..."):
            try:
                prompt = f"""As a medical assistant, create a patient-friendly summary from this report:
{comprehensive_report}
Keep it simple, clear, and actionable, reflecting any uncertainties."""
                messages = [{"role": "user", "content": prompt}]
                loop = asyncio.get_running_loop()
                response_data = await loop.run_in_executor(None, gemini_client_instance.chat, messages)
                patient_summary_report = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Summary error.")
                summary_status.success("Summary generated.")
            except Exception as e:
                summary_status.error(f"Summary failed: {e}")
                patient_summary_report = f"Summary error: {e}"
    else:
        summary_status.warning("Skipping summary due to report issues.")
    analysis_results['comprehensive_report'] = comprehensive_report
    analysis_results['patient_summary_report'] = patient_summary_report
    analysis_results['citations'] = citations
    progress_bar.progress(100, text="Analysis complete!")
    st.success("Analysis complete!")
    return analysis_results

# --- Streamlit UI Layout ---

with st.sidebar:
    st.header("Configuration")
    query = st.text_area("1. Enter Patient Symptoms/Issue:", height=150, key="query_input")
    st.subheader("Web Search Options")
    include_urls_str = st.text_area("Include URLs (one per line, optional):", height=150, key="include_urls")
    omit_urls_str = st.text_area("Omit URLs Containing (one per line, optional):", height=150, key="omit_urls")
    search_depth = st.selectbox("Search Depth:", ["basic", "advanced"], index=1, key="search_depth")
    search_breadth = st.number_input("Search Breadth:", min_value=3, max_value=20, value=7, key="search_breadth")
    st.subheader("Reference Files")
    uploaded_files = st.file_uploader("Upload Files (PDF, DOCX, CSV, XLSX, TXT):", accept_multiple_files=True, type=['pdf', 'docx', 'csv', 'xlsx', 'xls', 'txt'], key="file_uploader")
    uploaded_wearable_file = st.file_uploader("Wearable Data CSV:", accept_multiple_files=False, type=['csv'], key="wearable_uploader")
    uploaded_survey_file = st.file_uploader("Survey Data CSV:", accept_multiple_files=False, type=['csv'], key="survey_uploader")
    st.subheader("Advanced Options")
    use_faiss = st.checkbox("Use FAISS for RAG", value=True, key="use_faiss")
    st.divider()
    submit_button = st.button("Run Analysis", type="primary", key="submit_button", use_container_width=True)

st.header("Analysis Results")
st.markdown("**Disclaimer:** This tool provides information only and is not a substitute for professional medical advice.")

default_state = {
    'analysis_complete': False, 'results': {}, 'sentiment': "", 'entities': "",
    'prediction': {}, 'explanation': "", 'classifier_trained': False,
    'vectorizer': None, 'model': None, 'background_texts_sample': None
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Train Classifier
survey_df = None
if uploaded_survey_file:
    file_details = (uploaded_survey_file.name, uploaded_survey_file.size)
    if 'last_survey_file' not in st.session_state or st.session_state.last_survey_file != file_details:
        st.session_state.classifier_trained = False
        st.session_state.vectorizer = None
        st.session_state.model = None
        st.session_state.background_texts_sample = None
        st.session_state.last_survey_file = file_details
        try:
            survey_df = pd.read_csv(uploaded_survey_file)
            st.sidebar.success(f"Loaded survey: {uploaded_survey_file.name}")
            with st.spinner("Training classifier..."):
                vectorizer, model = train_symptom_classifier(survey_df)
            if vectorizer and model:
                st.session_state.classifier_trained = True
                st.session_state.vectorizer = vectorizer
                st.session_state.model = model
                symptom_col = 'What are the current symptoms or health issues you are facing'
                if symptom_col in survey_df.columns:
                    background_texts = survey_df[symptom_col].astype(str).fillna("").tolist()
                    sample_size = min(50, len(background_texts))
                    st.session_state.background_texts_sample = np.random.choice(background_texts, sample_size, replace=False).tolist()
                st.sidebar.success("Classifier trained.")
            else:
                st.session_state.classifier_trained = False
                st.sidebar.warning("Training failed.")
        except Exception as e:
            st.sidebar.error(f"Training error: {e}")
            st.session_state.classifier_trained = False
            st.session_state.last_survey_file = None
elif 'last_survey_file' in st.session_state:
    st.session_state.classifier_trained = False
    st.session_state.vectorizer = None
    st.session_state.model = None
    st.session_state.background_texts_sample = None
    del st.session_state.last_survey_file

# Handle Analysis
if submit_button and query:
    st.session_state.analysis_complete = False
    st.session_state.results = {}
    include_urls_list = [url.strip() for url in include_urls_str.split('\n') if url.strip()]
    omit_urls_list = [url.strip() for url in omit_urls_str.split('\n') if url.strip()]
    additional_files_content = {}
    if uploaded_files:
        st.markdown("### Processing Files")
        with st.expander("File Status", expanded=True):
            for file in uploaded_files:
                content = extract_text_from_file(file)
                if content and "Error" not in content and "[Unsupported" not in content:
                    additional_files_content[file.name] = content
                    st.write(f"✅ {file.name}")
                else:
                    st.warning(f"⚠️ {file.name}: {content}")
    wearable_summary = read_wearable_data(uploaded_wearable_file)
    if wearable_summary and "Error" not in wearable_summary:
        st.markdown("### Wearable Data")
        with st.expander("Wearable Summary", expanded=False):
            st.text(wearable_summary)
    if st.session_state.classifier_trained:
        st.markdown("### Diagnosis Prediction")
        with st.spinner("Predicting..."):
            pred_diag, pred_proba = predict_diagnosis(query, st.session_state.vectorizer, st.session_state.model)
            st.session_state.prediction = {"diagnosis": pred_diag, "probabilities": pred_proba}
            st.session_state.explanation = explain_diagnosis_shap(query, st.session_state.vectorizer, st.session_state.model, st.session_state.background_texts_sample)
        st.write(f"**Predicted Diagnosis:** {pred_diag}")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Probabilities", expanded=False):
                st.json(pred_proba)
        with col2:
            with st.expander("Explanation (SHAP)", expanded=False):
                st.text(st.session_state.explanation)
    else:
        st.info("Prediction skipped (no survey data).")
    st.divider()
    st.header("Main Analysis Pipeline")
    try:
        analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
        subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config)
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
        st.session_state.results = analysis_results
        st.session_state.sentiment = analysis_results['sentiment']
        st.session_state.entities = analysis_results['entities']
        st.session_state.analysis_complete = True
        save_query_history(query, st.session_state.prediction.get('diagnosis', 'N/A'), analysis_results['sentiment'], analysis_results['entities'])
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        logger.error(f"Pipeline error: {e}\n{traceback.format_exc()}")

# Handle Analysis completion
if st.session_state.analysis_complete:
    st.divider()  # Adds a horizontal line to separate sections
    # Display analysis results
    st.markdown("### Comprehensive Report")
    st.write(st.session_state.results.get('comprehensive_report', 'No report available.'))
    st.markdown("### Patient-Friendly Summary")
    st.write(st.session_state.results.get('patient_summary_report', 'No summary available.'))
    if st.session_state.results.get('citations'):
        st.markdown("### Citations")
        for citation in st.session_state.results['citations']:
            st.write(f"- {citation}")
