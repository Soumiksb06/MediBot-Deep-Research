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
- Asynchronous operations handled within Streamlit.
- API Key management via st.secrets.
"""

import os
import sys
import asyncio
import logging
import csv
import io # Used for handling file uploads
import traceback # For detailed error logging
from datetime import datetime
from dotenv import load_dotenv # Still useful for local development if needed

# Streamlit
import streamlit as st

# Rich for console output (optional in Streamlit, but kept if internal logic uses it)
from rich.console import Console
from rich.logging import RichHandler # For better logging display if needed

# File Processing
import PyPDF2
import docx
import pandas as pd

# API Clients & Services (Assuming these are structured as in the original script)
# It's better if these modules are installable or in the same directory structure.
# If they are not modules, you might need to copy the class definitions here.
try:
    # NOTE: Ensure these paths are correct relative to where streamlit_app.py is run
    # Or that these packages are properly installed in the environment.
    from web_agent.src.services.web_search import WebSearchService
    from web_agent.src.models.search_models import SearchConfig
    from subject_analyzer.src.services.tavily_client import TavilyClient
    from subject_analyzer.src.services.tavily_extractor import TavilyExtractor
    from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer
    from subject_analyzer.src.services.gemini_client import GeminiClient # IMPORTANT: Use the Gemini client
    from subject_analyzer.src.models.analysis_models import AnalysisConfig
except ImportError as e:
    st.error(f"Required custom agent modules (web_agent, subject_analyzer) not found: {e}. "
             "Please ensure they are installed or in the correct Python path.")
    # Add placeholder classes or exit if these are critical
    logger.error(f"ImportError for custom modules: {e}\n{traceback.format_exc()}")
    st.stop()


# Supabase
from supabase import create_client, Client

# ML/NLP & Data Science
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import shap
import faiss

# Google Generative AI (for Gemini)
try:
    import google.generativeai as genai
    from google.generativeai import types as genai_types # Explicit import
except ImportError:
    st.error("Google Generative AI SDK not found. Please install it: `pip install google-generativeai`")
    st.stop()

# --- Configuration & Initialization ---

# Configure logging (optional for Streamlit, but good practice)
# Log to a file and optionally display important logs in Streamlit
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="streamlit_medical_agent.log",
    filemode="a" # Append to log file
)
logger = logging.getLogger(__name__)

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide", page_title="Medical Diagnostic Agent")
st.title("🩺 Medical Diagnostic Agent")
st.markdown("Enter patient symptoms and configure options to generate a diagnostic report.")

# --- Load API Keys and Config from st.secrets ---
# Ensure these are set in your Streamlit Cloud secrets or a local .streamlit/secrets.toml file
try:
    GEMINI_API_KEY = st.secrets["api_keys"]["GEMINI_API_KEY"]
    TAVILY_API_KEY = st.secrets["api_keys"]["TAVILY_API_KEY"]
    SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]
    # Optional: Specify models via secrets or use defaults
    GEMINI_MODEL_NAME = st.secrets.get("models", {}).get("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") # Use a standard model
    GOOGLE_EMBEDDING_MODEL = st.secrets.get("models", {}).get("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
except KeyError as e:
    st.error(f"Missing required secret: {e}. Please configure your `secrets.toml` file or Streamlit Cloud secrets.")
    logger.error(f"Missing secret: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading secrets: {e}")
    logger.error(f"Error loading secrets: {e}")
    st.stop()


# --- Global Clients and Resources (Cached) ---

@st.cache_resource
def get_gemini_client():
    """Initializes and returns the Gemini client."""
    try:
        # Configure the SDK
        genai.configure(api_key=GEMINI_API_KEY)

        # Use the custom GeminiClient wrapper if it provides necessary abstractions
        # Ensure AnalysisConfig is available and correctly defined
        analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
        client_wrapper = GeminiClient(api_key=GEMINI_API_KEY, config=analysis_config)
        logger.info("Custom GeminiClient wrapper initialized successfully.")
        return client_wrapper
    except NameError:
         st.error("AnalysisConfig class not found. Ensure subject_analyzer module is correctly loaded.")
         logger.error("NameError: AnalysisConfig not defined.")
         st.stop()
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
        # Ensure SearchConfig is available
        search_config = SearchConfig()
        search_service = WebSearchService(search_client, search_config)
        logger.info("Tavily clients initialized successfully.")
        return search_service, extractor
    except NameError:
         st.error("SearchConfig class not found. Ensure web_agent module is correctly loaded.")
         logger.error("NameError: SearchConfig not defined.")
         st.stop()
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
    base = ''.join(c if c.isalnum() or c.isspace() else '_' for c in query[:50]) # Limit length
    base = base.strip().replace(' ', '_')
    return f"{base or 'report'}_{suffix}"

# --- Gemini-based Replacements for NLTK/spaCy ---

async def analyze_sentiment_gemini(query, client):
    """Performs sentiment analysis using Gemini."""
    prompt = f"""Analyze the sentiment of the following patient query. Respond with a simple classification (Positive, Negative, Neutral) and a brief explanation.

Patient Query: "{query}"

Sentiment Classification:
Explanation:"""
    try:
        # Assuming gemini_client is the custom wrapper with a 'chat' method
        if hasattr(client, 'chat'):
             messages = [{"role": "user", "content": prompt}]
             # Ensure the chat method handles async or is called appropriately
             # If client.chat is sync, it might block the event loop if called directly in async func
             # Consider running sync methods in an executor if needed:
             # loop = asyncio.get_event_loop()
             # response_data = await loop.run_in_executor(None, client.chat, messages)
             response_data = client.chat(messages) # Assuming it's okay or handles async internally

             sentiment_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse sentiment response.")
             return sentiment_text
        else:
             # Fallback or direct SDK usage if the wrapper isn't the primary interface
             logger.warning("Using fallback for sentiment analysis - GeminiClient 'chat' method preferred.")
             sdk_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
             response = await sdk_model.generate_content_async(prompt)
             return response.text

    except AttributeError:
         logger.error("Gemini client object does not have the expected 'chat' method.")
         return "Error: Gemini client misconfigured (no chat method)."
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
        # Assuming gemini_client is the custom wrapper with a 'chat' method
        if hasattr(client, 'chat'):
             messages = [{"role": "user", "content": prompt}]
             # Similar async consideration as in sentiment analysis
             # loop = asyncio.get_event_loop()
             # response_data = await loop.run_in_executor(None, client.chat, messages)
             response_data = client.chat(messages) # Assuming okay

             entities_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse entities response.")
             return entities_text
        else:
             logger.warning("Using fallback for entity extraction - GeminiClient 'chat' method preferred.")
             sdk_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
             response = await sdk_model.generate_content_async(prompt)
             return response.text

    except AttributeError:
         logger.error("Gemini client object does not have the expected 'chat' method.")
         return "Error: Gemini client misconfigured (no chat method)."
    except Exception as e:
        logger.error(f"Gemini entity extraction failed: {e}\n{traceback.format_exc()}")
        return f"Error during entity extraction: {e}"

# --- Core Logic Adapted from Original Script ---

def extract_text_from_file(uploaded_file):
    """Extracts text from various uploaded file types."""
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        bytes_data = uploaded_file.getvalue()

        if ext == '.pdf':
            text = ""
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text: # Add text only if extraction was successful
                    text += page_text + "\n" # Add newline between pages
            return text
        elif ext == '.docx':
            doc = docx.Document(io.BytesIO(bytes_data))
            return "\n".join(para.text for para in doc.paragraphs)
        elif ext == '.csv':
            # Read CSV data using pandas, return as string
            # Try different encodings if utf-8 fails
            try:
                df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin-1')
            return df.to_csv(index=False) # Return CSV content as string
        elif ext in ['.xls', '.xlsx']:
             # Specify engine based on extension
             engine = 'openpyxl' if ext == '.xlsx' else 'xlrd' # xlrd needed for .xls
             try:
                 df = pd.read_excel(io.BytesIO(bytes_data), engine=engine)
                 return df.to_csv(index=False) # Return as CSV string for consistency
             except ImportError as imp_err:
                  st.error(f"Missing library for {ext} files: {imp_err}. Please install 'openpyxl' for .xlsx or 'xlrd' for .xls.")
                  logger.error(f"Missing Excel library: {imp_err}")
                  return ""
             except Exception as excel_err:
                  st.warning(f"Could not read Excel file {uploaded_file.name} with {engine}: {excel_err}")
                  logger.error(f"Excel read error ({uploaded_file.name}, {engine}): {excel_err}")
                  # Try the other engine as a fallback? Maybe too complex.
                  return ""
        elif ext == '.txt':
             # Decode bytes to string, trying common encodings
             try:
                 return bytes_data.decode('utf-8')
             except UnicodeDecodeError:
                 try:
                     return bytes_data.decode('latin-1')
                 except Exception as decode_err:
                     logger.warning(f"Could not decode TXT file {uploaded_file.name}: {decode_err}")
                     return "[Could not decode TXT file]"
        else:
            logger.warning(f"Unsupported file type: {ext} for file {uploaded_file.name}")
            return f"[Unsupported file type: {ext}]"
    except PyPDF2.errors.PdfReadError as pdf_err:
         logger.error(f"Failed to read PDF file {uploaded_file.name}: {pdf_err}")
         st.warning(f"Could not read PDF file: {uploaded_file.name}. It might be corrupted or password-protected.")
         return ""
    except Exception as e:
        logger.error(f"Failed to extract text from {uploaded_file.name}: {e}\n{traceback.format_exc()}")
        st.warning(f"Could not process file: {uploaded_file.name}. Error: {e}")
        return ""

@st.cache_data # Cache the trained model and vectorizer
def train_symptom_classifier(survey_data_df):
    """Trains a symptom classifier from uploaded survey data."""
    if survey_data_df is None or survey_data_df.empty:
        logger.warning("Survey data is empty, cannot train classifier.")
        return None, None

    try:
        # Ensure required columns exist (adjust names if needed)
        symptom_col = 'What are the current symptoms or health issues you are facing'
        history_col = 'Medical Health History'
        if symptom_col not in survey_data_df.columns or history_col not in survey_data_df.columns:
            st.error(f"Survey data CSV must contain columns: '{symptom_col}' and '{history_col}'")
            logger.error("Missing required columns in survey data.")
            return None, None

        df = survey_data_df.copy()
        # Explicitly convert to string and handle potential non-string data
        symptoms = df[symptom_col].astype(str).fillna("").tolist()
        labels = df[history_col].astype(str).fillna("None").tolist()


        # Basic processing: use the first condition if multiple are listed
        processed_labels = [label.split(',')[0].strip() if isinstance(label, str) else 'None' for label in labels]

        # Filter out entries with no symptoms or 'None' labels if desired
        valid_data = [(s, l) for s, l in zip(symptoms, processed_labels) if s.strip() and l != 'None']
        if not valid_data:
            logger.warning("No valid symptom/label pairs found in survey data for training.")
            st.warning("Survey data contains no valid symptom/diagnosis pairs for training.")
            return None, None

        texts, final_labels = zip(*valid_data)

        if not texts or not final_labels:
            logger.warning("Not enough valid data to train classifier after filtering.")
            return None, None

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=2) # Limit features, require min frequency
        X = vectorizer.fit_transform(texts)

        # Check if vocabulary is empty
        if not vectorizer.vocabulary_:
             logger.warning("TF-IDF Vectorizer vocabulary is empty. Check input data.")
             st.warning("Could not build a feature vocabulary from the survey data. Check symptom descriptions.")
             return None, None

        model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear') # Add class_weight and specify solver
        model.fit(X, final_labels)
        logger.info("Symptom classifier trained successfully.")
        return vectorizer, model

    except Exception as e:
        st.error(f"Error training symptom classifier: {e}")
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
        # Sort probabilities descending
        sorted_proba = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        logger.info(f"Predicted diagnosis '{pred}' for query.")
        return pred, sorted_proba
    except Exception as e:
        logger.error(f"Diagnosis prediction failed: {e}")
        return f"Prediction error: {e}", {}

def explain_diagnosis_shap(query, vectorizer, model, background_texts=None):
    """
    Generates SHAP explanation for the prediction.
    Requires background_texts (a sample of texts used for training) for robust explanation.
    """
    if not vectorizer or not model:
        return "Model not trained, cannot explain."
    try:
        # Check if the query contains any known features
        X_query_check = vectorizer.transform([query])
        if X_query_check.nnz == 0: # Check if the transformed vector is all zeros
             logger.warning("Query contains no features known to the model. Cannot generate SHAP explanation.")
             return "Explanation not available (query contains no known terms)."

        # SHAP requires dense data for explainers
        X_query = X_query_check.toarray()

        # Define the prediction function for SHAP
        predict_proba_dense = lambda x: model.predict_proba(x)

        # --- Select Explainer and Background Data ---
        # Option 1: LinearExplainer (Fast, needs good background)
        # Option 2: KernelExplainer (Slower, more flexible background)

        # Using KernelExplainer as it's more robust without direct access to the full training matrix 'X'
        # We need a representative background dataset.
        if background_texts and len(background_texts) > 10: # Use provided texts if available and sufficient
             # Transform background texts using the *same* vectorizer
             background_data_sparse = vectorizer.transform(background_texts)
             # Use kmeans for summary, ensuring enough samples and features
             if background_data_sparse.shape[0] > 1 and background_data_sparse.shape[1] > 0:
                  num_clusters = min(10, background_data_sparse.shape[0]) # Limit clusters
                  background_summary = shap.kmeans(background_data_sparse.toarray(), num_clusters)
                  logger.info(f"Using KernelExplainer with k-means background summary ({num_clusters} clusters).")
             else:
                  logger.warning("Insufficient background data for k-means, using zero vector background (less accurate).")
                  background_summary = np.zeros((1, X_query.shape[1]))

        else: # Fallback if no good background texts provided
            logger.warning("No background texts provided for SHAP, using zero vector background (less accurate).")
            # Using a zero vector is a poor approximation, but a fallback.
            background_summary = np.zeros((1, X_query.shape[1]))


        explainer = shap.KernelExplainer(predict_proba_dense, background_summary)

        # Calculate SHAP values for the query
        # Specify l1_reg to potentially speed up by focusing on top features
        shap_values_kernel = explainer.shap_values(X_query, nsamples='auto') # Let SHAP determine samples

        # KernelExplainer returns shap_values for each class.
        # We need the values for the predicted class.
        predicted_class = model.predict(X_query)[0]
        try:
             # Find the index corresponding to the predicted class name
             predicted_class_index = list(model.classes_).index(predicted_class)
             shap_values = shap_values_kernel[predicted_class_index]
        except ValueError:
             logger.error(f"Predicted class '{predicted_class}' not found in model classes. Cannot select SHAP values.")
             return "Error: Could not map prediction to SHAP values."
        except IndexError:
             logger.error(f"SHAP values structure unexpected. Cannot select values for predicted class.")
             return "Error: Unexpected SHAP values format."


        # Process SHAP values to get top features
        feature_names = vectorizer.get_feature_names_out()
        # Ensure shap_values is treated as a 1D array for the single instance
        shap_values_instance = shap_values[0] if isinstance(shap_values, (list, np.ndarray)) and np.ndim(shap_values) > 1 else shap_values

        # Get indices sorted by absolute SHAP value magnitude
        sorted_indices = np.argsort(np.abs(shap_values_instance))[::-1]

        explanation = f"Top contributing features for prediction '{predicted_class}' (SHAP values):\n"
        num_features_to_show = 5
        features_shown = 0
        for idx in sorted_indices:
            # Ensure index is within bounds and shap value is non-zero
            if idx < len(feature_names) and not np.isclose(shap_values_instance[idx], 0):
                 feature = feature_names[idx]
                 shap_val = shap_values_instance[idx]
                 explanation += f"- {feature}: {shap_val:.4f}\n"
                 features_shown += 1
                 if features_shown >= num_features_to_show:
                     break

        if features_shown == 0:
            explanation += "No significant features identified by SHAP."

        logger.info("SHAP explanation generated successfully using KernelExplainer.")
        return explanation

    except ImportError:
        st.warning("SHAP library not installed. Cannot provide explanations. `pip install shap`")
        logger.warning("SHAP library not installed.")
        return "SHAP library not installed. Cannot provide explanations."
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")
        logger.error(f"SHAP explanation failed: {e}\n{traceback.format_exc()}")
        return f"SHAP explanation failed: {e}"

def save_query_history(query, diagnosis, sentiment, entities):
    """Saves query details to a CSV file."""
    filename = "query_history.csv"
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.path.getsize(filename) == 0: # Check if file is empty too
                writer.writerow(["timestamp", "query", "predicted_diagnosis", "sentiment_summary", "entities_summary"])
            # Store summaries or key parts of sentiment/entities if they are long strings
            sentiment_summary = str(sentiment).split('\n')[0] # Get first line, ensure string
            entities_summary = str(entities).split('\n')[0]   # Get first line, ensure string
            writer.writerow([datetime.now().isoformat(), query, diagnosis, sentiment_summary, entities_summary])
        logger.info(f"Query history saved for query: {query[:50]}...")
        return True
    except Exception as e:
        logger.error(f"Failed to save query history: {e}")
        st.warning(f"Could not save query history: {e}")
        return False

def visualize_query_trends():
    """Generates and returns matplotlib figures for query trends."""
    filename = "query_history.csv"
    figs = {}
    if not os.path.exists(filename):
        st.info("No query history available for visualization (`query_history.csv` not found).")
        return figs
    try:
        # Check if file is empty before reading
        if os.path.getsize(filename) == 0:
             st.info("Query history file (`query_history.csv`) is empty.")
             return figs

        df = pd.read_csv(filename)
        if df.empty:
            st.info("Query history is empty.")
            return figs

        # Convert timestamp safely
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']) # Drop rows where conversion failed
        if df.empty:
             st.info("No valid timestamps found in query history.")
             return figs
        df = df.sort_values("timestamp")

        # --- Plot 1: Diagnosis Frequency ---
        if 'predicted_diagnosis' in df.columns:
            # Exclude placeholder/error values before counting
            valid_diagnoses = df[~df['predicted_diagnosis'].isin(["N/A - Model not trained", "Model not trained", "Prediction error"])]['predicted_diagnosis']
            diag_counts = valid_diagnoses.value_counts().nlargest(10) # Top 10 valid diagnoses
            if not diag_counts.empty:
                fig_diag, ax_diag = plt.subplots(figsize=(8, 5))
                sns.barplot(x=diag_counts.index, y=diag_counts.values, ax=ax_diag, palette="viridis")
                ax_diag.set_title("Top 10 Predicted Diagnoses Frequency")
                ax_diag.set_xlabel("Diagnosis")
                ax_diag.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                figs['diagnosis_frequency'] = fig_diag
                logger.info("Generated diagnosis frequency plot.")
            else:
                 st.info("No valid predicted diagnosis data found for visualization.")

        # --- Plot 2: Sentiment Trend (Placeholder) ---
        st.info("Sentiment trend visualization requires numerical sentiment scores (currently using text-based sentiment).")


        # --- Plot 3: Entity Distribution ---
        if 'entities_summary' in df.columns:
             entity_list = []
             # Extract entities assuming they are listed after "Medical Entities:" or similar markers
             for entities_str in df['entities_summary'].fillna("").astype(str):
                 # Handle potential variations in Gemini's output format
                 if "Medical Entities:" in entities_str:
                     actual_entities_part = entities_str.split("Medical Entities:", 1)[1]
                 elif entities_str.startswith("- "): # Check if it starts like a list
                      actual_entities_part = entities_str
                 else:
                      actual_entities_part = "" # Skip if format is unclear

                 if actual_entities_part:
                     # Split by newline, remove '- ', strip whitespace
                     entities = [e.strip().lstrip('-').strip() for e in actual_entities_part.split('\n') if e.strip()]
                     # Further clean-up (remove brackets if present)
                     entities = [e.replace('[','').replace(']','') for e in entities if e]
                     entity_list.extend(entities)

             if entity_list:
                 # Convert to lower case for consistent counting
                 entity_list_lower = [e.lower() for e in entity_list]
                 entity_counts = Counter(entity_list_lower).most_common(15) # Top 15
                 if entity_counts:
                     labels, sizes = zip(*entity_counts)
                     fig_ent, ax_ent = plt.subplots(figsize=(8, 8))
                     # Prevent labels overlapping
                     wedges, texts, autotexts = ax_ent.pie(sizes, autopct="%1.1f%%", startangle=140, pctdistance=0.85)
                     ax_ent.legend(wedges, labels, title="Entities", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                     plt.setp(autotexts, size=8, weight="bold")
                     ax_ent.set_title("Top 15 Extracted Medical Entities Distribution")
                     plt.tight_layout()
                     figs['entities_distribution'] = fig_ent
                     logger.info("Generated entities distribution plot.")
                 else:
                      st.info("No significant medical entities found to plot.")
             else:
                  st.info("Could not extract entities from history for plotting.")


        return figs

    except FileNotFoundError:
        st.info("Query history file (`query_history.csv`) not found.")
        return figs
    except pd.errors.EmptyDataError:
         st.info("Query history file (`query_history.csv`) is empty.")
         return figs
    except Exception as e:
        st.error(f"Error generating visualizations: {e}")
        logger.error(f"Visualization generation failed: {e}\n{traceback.format_exc()}")
        return figs

def build_faiss_index(embeddings):
    """Builds a FAISS index."""
    if not embeddings or not isinstance(embeddings[0], (np.ndarray, list)):
        logger.warning("Invalid or empty embeddings for FAISS.")
        return None
    try:
        embeddings_array = np.array(embeddings).astype('float32')
        if embeddings_array.ndim != 2:
             logger.error(f"Embeddings array has incorrect dimensions: {embeddings_array.ndim}")
             st.error("Embeddings data has incorrect format for FAISS.")
             return None
        dim = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dim) # Using L2 distance
        index.add(embeddings_array)
        logger.info(f"Built FAISS index with {index.ntotal} vectors of dimension {dim}.")
        return index
    except ImportError:
         st.warning("FAISS library not installed. Cannot use FAISS optimization. `pip install faiss-cpu` (or `faiss-gpu`)")
         logger.warning("FAISS library not installed.")
         return None
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        logger.error(f"FAISS index build failed: {e}\n{traceback.format_exc()}")
        return None

def search_faiss(index, query_embedding, k):
    """Searches the FAISS index."""
    if index is None:
        logger.warning("Attempted to search FAISS with no index.")
        return None, None
    try:
        query_vec = np.array(query_embedding).reshape(1, -1).astype('float32')
        # Ensure k is not greater than the number of items in the index
        k_adjusted = min(k, index.ntotal)
        if k_adjusted <= 0:
             logger.warning("Adjusted k for FAISS search is zero or negative.")
             return np.array([]), np.array([])

        distances, indices = index.search(query_vec, k_adjusted)
        logger.info(f"FAISS search completed, found {len(indices[0])} indices for k={k_adjusted}.")
        # Return only valid indices (should be guaranteed by FAISS if k_adjusted > 0)
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
        # Try reading with common encodings
        try:
             df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
             uploaded_file.seek(0) # Reset file pointer
             df = pd.read_csv(uploaded_file, encoding='latin-1')

        summary = "Wearable Data Summary:\n"
        processed_columns = []

        # Standardize column names (convert to lower, replace spaces/symbols)
        df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]+', '_', regex=True).str.strip('_')

        if "heart_rate" in df.columns:
            # Attempt to convert heart_rate column to numeric, coercing errors
            hr_numeric = pd.to_numeric(df["heart_rate"], errors='coerce')
            hr_numeric = hr_numeric.dropna() # Remove rows where conversion failed
            if not hr_numeric.empty:
                 avg_hr = hr_numeric.mean()
                 min_hr = hr_numeric.min()
                 max_hr = hr_numeric.max()
                 summary += f"- Heart Rate: Avg={avg_hr:.1f}, Min={min_hr:.1f}, Max={max_hr:.1f} bpm\n"
                 processed_columns.append("heart_rate")
            else:
                 logger.warning("Heart rate column found but contains no valid numeric data.")


        if "steps" in df.columns:
            steps_numeric = pd.to_numeric(df["steps"], errors='coerce').dropna()
            if not steps_numeric.empty:
                 total_steps = steps_numeric.sum()
                 avg_steps = steps_numeric.mean()
                 summary += f"- Steps: Total={total_steps:.0f}, Avg={avg_steps:.0f}\n"
                 processed_columns.append("steps")
            else:
                 logger.warning("Steps column found but contains no valid numeric data.")


        # Add more metrics if needed (e.g., sleep, spo2) - example for sleep
        if "sleep_duration_hours" in df.columns: # Example column name
             sleep_numeric = pd.to_numeric(df["sleep_duration_hours"], errors='coerce').dropna()
             if not sleep_numeric.empty:
                  avg_sleep = sleep_numeric.mean()
                  summary += f"- Sleep: Avg={avg_sleep:.1f} hours\n"
                  processed_columns.append("sleep_duration_hours")


        if not processed_columns:
             summary += "- No recognized columns (e.g., 'heart_rate', 'steps', 'sleep_duration_hours') with valid numeric data found."
        logger.info(f"Wearable data processed. Found metrics for: {', '.join(processed_columns)}")
        return summary
    except Exception as e:
        st.warning(f"Could not process wearable data file ({uploaded_file.name}): {e}")
        logger.error(f"Wearable data processing error: {e}\n{traceback.format_exc()}")
        return "Error processing wearable data."


# --- MedicalTask Class (Adapted for Streamlit) ---
# This class holds the state for a single analysis run.
# We might instantiate this within the main app logic or pass state differently.
# For simplicity, let's adapt its methods to be called directly.

async def run_analysis_pipeline(
    original_query,
    include_urls,
    omit_urls,
    additional_files_content, # Dict[filename, content]
    wearable_data_summary,
    search_depth,
    search_breadth,
    use_faiss,
    gemini_client_instance, # Pass the initialized client
    supabase_client_instance, # Pass the initialized client
    search_service_instance, # Pass the initialized client
    extractor_instance, # Pass the initialized client
    subject_analyzer_instance # Pass the initialized analyzer
    ):
    """
    Runs the full medical analysis pipeline, adapted from MedicalTask.
    Returns generated reports and citations.
    Uses Streamlit widgets for progress updates.
    """
    st.info("Starting analysis pipeline...")
    analysis_results = {} # Store results here
    current_date = datetime.today().strftime("%Y-%m-%d")
    current_query = original_query # Start with the original query
    progress_bar = st.progress(0, text="Initializing...")

    # --- 1. Subject Analysis ---
    st.markdown("### 1. Analyzing Query Subject...")
    progress_bar.progress(5, text="Analyzing patient query context...")
    analysis = {}
    try:
        with st.spinner("Analyzing patient query context..."):
            # Assuming SubjectAnalyzer uses the passed gemini_client internally
            # Ensure subject_analyzer_instance is not None
            if not subject_analyzer_instance:
                 raise ValueError("Subject Analyzer instance is not available.")
            analysis = subject_analyzer_instance.analyze(f"{current_query} (as of {current_date})")
            analysis_results['subject_analysis'] = analysis
            logger.info("Subject analysis successful.")
            st.success("Query analysis complete.")
            # Display Agent's Understanding
            with st.expander("Agent's Understanding", expanded=True):
                st.write(f"**Patient Query:** {original_query}")
                st.write(f"**Identified Medical Issue:** {analysis.get('main_subject', 'Unknown Issue')}")
                temporal = analysis.get("temporal_context", {})
                if temporal:
                    st.write("**Temporal Context:**")
                    for key, value in temporal.items():
                        st.write(f"- {key.capitalize()}: {value}")
                needs = analysis.get("What_needs_to_be_researched", [])
                st.write("**Key aspects to investigate:**")
                st.write(f"{', '.join(needs) if needs else 'None specified'}")

    except Exception as e:
        st.error(f"Subject analysis failed: {e}")
        logger.error(f"Subject analysis failed: {e}\n{traceback.format_exc()}")
        analysis_results['subject_analysis_error'] = str(e)
        # Decide whether to stop or continue
        st.warning("Continuing without full subject analysis.")
        analysis = {} # Reset analysis to avoid errors later
    progress_bar.progress(15, text="Query analysis finished.")

    # --- 2. Web Search and Extraction ---
    st.markdown("### 2. Searching and Extracting Information...")
    search_results_dict = {}
    extracted_content_dict = {}
    search_performed = False
    search_status = st.empty() # Placeholder for status updates

    if not include_urls:
        topics_to_search = [analysis.get("main_subject", current_query)] # Fallback to query
        topics_to_search += analysis.get("What_needs_to_be_researched", [])
        topics_to_search = list(set(filter(None, topics_to_search))) # Unique, non-empty topics

        if not topics_to_search:
            search_status.warning("No specific topics identified for web search based on analysis.")
        else:
            search_performed = True
            search_status.info(f"Starting web search for: {', '.join(topics_to_search)}")
            num_topics = len(topics_to_search)
            with st.spinner(f"Searching web and extracting content for {num_topics} topics..."):
                for i, topic in enumerate(topics_to_search):
                    progress_val = 15 + int(35 * (i / num_topics)) # Progress: 15% -> 50%
                    progress_bar.progress(progress_val, text=f"Searching & Extracting for: {topic}...")
                    search_status.info(f"Searching & Extracting for: {topic}...")
                    # st.write(f"--- Searching for: {topic} ---") # Can be verbose
                    try:
                        # Ensure search_service_instance is not None
                        if not search_service_instance:
                             raise ValueError("Search Service instance is not available.")
                        response = search_service_instance.search_subject(
                            topic, "medical", search_depth=search_depth, results=search_breadth
                        )
                        results = response.get("results", [])
                        # Filter results by omit_urls
                        filtered_results = [
                            res for res in results
                            if res.get("url") and not any(omit.lower() in res.get("url").lower() for omit in omit_urls)
                        ]
                        search_results_dict[topic] = filtered_results
                        logger.info(f"Found {len(filtered_results)} relevant results for '{topic}'.")

                        # Extract content for this topic's URLs
                        urls_to_extract = [res.get("url") for res in filtered_results if res.get("url")]
                        if urls_to_extract:
                             # Ensure extractor_instance is not None
                             if not extractor_instance:
                                 raise ValueError("Extractor instance is not available.")
                             extraction_response = extractor_instance.extract(
                                 urls=urls_to_extract,
                                 extract_depth="advanced", # Or make configurable
                                 include_images=False
                             )
                             extracted_content_dict[topic] = extraction_response.get("results", [])
                             # Log extraction success/failure counts
                             failed_count = sum(1 for item in extracted_content_dict[topic] if item.get("error"))
                             logger.info(f"Extracted content for {len(urls_to_extract) - failed_count}/{len(urls_to_extract)} URLs for topic '{topic}'.")
                             if failed_count > 0: logger.warning(f"Failed to extract content from {failed_count} URLs for '{topic}'.")

                    except Exception as e:
                        st.error(f"Search/Extraction failed for topic '{topic}': {e}")
                        logger.error(f"Search/Extraction failed for '{topic}': {e}\n{traceback.format_exc()}")
                        search_results_dict[topic] = []
                        extracted_content_dict[topic] = []
    else:
        # Use user-provided URLs
        search_performed = True
        search_status.info(f"Using {len(include_urls)} user-provided URLs.")
        filtered_urls = [url for url in include_urls if not any(omit.lower() in url.lower() for omit in omit_urls)]
        logger.info(f"Filtered user URLs: {len(filtered_urls)} remaining.")
        search_results_dict["User Provided"] = [{"title": "User Provided", "url": url, "score": "N/A"} for url in filtered_urls]

        if filtered_urls:
            progress_bar.progress(30, text="Extracting from provided URLs...")
            with st.spinner("Extracting content from provided URLs..."):
                try:
                    if not extractor_instance:
                         raise ValueError("Extractor instance is not available.")
                    extraction_response = extractor_instance.extract(
                        urls=filtered_urls,
                        extract_depth="advanced",
                        include_images=False
                    )
                    extracted_content_dict["User Provided"] = extraction_response.get("results", [])
                    failed_count = sum(1 for item in extracted_content_dict["User Provided"] if item.get("error"))
                    logger.info(f"Extracted content for {len(filtered_urls) - failed_count}/{len(filtered_urls)} user URLs.")
                    if failed_count > 0: st.warning(f"Failed to extract content from {failed_count} provided URLs.")
                except Exception as e:
                    st.error(f"Extraction failed for user provided URLs: {e}")
                    logger.error(f"Extraction failed for user URLs: {e}\n{traceback.format_exc()}")
                    extracted_content_dict["User Provided"] = []

    analysis_results['search_results'] = search_results_dict
    analysis_results['extracted_content'] = extracted_content_dict
    if search_performed:
        search_status.success("Search and extraction phase complete.")
    else:
        search_status.info("No web search performed (either used provided URLs or no topics found).")
    progress_bar.progress(50, text="Search & Extraction finished.")


    # --- 3. RAG Analysis (Embeddings, Supabase/FAISS, Gemini Synthesis) ---
    st.markdown("### 3. Performing RAG Analysis...")
    rag_status = st.empty()
    comprehensive_report = "RAG analysis did not run or failed."
    citations = []
    all_chunks = []
    all_embeddings = []

    # Aggregate content from web extraction and additional files
    full_content_for_rag = ""
    citations_map = {} # url/filename -> title/description

    # Add extracted web content
    for topic, items in extracted_content_dict.items():
        for item in items:
            url = item.get("url", "No URL")
            title = item.get("title", "No Title")
            # Prioritize 'text' but fallback to 'raw_content'
            content = item.get("text") or item.get("raw_content", "")
            if content and isinstance(content, str) and content.strip(): # Ensure content is usable string
                # Truncate long content to avoid excessive RAG context? Optional.
                # max_content_len = 5000
                # content = content[:max_content_len] + ("..." if len(content) > max_content_len else "")
                full_content_for_rag += f"\n\n=== Content from {title} ({url}) ===\n{content}\n"
                if url != "No URL": citations_map[url] = title

    # Add content from uploaded files
    for filename, content in additional_files_content.items():
         if content and isinstance(content, str) and content.strip():
             full_content_for_rag += f"\n\n=== Content from Uploaded File: {filename} ===\n{content}\n"
             citations_map[filename] = f"Uploaded File: {filename}" # Add file to citations map

    # Add wearable data summary if available
    if wearable_data_summary and isinstance(wearable_data_summary, str) and wearable_data_summary.strip():
         full_content_for_rag += f"\n\n=== Wearable Data Summary ===\n{wearable_data_summary}\n"
         citations_map["Wearable Data"] = "Wearable Data Summary"


    if not full_content_for_rag.strip():
        rag_status.warning("No content available for RAG analysis (no web results or file content). Skipping RAG.")
        progress_bar.progress(85, text="RAG skipped (no content).")
    else:
        rag_status.info("Starting RAG pipeline (Chunking, Embedding, Matching, Synthesis)...")
        with st.spinner("Performing RAG analysis... (this may take a while)"):
            try:
                # Chunking
                progress_bar.progress(55, text="Chunking content...")
                def chunk_text(text, chunk_size=700, overlap=50): # Smaller chunk size, add overlap
                    # Use a text splitter for more robust chunking if needed (e.g., from langchain)
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunks.append(text[start:end])
                        start += chunk_size - overlap
                    # Filter out very small or empty chunks
                    chunks = [c for c in chunks if len(c.strip()) > 20] # Min length filter
                    logger.info(f"Chunked content into {len(chunks)} chunks.")
                    if not chunks: logger.warning("Chunking resulted in zero valid chunks.")
                    return chunks

                all_chunks = chunk_text(full_content_for_rag)

                if not all_chunks:
                     rag_status.warning("Content chunking resulted in no usable chunks. Skipping RAG.")
                     progress_bar.progress(85, text="RAG skipped (chunking failed).")

                else:
                    # Embedding Generation (using Gemini Embeddings)
                    progress_bar.progress(60, text=f"Generating embeddings for {len(all_chunks)} chunks...")
                    rag_status.info(f"Generating embeddings for {len(all_chunks)} text chunks...")

                    async def get_embedding_batch(texts, model_name, batch_num, total_batches):
                        """Helper to get embeddings for a batch with progress."""
                        try:
                            # Use genai.embed_content for batching
                            progress_bar.progress(60 + int(15 * (batch_num / total_batches)), text=f"Embedding batch {batch_num}/{total_batches}...")
                            result = await genai.embed_content_async( # Use async version
                                model=model_name,
                                content=texts,
                                task_type="RETRIEVAL_DOCUMENT" # Specify task type
                            )
                            return result['embedding'] # Assuming result structure
                        except Exception as e:
                            logger.error(f"Batch embedding failed (Batch {batch_num}): {e}")
                            st.warning(f"Embedding failed for batch {batch_num}. Some content may be ignored.")
                            return [None] * len(texts) # Indicate failure for each text

                    # Process in batches to avoid hitting API limits
                    batch_size = 100 # Gemini batch limit can be up to 100
                    all_embeddings = []
                    num_batches = (len(all_chunks) + batch_size - 1) // batch_size
                    for i in range(0, len(all_chunks), batch_size):
                        batch_texts = all_chunks[i:i+batch_size]
                        current_batch_num = (i // batch_size) + 1
                        # Run async batch embedding
                        try:
                             batch_embeddings = await get_embedding_batch(batch_texts, GOOGLE_EMBEDDING_MODEL, current_batch_num, num_batches)
                             all_embeddings.extend(batch_embeddings)
                        except RuntimeError as e:
                             logger.error(f"Async embedding call issue ({e}), trying sync fallback for batch {current_batch_num}.")
                             try:
                                 sync_embeddings = genai.embed_content(
                                     model=GOOGLE_EMBEDDING_MODEL,
                                     content=batch_texts,
                                     task_type="RETRIEVAL_DOCUMENT"
                                 )['embedding']
                                 all_embeddings.extend(sync_embeddings)
                             except Exception as sync_e:
                                 logger.error(f"Sync embedding fallback failed for batch {current_batch_num}: {sync_e}")
                                 all_embeddings.extend([None] * len(batch_texts)) # Mark as failed

                    # Filter out failed embeddings and corresponding chunks
                    valid_indices = [i for i, emb in enumerate(all_embeddings) if emb is not None and len(emb) > 0] # Check embed not None or empty
                    num_failed = len(all_chunks) - len(valid_indices)
                    if num_failed > 0:
                        rag_status.warning(f"Failed to generate embeddings for {num_failed} chunks.")
                        logger.warning(f"Failed embeddings for {num_failed} chunks.")
                    all_chunks = [all_chunks[i] for i in valid_indices]
                    all_embeddings = [all_embeddings[i] for i in valid_indices]

                    if not all_chunks:
                         rag_status.error("Embedding generation failed for all chunks. Cannot proceed with RAG.")
                         progress_bar.progress(85, text="RAG failed (embedding failed).")

                    else:
                        rag_status.info(f"Successfully generated {len(all_embeddings)} embeddings. Matching relevant content...")
                        progress_bar.progress(75, text="Matching relevant content...")

                        # --- RAG Matching (FAISS or Supabase) ---
                        matched_chunks = []
                        # Get query embedding
                        try:
                            query_embedding_response = await genai.embed_content_async(
                                 model=GOOGLE_EMBEDDING_MODEL,
                                 content=current_query,
                                 task_type="RETRIEVAL_QUERY" # Use query task type
                            )
                            query_embedding = query_embedding_response['embedding']

                            k = min(15, len(all_chunks)) # Number of chunks to retrieve

                            if use_faiss:
                                faiss_index = build_faiss_index(all_embeddings)
                                if faiss_index:
                                    rag_status.info("Using FAISS for matching...")
                                    distances, indices = search_faiss(faiss_index, query_embedding, k)
                                    if indices is not None and len(indices) > 0:
                                        # Filter out potential -1 indices if search fails unexpectedly
                                        matched_chunks = [all_chunks[i] for i in indices if 0 <= i < len(all_chunks)]
                                        logger.info(f"Retrieved {len(matched_chunks)} chunks via FAISS.")
                                    else:
                                        rag_status.warning("FAISS search returned no valid indices, falling back to Supabase (if configured).")
                                        logger.warning("FAISS search failed or returned no results.")
                                        use_faiss = False # Force fallback
                                else:
                                    rag_status.warning("Failed to build FAISS index, falling back to Supabase (if configured).")
                                    use_faiss = False # Force fallback

                            if not use_faiss: # Fallback or default
                                rag_status.info("Using Supabase for matching...")
                                try:
                                    # Ensure Supabase client is available
                                    if supabase_client_instance:
                                         # Ensure the RPC function and parameters match your Supabase setup
                                         match_params = {
                                             "query_embedding": query_embedding, # Ensure this is a list
                                             "match_threshold": 0.75, # Adjust as needed
                                             "match_count": k
                                         }
                                         logger.debug(f"Calling Supabase RPC 'match_chunks' with k={k}, threshold={match_params['match_threshold']}")
                                         match_response = supabase_client_instance.rpc("match_chunks", match_params).execute()

                                         if match_response.data:
                                             # Adjust key based on your RPC function's return value ('chunk' or 'content')
                                             matched_chunks = [row.get("chunk", row.get("content")) for row in match_response.data if row.get("chunk", row.get("content"))]
                                             logger.info(f"Retrieved {len(matched_chunks)} chunks via Supabase RPC.")
                                             if not matched_chunks:
                                                  rag_status.warning("Supabase RPC returned data, but no valid 'chunk' or 'content' found.")
                                         else:
                                             rag_status.warning("Supabase RPC returned no matching chunks.")
                                             logger.warning(f"Supabase RPC 'match_chunks' returned no data. Response status: {match_response.status_code}, Error: {getattr(match_response, 'error', None)}")

                                    else:
                                         rag_status.error("Supabase client not available for matching.")
                                         logger.error("Supabase client instance is None during RAG matching.")

                                except Exception as rpc_error:
                                     rag_status.error(f"Supabase RPC 'match_chunks' failed: {rpc_error}")
                                     logger.error(f"Supabase RPC failed: {rpc_error}\n{traceback.format_exc()}")


                        except Exception as q_embed_error:
                             rag_status.error(f"Failed to generate query embedding: {q_embed_error}")
                             logger.error(f"Query embedding failed: {q_embed_error}\n{traceback.format_exc()}")


                        # --- Synthesis with Gemini ---
                        progress_bar.progress(80, text="Synthesizing report...")
                        if not matched_chunks:
                            rag_status.warning("No relevant content chunks found after matching. Cannot generate comprehensive report.")
                            comprehensive_report = "Could not generate report: No relevant information found in provided sources."
                        else:
                            rag_status.info(f"Synthesizing report from {len(matched_chunks)} relevant chunks...")
                            aggregated_relevant = "\n\n".join(matched_chunks)
                            # Use the citations_map to create the list for the prompt/output
                            citations = [f"{title}: {url_or_file}" for url_or_file, title in citations_map.items()]

                            # Construct the prompt for the final report
                            synthesis_prompt = f"""You are an expert diagnostic report generator. Based ONLY on the following aggregated content and the patient's query, generate a comprehensive diagnostic report.

Patient Query: "{current_query}"

Key Information Identified by Agent (for context):
- Main Subject: {analysis.get('main_subject', 'N/A')}
- Research Needs: {', '.join(analysis.get('What_needs_to_be_researched', []))}

Aggregated Relevant Content Extracted from Sources:
--- START CONTENT ---
{aggregated_relevant}
--- END CONTENT ---

Instructions:
- Directly address the patient's query using ONLY the aggregated content provided above.
- Provide detailed medical analysis based *strictly* on the text provided. Do NOT add external knowledge or assumptions.
- Include actionable recommendations *only if explicitly supported by the provided text*.
- Structure the report clearly (e.g., Findings, Analysis, Recommendations).
- **Crucially, cite the sources for your claims.** Use the source information provided within the '===' markers (e.g., "[Source: Title (URL)]" or "[Source: Uploaded File: filename]" or "[Source: Wearable Data Summary]"). If content source isn't clear from the markers for a specific piece of information, state that the source is unclear from the provided text.
- Respond with a detailed Markdown-formatted report. If no relevant information is found in the content to answer the query, state that clearly.
"""

                            try:
                                # Use the gemini_client wrapper or SDK directly
                                messages = [{"role": "user", "content": synthesis_prompt}]
                                if hasattr(gemini_client_instance, 'chat'):
                                     # loop = asyncio.get_event_loop()
                                     # response_data = await loop.run_in_executor(None, gemini_client_instance.chat, messages)
                                     response_data = gemini_client_instance.chat(messages) # Assuming okay
                                     comprehensive_report = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse synthesis response.")
                                elif hasattr(gemini_client_instance, 'generate_content'): # If using SDK model directly
                                     sdk_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                                     response = await sdk_model.generate_content_async(synthesis_prompt) # Adjust if sync
                                     comprehensive_report = response.text
                                else:
                                     comprehensive_report = "Error: Gemini client misconfigured for synthesis."

                                logger.info("Comprehensive report generated via RAG.")
                                rag_status.success("RAG analysis and report synthesis complete.")

                            except Exception as synth_error:
                                rag_status.error(f"Report synthesis with Gemini failed: {synth_error}")
                                logger.error(f"Gemini synthesis failed: {synth_error}\n{traceback.format_exc()}")
                                comprehensive_report = f"Report synthesis failed: {synth_error}"


            except Exception as rag_error:
                rag_status.error(f"An error occurred during the RAG pipeline: {rag_error}")
                logger.error(f"RAG pipeline error: {rag_error}\n{traceback.format_exc()}")
                comprehensive_report = f"RAG pipeline failed: {rag_error}"
    progress_bar.progress(85, text="RAG analysis finished.")


    # --- 4. Patient-Friendly Summary Generation ---
    st.markdown("### 4. Generating Patient-Friendly Summary...")
    summary_status = st.empty()
    patient_summary_report = "Summary generation did not run or failed."
    if comprehensive_report and "failed" not in comprehensive_report.lower() and "Could not generate report" not in comprehensive_report :
        progress_bar.progress(90, text="Generating patient summary...")
        summary_status.info("Generating patient-friendly summary...")
        with st.spinner("Generating patient-friendly summary..."):
            prompt = f"""You are a medical assistant skilled at explaining complex diagnostic reports in simple, clear language for patients. Based ONLY on the following comprehensive diagnostic report, produce a short summary (2-3 paragraphs) that a non-medical professional can easily understand.

Comprehensive Diagnostic Report:
--- START REPORT ---
{comprehensive_report}
--- END REPORT ---

Instructions:
- Summarize the main findings clearly and concisely based *only* on the report.
- List any specific, actionable recommendations *explicitly mentioned* in the report (like tests, referrals, lifestyle changes). Do not infer or add recommendations.
- Tell the patient the suggested next steps based *only* on the report content.
- Use simple, empathetic language. Avoid jargon.
- Do NOT add information or interpretations not present in the report above.
- If the report indicates uncertainty or lack of information, reflect that in the summary.
"""
            try:
                messages = [{"role": "user", "content": prompt}]
                if hasattr(gemini_client_instance, 'chat'):
                     # loop = asyncio.get_event_loop()
                     # response_data = await loop.run_in_executor(None, gemini_client_instance.chat, messages)
                     response_data = gemini_client_instance.chat(messages) # Assuming okay
                     patient_summary_report = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse patient summary response.")
                elif hasattr(gemini_client_instance, 'generate_content'):
                     sdk_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                     response = await sdk_model.generate_content_async(prompt) # Adjust if sync
                     patient_summary_report = response.text
                else:
                     patient_summary_report = "Error: Gemini client misconfigured for patient summary."

                logger.info("Patient-friendly summary generated.")
                summary_status.success("Patient-friendly summary generated.")

            except Exception as patient_summary_error:
                summary_status.error(f"Patient-friendly summary generation failed: {patient_summary_error}")
                logger.error(f"Patient summary generation failed: {patient_summary_error}\n{traceback.format_exc()}")
                patient_summary_report = f"Patient summary generation failed: {patient_summary_error}"
    else:
        summary_status.warning("Skipping patient-friendly summary because the comprehensive report was not generated successfully.")
        logger.warning("Skipping patient summary due to failed comprehensive report.")

    analysis_results['comprehensive_report'] = comprehensive_report
    analysis_results['patient_summary_report'] = patient_summary_report
    # Pass back the final citations list derived from the map
    analysis_results['citations'] = [f"{title}: {url_or_file}" for url_or_file, title in citations_map.items()]


    progress_bar.progress(100, text="Analysis Pipeline Completed!")
    st.balloons()
    st.success("Analysis Pipeline Completed!")
    # Clear progress bar after completion
    # progress_bar.empty() # Keep it at 100% for visibility

    return analysis_results

# --- Streamlit UI Layout ---

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")

    # Input Query
    query = st.text_area("1. Enter Patient Symptoms/Issue:", height=150, key="query_input", help="Describe the primary health concern or symptoms.")

    # URLs
    st.subheader("Web Search Options")
    include_urls_str = st.text_area("Include Specific URLs (one per line, optional):", height=50, key="include_urls", help="Force the agent to use only these URLs for information.")
    omit_urls_str = st.text_area("Omit URLs Containing (one per line, optional):", height=50, key="omit_urls", help="Exclude search results from URLs containing these strings (e.g., 'youtube.com', 'forum').")
    search_depth = st.selectbox("Search Depth:", ["basic", "advanced"], index=1, key="search_depth", help="'basic' is faster, 'advanced' is more thorough.")
    search_breadth = st.number_input("Search Breadth (results per query):", min_value=3, max_value=20, value=7, key="search_breadth", help="Number of search results to retrieve for each identified topic.") # Reduced default

    # File Uploads
    st.subheader("Reference Files (Optional)")
    uploaded_files = st.file_uploader(
        "Upload Local Files (PDF, DOCX, CSV, XLSX, TXT):",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'csv', 'xlsx', 'xls', 'txt'],
        key="file_uploader",
        help="Upload relevant medical records, articles, or notes."
    )
    uploaded_wearable_file = st.file_uploader(
        "Upload Wearable Data CSV (Optional):",
        accept_multiple_files=False,
        type=['csv'],
        key="wearable_uploader",
        help="Upload CSV with columns like 'heart_rate', 'steps', etc."
    )
    uploaded_survey_file = st.file_uploader(
        "Upload Survey Data CSV (for Diagnosis Prediction):",
        accept_multiple_files=False,
        type=['csv'],
        key="survey_uploader",
        help="Requires specific columns: 'What are the current symptoms...' and 'Medical Health History'."
    )

    # Advanced Options
    st.subheader("Advanced Options")
    use_faiss = st.checkbox("Use FAISS for RAG Matching (if installed)", value=True, key="use_faiss", help="Use local FAISS index for faster similarity search during RAG. Falls back to Supabase if unchecked or FAISS fails.")

    # Submit Button
    st.divider()
    submit_button = st.button("Run Diagnostic Analysis", type="primary", key="submit_button", use_container_width=True)

# --- Main Area for Outputs ---
st.header("Analysis Results")

# Initialize session state variables if they don't exist
default_state = {
    'analysis_complete': False,
    'results': {},
    'sentiment': "",
    'entities': "",
    'prediction': {},
    'explanation': "",
    'classifier_trained': False,
    'vectorizer': None,
    'model': None,
    'background_texts_sample': None # To store sample texts for SHAP
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Train Classifier ---
# Train the classifier if survey data is uploaded, happens outside the main button click
# to allow caching and prevent retraining on every run.
survey_df = None
if uploaded_survey_file:
    # Check if file has changed since last run to potentially retrain
    # Simple check based on file name and size (could be more robust)
    file_details = (uploaded_survey_file.name, uploaded_survey_file.size)
    if 'last_survey_file' not in st.session_state or st.session_state.last_survey_file != file_details:
        st.session_state.classifier_trained = False # Force retrain check
        st.session_state.vectorizer = None
        st.session_state.model = None
        st.session_state.background_texts_sample = None
        st.session_state.last_survey_file = file_details
        logger.info(f"New survey file detected: {uploaded_survey_file.name}")

        try:
            survey_df = pd.read_csv(uploaded_survey_file)
            st.sidebar.success(f"Loaded survey data: {uploaded_survey_file.name}")
            # Train or load cached model
            with st.spinner("Training symptom classifier..."):
                 vectorizer, model = train_symptom_classifier(survey_df)
            if vectorizer and model:
                 st.session_state.classifier_trained = True
                 st.session_state.vectorizer = vectorizer
                 st.session_state.model = model
                 # Store a sample of the texts used for training for SHAP background
                 symptom_col = 'What are the current symptoms or health issues you are facing'
                 if symptom_col in survey_df.columns:
                      background_texts = survey_df[symptom_col].astype(str).fillna("").tolist()
                      # Sample ~50 texts, or fewer if dataset is small
                      sample_size = min(50, len(background_texts))
                      st.session_state.background_texts_sample = np.random.choice(background_texts, sample_size, replace=False).tolist()
                      logger.info(f"Classifier trained. Stored {sample_size} background texts for SHAP.")
                 else:
                      logger.warning("Symptom column not found for SHAP background text sampling.")

                 st.sidebar.success("Symptom classifier trained.")

            else:
                 st.session_state.classifier_trained = False
                 st.sidebar.warning("Classifier training failed or yielded no model.")

        except Exception as e:
            st.sidebar.error(f"Error loading/training survey data: {e}")
            logger.error(f"Survey data loading/training error: {e}")
            st.session_state.classifier_trained = False
            st.session_state.last_survey_file = None # Reset file tracking on error
    # If file is the same, rely on cached state
    elif st.session_state.classifier_trained:
         st.sidebar.info("Using previously trained classifier.")


elif 'last_survey_file' in st.session_state: # File removed
    # Clear classifier state if file is removed
    st.session_state.classifier_trained = False
    st.session_state.vectorizer = None
    st.session_state.model = None
    st.session_state.background_texts_sample = None
    del st.session_state.last_survey_file # Remove tracking
    logger.info("Survey file removed, classifier unloaded.")


# --- Handle Analysis on Button Click ---
if submit_button and query:
    st.session_state.analysis_complete = False # Reset flag
    st.session_state.results = {} # Clear previous results
    st.session_state.sentiment = ""
    st.session_state.entities = ""
    st.session_state.prediction = {}
    st.session_state.explanation = ""

    # Prepare inputs
    include_urls_list = [url.strip() for url in include_urls_str.split('\n') if url.strip()]
    omit_urls_list = [url.strip() for url in omit_urls_str.split('\n') if url.strip()]

    # --- Process uploaded files ---
    additional_files_content = {}
    if uploaded_files:
        st.markdown("### Processing Uploaded Files")
        file_status = st.expander("File Processing Status", expanded=True)
        with file_status:
            st.write(f"Processing {len(uploaded_files)} uploaded reference files...")
            for file in uploaded_files:
                with st.spinner(f"Extracting text from {file.name}..."):
                    content = extract_text_from_file(file)
                    if content and isinstance(content, str) and content.strip() and "[Unsupported file type:" not in content and "[Could not decode TXT file]" not in content:
                        additional_files_content[file.name] = content
                        st.write(f"✅ Successfully processed: {file.name}")
                        logger.info(f"Processed uploaded file: {file.name}")
                    elif content: # Handled error message from extract_text_from_file
                         st.warning(f"⚠️ Could not fully process: {file.name} ({content})")
                    else: # Generic failure
                         st.warning(f"⚠️ Failed to extract text from: {file.name}")

    # --- Process wearable data ---
    wearable_summary = None
    if uploaded_wearable_file:
        st.markdown("### Processing Wearable Data")
        with st.spinner("Processing wearable data..."):
             wearable_summary = read_wearable_data(uploaded_wearable_file)
             if wearable_summary and "Error processing" not in wearable_summary:
                 st.success("Wearable data processed.")
                 with st.expander("Wearable Data Summary", expanded=False):
                      st.text(wearable_summary)
             elif wearable_summary: # Contains error message
                  st.warning(f"Could not process wearable data: {wearable_summary}")
             else: # Should not happen if function returns string, but safeguard
                  st.warning("Failed to process wearable data.")


    # --- Run Initial Gemini Analysis (Sentiment, Entities) ---
    st.markdown("### Preliminary Analysis")
    prelim_status = st.empty()
    with st.spinner("Analyzing sentiment and extracting entities..."):
         prelim_status.info("Analyzing sentiment and extracting entities...")
         # Run async functions using asyncio.run or Streamlit's event loop management
         try:
             # Using asyncio.gather to run concurrently
             sentiment_result, entities_result = await asyncio.gather(
                 analyze_sentiment_gemini(query, gemini_client),
                 extract_medical_entities_gemini(query, gemini_client)
             )
             st.session_state.sentiment = sentiment_result
             st.session_state.entities = entities_result
             prelim_status.success("Sentiment and entity analysis complete.")

         except Exception as initial_analysis_err:
              st.error(f"Error during initial sentiment/entity analysis: {initial_analysis_err}")
              logger.error(f"Initial Gemini analysis failed: {initial_analysis_err}")
              st.session_state.sentiment = f"Error: {initial_analysis_err}"
              st.session_state.entities = f"Error: {initial_analysis_err}"
              prelim_status.error("Sentiment and entity analysis failed.")

    # Display Sentiment & Entities
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Sentiment Analysis (Gemini)", expanded=False):
            st.write(st.session_state.sentiment)
    with col2:
        with st.expander("Extracted Medical Entities (Gemini)", expanded=False):
            st.write(st.session_state.entities)

    # --- Run Diagnosis Prediction & Explanation (if model trained) ---
    if st.session_state.classifier_trained:
        st.markdown("### Diagnosis Prediction (Experimental)")
        diag_status = st.empty()
        with st.spinner("Predicting diagnosis and generating explanation..."):
            diag_status.info("Predicting diagnosis and generating explanation...")
            pred_diag, pred_proba = predict_diagnosis(query, st.session_state.vectorizer, st.session_state.model)
            st.session_state.prediction = {"diagnosis": pred_diag, "probabilities": pred_proba}

            # Generate SHAP explanation, passing background text sample
            st.session_state.explanation = explain_diagnosis_shap(
                query,
                st.session_state.vectorizer,
                st.session_state.model,
                st.session_state.background_texts_sample # Pass the sample
            )
            diag_status.success("Diagnosis prediction and explanation complete.")

        st.write(f"**Predicted Diagnosis:** {st.session_state.prediction.get('diagnosis', 'N/A')}")
        # Display probabilities and explanation in columns
        col_prob, col_shap = st.columns(2)
        with col_prob:
             with st.expander("Prediction Probabilities", expanded=False):
                  st.json(st.session_state.prediction.get('probabilities', {}))
        with col_shap:
             with st.expander("Diagnosis Explanation (SHAP)", expanded=False):
                  st.text(st.session_state.explanation) # Use st.text for preformatted text

        # Save query history including prediction
        save_query_history(
            query,
            st.session_state.prediction.get('diagnosis', 'N/A'),
            st.session_state.sentiment,
            st.session_state.entities
        )
    else:
        st.info("Diagnosis prediction skipped (Survey data for training not provided or classifier training failed).")
        # Save query history without prediction
        save_query_history(query, "N/A - Model not trained", st.session_state.sentiment, st.session_state.entities)


    # --- Run the Main Analysis Pipeline ---
    st.divider()
    st.header("Main Analysis Pipeline")
    main_pipeline_status = st.empty()
    try:
        # Instantiate SubjectAnalyzer here if needed, passing the gemini client
        # Ensure AnalysisConfig is defined or imported
        try:
             analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
             subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config)
        except NameError as ne:
             st.error(f"Failed to create SubjectAnalyzer: {ne}. Check imports.")
             logger.error(f"NameError during SubjectAnalyzer creation: {ne}")
             subject_analyzer = None # Ensure it's None if creation fails
        except Exception as config_err:
             st.error(f"Error creating SubjectAnalyzer config: {config_err}")
             logger.error(f"Error creating SubjectAnalyzer config: {config_err}")
             subject_analyzer = None


        # Run the main async pipeline
        # Use asyncio.run() to execute the async function from the sync Streamlit context
        # Need to ensure an event loop is available. Streamlit usually manages this.
        main_pipeline_status.info("Running main analysis pipeline...")
        analysis_results = await run_analysis_pipeline(
            original_query=query,
            include_urls=include_urls_list,
            omit_urls=omit_urls_list,
            additional_files_content=additional_files_content,
            wearable_data_summary=wearable_summary,
            search_depth=search_depth,
            search_breadth=search_breadth,
            use_faiss=use_faiss,
            gemini_client_instance=gemini_client, # Pass the client
            supabase_client_instance=supabase,   # Pass the client
            search_service_instance=search_service, # Pass the service
            extractor_instance=extractor,         # Pass the extractor
            subject_analyzer_instance=subject_analyzer # Pass the analyzer
        )
        st.session_state.results = analysis_results
        st.session_state.analysis_complete = True
        main_pipeline_status.empty() # Clear status message on success

    except Exception as pipeline_error:
        st.error(f"An critical error occurred during the main analysis pipeline: {pipeline_error}")
        logger.error(f"Main pipeline execution failed: {pipeline_error}\n{traceback.format_exc()}")
        st.session_state.analysis_complete = False
        main_pipeline_status.error(f"Main analysis pipeline failed: {pipeline_error}")


# --- Display Results After Analysis ---
if st.session_state.analysis_complete:
    st.divider()
    st.header("Generated Reports")

    results = st.session_state.results
    comp_report = results.get('comprehensive_report', 'Not generated.')
    patient_report = results.get('patient_summary_report', 'Not generated.')
    citations = results.get('citations', [])

    # Display Reports using Tabs
    tab1, tab2, tab3 = st.tabs(["Comprehensive Report", "Patient Summary", "Citations & Sources"])

    with tab1:
        st.subheader("Comprehensive Diagnostic Report")
        st.markdown(comp_report) # Display markdown
        st.download_button(
            label="Download Comprehensive Report (.md)",
            data=comp_report,
            file_name=safe_filename(query, "comprehensive_report.md"),
            mime="text/markdown",
            key="download_comp"
        )

    with tab2:
        st.subheader("Patient-Friendly Summary")
        st.markdown(patient_report)
        st.download_button(
            label="Download Patient Summary (.md)",
            data=patient_report,
            file_name=safe_filename(query, "patient_summary.md"),
            mime="text/markdown",
            key="download_patient"
        )

    with tab3:
        st.subheader("Citations / Sources Used")
        if citations:
            st.write("Sources referenced in the comprehensive report:")
            for cit in citations:
                # Try to make URLs clickable if they are present
                parts = cit.split(":", 1)
                if len(parts) == 2:
                     title, url_or_file = parts[0].strip(), parts[1].strip()
                     if url_or_file.startswith("http://") or url_or_file.startswith("https://"):
                          st.markdown(f"- **{title}:** [{url_or_file}]({url_or_file})")
                     else:
                          st.markdown(f"- **{title}:** {url_or_file}")
                else:
                     st.markdown(f"- {cit}") # Fallback for unexpected format
        else:
            st.info("No specific citations were generated or extracted for this report.")


    # Display Visualizations (Optional)
    st.divider()
    st.header("Query Trend Visualizations")
    st.markdown("Visualize trends based on historical queries stored in `query_history.csv`.")
    if st.button("Generate Trend Visualizations", key="viz_button"):
        viz_placeholder = st.empty()
        with viz_placeholder.spinner("Generating visualizations..."):
            figs = visualize_query_trends()
            if figs:
                viz_placeholder.success("Visualizations generated.")
                for name, fig in figs.items():
                    st.subheader(name.replace('_', ' ').title())
                    st.pyplot(fig)
                    # Add download button for the plot
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format="png", bbox_inches='tight')
                    st.download_button(
                        label=f"Download {name.replace('_', ' ')} plot (.png)",
                        data=img_buffer,
                        file_name=f"{name}.png",
                        mime="image/png",
                        key=f"download_{name}"
                    )
                    plt.close(fig) # Close the figure to free memory
            else:
                viz_placeholder.info("No visualizations could be generated (check `query_history.csv`).")


elif submit_button and not query:
    st.warning("⚠️ Please enter patient symptoms or issue in the sidebar.")

# Add a footer or additional information
st.divider()
st.caption("Medical Diagnostic Agent v1.0 | For informational purposes only. Always consult a qualified healthcare professional for medical advice.")

