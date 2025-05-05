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
    from web_agent.src.services.web_search import WebSearchService
    from web_agent.src.models.search_models import SearchConfig
    from subject_analyzer.src.services.tavily_client import TavilyClient
    from subject_analyzer.src.services.tavily_extractor import TavilyExtractor
    from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer
    from subject_analyzer.src.services.gemini_client import GeminiClient # IMPORTANT: Use the Gemini client
    from subject_analyzer.src.models.analysis_models import AnalysisConfig
except ImportError:
    st.error("Required custom agent modules (web_agent, subject_analyzer) not found. "
             "Please ensure they are installed or in the correct path.")
    # Add placeholder classes or exit if these are critical
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
    st.error(f"Missing required secret: {e}. Please configure your `secrets.toml` file.")
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

        # Directly use the genai library functions for chat and embeddings
        # The custom GeminiClient class might need adaptation or replacement
        # For simplicity, let's use the SDK directly here.
        # If GeminiClient has complex logic, it needs to be integrated.
        # Assuming GeminiClient wraps genai.GenerativeModel
        analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3) # Assuming AnalysisConfig exists
        # This might need adjustment based on how GeminiClient is implemented
        client_wrapper = GeminiClient(api_key=GEMINI_API_KEY, config=analysis_config)
        logger.info("Gemini client initialized successfully.")
        # Return the wrapper or the model directly depending on usage
        # return genai.GenerativeModel(GEMINI_MODEL_NAME)
        return client_wrapper # Return the custom client if it's used extensively
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
        search_config = SearchConfig() # Assuming default config
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
        # Assuming gemini_client has a method like 'generate_content' or similar
        # Adjust based on the actual client implementation (custom wrapper or genai SDK)
        if hasattr(client, 'generate_content'): # Using the SDK directly
             response = await client.generate_content(prompt) # Use await if client method is async
             return response.text
        elif hasattr(client, 'chat'): # Using the custom GeminiClient wrapper
             # The wrapper's chat method might expect a list of messages
             messages = [{"role": "user", "content": prompt}]
             response_data = client.chat(messages) # Assuming this is synchronous or handled by the wrapper
             # Extract text from the response structure provided by the wrapper
             # This might need adjustment based on GeminiClient's response format
             sentiment_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse sentiment response.")
             return sentiment_text
        else:
             logger.error("Gemini client does not have a recognized method for content generation.")
             return "Error: Gemini client misconfigured."

    except Exception as e:
        logger.error(f"Gemini sentiment analysis failed: {e}\n{traceback.format_exc()}")
        return f"Error during sentiment analysis: {e}"

async def extract_medical_entities_gemini(query, client):
    """Extracts medical entities using Gemini."""
    prompt = f"""Extract the key medical entities (like symptoms, conditions, medications mentioned) from the following patient query. List them clearly.

Patient Query: "{query}"

Medical Entities:
- [Entity 1]
- [Entity 2]
..."""
    try:
        # Similar logic as analyze_sentiment_gemini for calling the client
        if hasattr(client, 'generate_content'):
             response = await client.generate_content(prompt)
             return response.text
        elif hasattr(client, 'chat'):
             messages = [{"role": "user", "content": prompt}]
             response_data = client.chat(messages)
             entities_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse entities response.")
             return entities_text
        else:
             logger.error("Gemini client does not have a recognized method for content generation.")
             return "Error: Gemini client misconfigured."

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
                text += page.extract_text() or ""
            return text
        elif ext == '.docx':
            doc = docx.Document(io.BytesIO(bytes_data))
            return "\n".join(para.text for para in doc.paragraphs)
        elif ext == '.csv':
            # Read CSV data using pandas, return as string or DataFrame
            df = pd.read_csv(io.BytesIO(bytes_data))
            return df.to_csv(index=False) # Return CSV content as string
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(io.BytesIO(bytes_data))
            return df.to_csv(index=False) # Return as CSV string for consistency
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
        symptoms = df[symptom_col].fillna("").astype(str).tolist()
        labels = df[history_col].fillna("None").astype(str).tolist()

        # Basic processing: use the first condition if multiple are listed
        processed_labels = [label.split(',')[0].strip() if isinstance(label, str) else 'None' for label in labels]

        # Filter out entries with no symptoms or 'None' labels if desired
        valid_data = [(s, l) for s, l in zip(symptoms, processed_labels) if s and l != 'None']
        if not valid_data:
            logger.warning("No valid symptom/label pairs found in survey data for training.")
            return None, None

        texts, final_labels = zip(*valid_data)

        if not texts or not final_labels:
            logger.warning("Not enough valid data to train classifier after filtering.")
            return None, None

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000) # Limit features
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=1000, class_weight='balanced') # Add class_weight
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

def explain_diagnosis_shap(query, vectorizer, model):
    """Generates SHAP explanation for the prediction."""
    if not vectorizer or not model:
        return "Model not trained, cannot explain."
    try:
        # Check if the query contains any known features
        X_query_check = vectorizer.transform([query])
        if X_query_check.nnz == 0: # Check if the transformed vector is all zeros
             logger.warning("Query contains no features known to the model. Cannot generate SHAP explanation.")
             return "Explanation not available (query contains no known terms)."


        # SHAP requires dense data for LinearExplainer
        X_query = vectorizer.transform([query]).toarray()

        # Create a background dataset for the explainer
        # Using feature means or a sample of training data is common.
        # Here, we'll use the transformed feature names as a simple background.
        # Note: This might not be the most robust background dataset.
        # Consider using a sample of the actual training data if performance allows.
        # We need the original texts used for training to transform for background
        # This part is tricky without access to the original training texts here.
        # Let's try a simplified approach using feature vectors directly.
        # This might require adjusting the SHAP explainer type or background data.

        # Alternative: Use KernelExplainer (slower but more flexible)
        # Need a prediction function that takes dense array
        predict_proba_dense = lambda x: model.predict_proba(x)
        # Use shap.sample to create a background dataset (e.g., using feature means)
        # background_data = shap.sample(vectorizer.transform(vectorizer.get_feature_names_out()).toarray(), 50) # Sample 50 features
        # For simplicity, let's stick to LinearExplainer if possible, but it might fail
        # If LinearExplainer fails, KernelExplainer is the fallback.

        try:
            # Try LinearExplainer first (faster if it works)
            # It needs a background dataset representative of the training data distribution.
            # Using the means of the TF-IDF vectors might be a proxy.
            # This requires the original training matrix 'X' from train_symptom_classifier
            # which is not directly available here due to caching.
            # --> This design makes SHAP explanation difficult without passing training data.

            # Let's skip the complex background data for now and see if it works simply.
            # This might produce less accurate explanations.
            explainer = shap.LinearExplainer(model, vectorizer.transform(vectorizer.get_feature_names_out()), feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_query) # shap_values for the query

        except Exception as linear_error:
            logger.warning(f"LinearExplainer failed ({linear_error}), trying KernelExplainer.")
            try:
                 # Fallback to KernelExplainer
                 # Need a representative background dataset. Using zeros or means is common but potentially biased.
                 # Let's use a small sample of feature vectors (e.g., means or random sample if possible)
                 # For now, using zeros as a placeholder background. This is NOT ideal.
                 background_data = shap.kmeans(vectorizer.transform(vectorizer.get_feature_names_out()).toarray(), 10) # K-means for background
                 explainer = shap.KernelExplainer(predict_proba_dense, background_data)
                 # Need to specify the output index for the predicted class
                 predicted_class_index = np.where(model.classes_ == model.predict(X_query)[0])[0][0]
                 shap_values_kernel = explainer.shap_values(X_query, l1_reg="num_features(5)") # Explain top 5 features
                 # KernelExplainer returns shap_values for each class, select the one for the predicted class
                 shap_values = shap_values_kernel[predicted_class_index]

            except Exception as kernel_error:
                logger.error(f"SHAP explanation failed with both Linear and Kernel explainers: {kernel_error}\n{traceback.format_exc()}")
                return f"SHAP explanation failed: {kernel_error}"


        # Process SHAP values to get top features
        feature_names = vectorizer.get_feature_names_out()
        # Ensure shap_values is treated as a 1D array for the single instance
        shap_values_instance = shap_values[0] if isinstance(shap_values, list) or shap_values.ndim > 1 else shap_values

        # Filter out zero shap values before sorting
        non_zero_indices = np.where(shap_values_instance != 0)[0]
        if len(non_zero_indices) == 0:
             return "Explanation: No features significantly contributed to the prediction based on SHAP analysis."


        # Get indices sorted by absolute SHAP value magnitude
        sorted_indices = np.argsort(np.abs(shap_values_instance[non_zero_indices]))[::-1]
        top_indices = non_zero_indices[sorted_indices[:5]] # Get top 5 non-zero features


        explanation = "Top contributing features (SHAP values):\n"
        if len(top_indices) > 0:
            for idx in top_indices:
                 # Ensure index is within bounds of feature_names
                 if idx < len(feature_names):
                     feature = feature_names[idx]
                     shap_val = shap_values_instance[idx]
                     explanation += f"- {feature}: {shap_val:.4f}\n"
                 else:
                     logger.warning(f"SHAP index {idx} out of bounds for feature names.")
        else:
            explanation += "No significant features identified by SHAP."

        logger.info("SHAP explanation generated successfully.")
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
            if not file_exists:
                writer.writerow(["timestamp", "query", "predicted_diagnosis", "sentiment_summary", "entities_summary"])
            # Store summaries or key parts of sentiment/entities if they are long strings
            sentiment_summary = sentiment.split('\n')[0] # Get first line
            entities_summary = entities.split('\n')[0]   # Get first line
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
        st.info("No query history available for visualization.")
        return figs
    try:
        df = pd.read_csv(filename)
        if df.empty:
            st.info("Query history is empty.")
            return figs

        # Convert timestamp safely
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']) # Drop rows where conversion failed
        df = df.sort_values("timestamp")

        # --- Plot 1: Diagnosis Frequency ---
        if 'predicted_diagnosis' in df.columns:
            diag_counts = df['predicted_diagnosis'].value_counts().nlargest(10) # Top 10
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

        # --- Plot 2: Sentiment Trend (Placeholder - requires parsing sentiment_summary) ---
        # This requires extracting a numerical score from the 'sentiment_summary' string
        # For now, we skip this plot as the Gemini sentiment is text-based.
        # If you adapt Gemini to return a score, you can implement this.
        # Example: df['sentiment_score'] = df['sentiment_summary'].apply(extract_score_function)
        st.info("Sentiment trend visualization requires numerical sentiment scores (not implemented for text-based sentiment).")


        # --- Plot 3: Entity Distribution ---
        if 'entities_summary' in df.columns:
             entity_list = []
             # Extract entities assuming they are listed after "Medical Entities:"
             for entities_str in df['entities_summary'].fillna(""):
                 if "Medical Entities:" in entities_str:
                     actual_entities = entities_str.split("Medical Entities:")[1]
                     entities = [e.strip().replace('-','').strip() for e in actual_entities.split('\n') if e.strip() and e.strip() != '-']
                     entity_list.extend(entities)

             if entity_list:
                 entity_counts = Counter(entity_list).most_common(15) # Top 15
                 if entity_counts:
                     labels, sizes = zip(*entity_counts)
                     fig_ent, ax_ent = plt.subplots(figsize=(8, 8))
                     ax_ent.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
                     ax_ent.set_title("Top 15 Extracted Medical Entities Distribution")
                     plt.tight_layout()
                     figs['entities_distribution'] = fig_ent
                     logger.info("Generated entities distribution plot.")


        return figs

    except FileNotFoundError:
        st.info("Query history file not found.")
        return figs
    except pd.errors.EmptyDataError:
         st.info("Query history file is empty.")
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
        dim = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_array)
        logger.info(f"Built FAISS index with {len(embeddings)} vectors of dimension {dim}.")
        return index
    except ImportError:
         st.warning("FAISS library not installed. Cannot use FAISS optimization. `pip install faiss-cpu`")
         logger.warning("FAISS library not installed.")
         return None
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        logger.error(f"FAISS index build failed: {e}\n{traceback.format_exc()}")
        return None

def search_faiss(index, query_embedding, k):
    """Searches the FAISS index."""
    if index is None:
        return None, None
    try:
        query_vec = np.array(query_embedding).reshape(1, -1).astype('float32')
        distances, indices = index.search(query_vec, k)
        logger.info(f"FAISS search completed, found {len(indices[0])} indices.")
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
        df = pd.read_csv(uploaded_file)
        summary = "Wearable Data Summary:\n"
        if "heart_rate" in df.columns:
            avg_hr = df["heart_rate"].mean()
            min_hr = df["heart_rate"].min()
            max_hr = df["heart_rate"].max()
            summary += f"- Heart Rate: Avg={avg_hr:.1f}, Min={min_hr:.1f}, Max={max_hr:.1f}\n"
        if "steps" in df.columns:
            total_steps = df["steps"].sum()
            avg_steps = df["steps"].mean()
            summary += f"- Steps: Total={total_steps}, Avg Daily (if applicable)={avg_steps:.0f}\n"
        # Add more metrics if needed (e.g., sleep, spo2)
        if summary == "Wearable Data Summary:\n":
             summary += "- No recognized columns (e.g., 'heart_rate', 'steps') found."
        logger.info("Wearable data processed.")
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
    """
    st.info("Starting analysis pipeline...")
    analysis_results = {} # Store results here
    current_date = datetime.today().strftime("%Y-%m-%d")
    current_query = original_query # Start with the original query

    # --- 1. Subject Analysis ---
    st.markdown("### 1. Analyzing Query Subject...")
    analysis = {}
    try:
        with st.spinner("Analyzing patient query context..."):
            # Assuming SubjectAnalyzer uses the passed gemini_client internally
            analysis = subject_analyzer_instance.analyze(f"{current_query} (as of {current_date})")
            analysis_results['subject_analysis'] = analysis
            logger.info("Subject analysis successful.")
            st.success("Query analysis complete.")
            # Display Agent's Understanding
            st.subheader("Agent's Understanding:")
            st.write(f"**Patient Query:** {original_query}")
            st.write(f"**Identified Medical Issue:** {analysis.get('main_subject', 'Unknown Issue')}")
            temporal = analysis.get("temporal_context", {})
            if temporal:
                st.write("**Temporal Context:**")
                for key, value in temporal.items():
                    st.write(f"- {key.capitalize()}: {value}")
            needs = analysis.get("What_needs_to_be_researched", [])
            st.write("**Key aspects to investigate:**")
            st.write(f"{', '.join(needs) if needs else 'None'}")

    except Exception as e:
        st.error(f"Subject analysis failed: {e}")
        logger.error(f"Subject analysis failed: {e}\n{traceback.format_exc()}")
        analysis_results['subject_analysis_error'] = str(e)
        # Decide whether to stop or continue
        st.warning("Continuing without full subject analysis.")
        analysis = {} # Reset analysis to avoid errors later

    # --- 2. Web Search and Extraction ---
    st.markdown("### 2. Searching and Extracting Information...")
    search_results_dict = {}
    extracted_content_dict = {}
    search_performed = False

    if not include_urls:
        topics_to_search = [analysis.get("main_subject", current_query)] # Fallback to query
        topics_to_search += analysis.get("What_needs_to_be_researched", [])
        topics_to_search = list(set(filter(None, topics_to_search))) # Unique, non-empty topics

        if not topics_to_search:
            st.warning("No specific topics identified for web search based on analysis.")
        else:
            search_performed = True
            st.write(f"**Searching web for:** {', '.join(topics_to_search)}")
            with st.spinner(f"Searching web and extracting content for {len(topics_to_search)} topics..."):
                for topic in topics_to_search:
                    st.write(f"--- Searching for: {topic} ---")
                    try:
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
                        st.write(f"Found {len(filtered_results)} relevant results for '{topic}'.")
                        # Log found URLs (optional)
                        # for res in filtered_results: logger.debug(f"Found URL: {res.get('url')}")

                        # Extract content for this topic's URLs
                        urls_to_extract = [res.get("url") for res in filtered_results if res.get("url")]
                        if urls_to_extract:
                             extraction_response = extractor_instance.extract(
                                 urls=urls_to_extract,
                                 extract_depth="advanced", # Or make configurable
                                 include_images=False
                             )
                             extracted_content_dict[topic] = extraction_response.get("results", [])
                             # Log extraction success/failure counts
                             failed_count = sum(1 for item in extracted_content_dict[topic] if item.get("error"))
                             logger.info(f"Extracted content for {len(urls_to_extract) - failed_count}/{len(urls_to_extract)} URLs for topic '{topic}'.")
                             if failed_count > 0: st.warning(f"Failed to extract content from {failed_count} URLs for '{topic}'.")

                    except Exception as e:
                        st.error(f"Search/Extraction failed for topic '{topic}': {e}")
                        logger.error(f"Search/Extraction failed for '{topic}': {e}\n{traceback.format_exc()}")
                        search_results_dict[topic] = []
                        extracted_content_dict[topic] = []
    else:
        # Use user-provided URLs
        search_performed = True
        st.write(f"**Using {len(include_urls)} user-provided URLs.**")
        filtered_urls = [url for url in include_urls if not any(omit.lower() in url.lower() for omit in omit_urls)]
        st.write(f"({len(filtered_urls)} after filtering omitted URLs)")
        search_results_dict["User Provided"] = [{"title": "User Provided", "url": url, "score": "N/A"} for url in filtered_urls]

        if filtered_urls:
            with st.spinner("Extracting content from provided URLs..."):
                try:
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
        st.success("Search and extraction phase complete.")
    else:
        st.info("No web search performed (either used provided URLs or no topics found).")


    # --- 3. RAG Analysis (Embeddings, Supabase/FAISS, Gemini Synthesis) ---
    st.markdown("### 3. Performing RAG Analysis...")
    comprehensive_report = "RAG analysis did not run or failed."
    citations = []
    all_chunks = []
    all_embeddings = []

    # Aggregate content from web extraction and additional files
    full_content_for_rag = ""
    citations_map = {} # url -> title

    # Add extracted web content
    for topic, items in extracted_content_dict.items():
        for item in items:
            url = item.get("url", "No URL")
            title = item.get("title", "No Title")
            content = item.get("text") or item.get("raw_content", "")
            if content:
                full_content_for_rag += f"\n\n=== Content from {title} ({url}) ===\n{content}\n"
                if url != "No URL": citations_map[url] = title

    # Add content from uploaded files
    for filename, content in additional_files_content.items():
         if content:
             full_content_for_rag += f"\n\n=== Content from Uploaded File: {filename} ===\n{content}\n"
             citations_map[filename] = f"Uploaded File: {filename}" # Add file to citations

    # Add wearable data summary if available
    if wearable_data_summary:
         full_content_for_rag += f"\n\n=== Wearable Data Summary ===\n{wearable_data_summary}\n"
         # Optionally add a citation for wearable data if needed

    if not full_content_for_rag.strip():
        st.warning("No content available for RAG analysis (no web results or file content). Skipping.")
    else:
        with st.spinner("Generating embeddings and performing RAG..."):
            try:
                # Chunking
                def chunk_text(text, chunk_size=700, overlap=50): # Smaller chunk size, add overlap
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunks.append(text[start:end])
                        start += chunk_size - overlap
                        if start >= len(text): # Ensure we don't overshoot due to overlap logic error
                             break # Exit if start index goes beyond text length
                    # Ensure the last part is captured if overlap causes skipping
                    if start < len(text) and text[start:].strip():
                         last_chunk = text[start:]
                         if len(last_chunk) > chunk_size // 2: # Add if reasonably sized
                             chunks.append(last_chunk)
                         elif chunks: # Append to previous chunk if small
                             chunks[-1] += last_chunk
                         else: # Only one small chunk
                             chunks.append(last_chunk)

                    # Filter out very small or empty chunks
                    chunks = [c for c in chunks if len(c.strip()) > 20] # Min length filter
                    logger.info(f"Chunked content into {len(chunks)} chunks.")
                    if not chunks: logger.warning("Chunking resulted in zero valid chunks.")
                    return chunks

                all_chunks = chunk_text(full_content_for_rag)

                if not all_chunks:
                     st.warning("Content chunking resulted in no usable chunks. Skipping RAG.")
                else:
                    # Embedding Generation (using Gemini Embeddings)
                    st.write(f"Generating embeddings for {len(all_chunks)} text chunks...")

                    async def get_embedding_batch(texts, model_name):
                        """Helper to get embeddings for a batch of texts."""
                        try:
                            # Use genai.embed_content for batching
                            result = await genai.embed_content_async( # Use async version
                                model=model_name,
                                content=texts,
                                task_type="RETRIEVAL_DOCUMENT" # Specify task type
                            )
                            return result['embedding'] # Assuming result structure
                        except Exception as e:
                            logger.error(f"Batch embedding failed: {e}")
                            # Return None or empty list for failed batches
                            return [None] * len(texts) # Indicate failure for each text

                    # Process in batches to avoid hitting API limits
                    batch_size = 100 # Gemini batch limit can be up to 100
                    all_embeddings = []
                    for i in range(0, len(all_chunks), batch_size):
                        batch_texts = all_chunks[i:i+batch_size]
                        st.write(f"...embedding batch {i//batch_size + 1}...")
                        # Run async batch embedding within Streamlit's event loop if possible
                        # Or use asyncio.run() if necessary, but might block
                        try:
                             # This might need adjustment depending on Streamlit's async handling
                             batch_embeddings = await get_embedding_batch(batch_texts, GOOGLE_EMBEDDING_MODEL)
                             all_embeddings.extend(batch_embeddings)
                        except RuntimeError as e:
                             # Handle cases where event loop is not running or misused
                             logger.warning(f"Async embedding call issue ({e}), trying synchronous fallback.")
                             try:
                                 # Synchronous fallback (less efficient)
                                 sync_embeddings = genai.embed_content(
                                     model=GOOGLE_EMBEDDING_MODEL,
                                     content=batch_texts,
                                     task_type="RETRIEVAL_DOCUMENT"
                                 )['embedding']
                                 all_embeddings.extend(sync_embeddings)
                             except Exception as sync_e:
                                 logger.error(f"Synchronous embedding fallback also failed: {sync_e}")
                                 all_embeddings.extend([None] * len(batch_texts)) # Mark as failed

                        # Check for None values indicating failures
                        if None in batch_embeddings:
                             logger.warning(f"Some embeddings failed in batch starting at index {i}.")
                             # Handle failed embeddings (e.g., skip chunks, use zero vectors)
                             # For now, we'll filter them out later

                    # Filter out failed embeddings and corresponding chunks
                    valid_indices = [i for i, emb in enumerate(all_embeddings) if emb is not None]
                    if len(valid_indices) < len(all_chunks):
                        st.warning(f"Failed to generate embeddings for {len(all_chunks) - len(valid_indices)} chunks.")
                    all_chunks = [all_chunks[i] for i in valid_indices]
                    all_embeddings = [all_embeddings[i] for i in valid_indices]

                    if not all_chunks:
                         st.error("Embedding generation failed for all chunks. Cannot proceed with RAG.")

                    else:
                        st.write(f"Successfully generated {len(all_embeddings)} embeddings.")

                        # --- RAG Matching (FAISS or Supabase) ---
                        st.write("Matching relevant content using embeddings...")
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
                                    st.write("Using FAISS for matching...")
                                    distances, indices = search_faiss(faiss_index, query_embedding, k)
                                    if indices is not None:
                                        matched_chunks = [all_chunks[i] for i in indices if i < len(all_chunks)]
                                        logger.info(f"Retrieved {len(matched_chunks)} chunks via FAISS.")
                                    else:
                                        st.warning("FAISS search failed, falling back to Supabase (if configured).")
                                        use_faiss = False # Force fallback
                                else:
                                    st.warning("Failed to build FAISS index, falling back to Supabase (if configured).")
                                    use_faiss = False # Force fallback

                            if not use_faiss: # Fallback or default
                                st.write("Using Supabase for matching...")
                                try:
                                    # Ensure Supabase client is available
                                    if supabase_client_instance:
                                         match_response = supabase_client_instance.rpc(
                                             "match_chunks", # Ensure this RPC function exists in your Supabase setup
                                             {
                                                 "query_embedding": query_embedding,
                                                 "match_threshold": 0.75, # Example threshold
                                                 "match_count": k
                                             }
                                         ).execute()

                                         if match_response.data:
                                             matched_chunks = [row["chunk"] for row in match_response.data] # Adjust key if needed
                                             logger.info(f"Retrieved {len(matched_chunks)} chunks via Supabase RPC.")
                                         else:
                                             st.warning("Supabase RPC returned no matching chunks.")
                                             logger.warning(f"Supabase RPC 'match_chunks' returned no data. Response: {match_response}")


                                    else:
                                         st.error("Supabase client not available for matching.")

                                except Exception as rpc_error:
                                     st.error(f"Supabase RPC 'match_chunks' failed: {rpc_error}")
                                     logger.error(f"Supabase RPC failed: {rpc_error}\n{traceback.format_exc()}")


                        except Exception as q_embed_error:
                             st.error(f"Failed to generate query embedding: {q_embed_error}")
                             logger.error(f"Query embedding failed: {q_embed_error}\n{traceback.format_exc()}")


                        # --- Synthesis with Gemini ---
                        if not matched_chunks:
                            st.warning("No relevant content chunks found after matching. Cannot generate comprehensive report.")
                        else:
                            st.write(f"Synthesizing report from {len(matched_chunks)} relevant chunks...")
                            aggregated_relevant = "\n\n".join(matched_chunks)
                            citations = [f"{title}: {url}" for url, title in citations_map.items()] # Use the map

                            # Construct the prompt for the final report
                            synthesis_prompt = f"""You are an expert diagnostic report generator. Based ONLY on the following aggregated content and the patient's query, generate a comprehensive diagnostic report.

Patient Query: "{current_query}"

Key Information Identified by Agent:
- Main Subject: {analysis.get('main_subject', 'N/A')}
- Research Needs: {', '.join(analysis.get('What_needs_to_be_researched', []))}

Aggregated Relevant Content:
--- START CONTENT ---
{aggregated_relevant}
--- END CONTENT ---

Instructions:
- Directly address the patient's query using the provided content.
- Provide detailed medical analysis based *only* on the text provided above. Do NOT add external knowledge.
- Include actionable recommendations *if supported by the text*.
- Structure the report clearly (e.g., Findings, Analysis, Recommendations).
- **Crucially, cite the sources for your claims.** Use the source information provided within the '===' markers (e.g., "[Source: Title (URL)]" or "[Source: Uploaded File: filename]"). If content source isn't clear from the markers, state that.
- Respond with a detailed Markdown-formatted report.
"""
                            # Add citations list to prompt context if helpful? Maybe not needed if markers are used.
                            # synthesis_prompt += f"\n\nAvailable Sources:\n{chr(10).join(citations)}"


                            try:
                                # Use the gemini_client wrapper or SDK directly
                                messages = [{"role": "user", "content": synthesis_prompt}]
                                if hasattr(gemini_client_instance, 'chat'):
                                     response_data = gemini_client_instance.chat(messages)
                                     comprehensive_report = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse synthesis response.")
                                elif hasattr(gemini_client_instance, 'generate_content'): # If using SDK model directly
                                     # Need to handle potential async call if using SDK directly
                                     response = await gemini_client_instance.generate_content(synthesis_prompt) # Adjust if sync
                                     comprehensive_report = response.text
                                else:
                                     comprehensive_report = "Error: Gemini client misconfigured for synthesis."

                                logger.info("Comprehensive report generated via RAG.")
                                st.success("RAG analysis and report synthesis complete.")

                            except Exception as synth_error:
                                st.error(f"Report synthesis with Gemini failed: {synth_error}")
                                logger.error(f"Gemini synthesis failed: {synth_error}\n{traceback.format_exc()}")
                                comprehensive_report = f"Report synthesis failed: {synth_error}"


            except Exception as rag_error:
                st.error(f"An error occurred during the RAG pipeline: {rag_error}")
                logger.error(f"RAG pipeline error: {rag_error}\n{traceback.format_exc()}")
                comprehensive_report = f"RAG pipeline failed: {rag_error}"


    analysis_results['comprehensive_report'] = comprehensive_report
    analysis_results['citations'] = citations # Store citations separately if needed

    # --- 4. Patient-Friendly Summary Generation ---
    st.markdown("### 4. Generating Patient-Friendly Summary...")
    patient_summary_report = "Summary generation did not run or failed."
    if comprehensive_report and "failed" not in comprehensive_report.lower():
        with st.spinner("Generating patient-friendly summary..."):
            prompt = f"""You are a medical assistant skilled at explaining complex diagnostic reports in simple, clear language for patients. Based ONLY on the following comprehensive diagnostic report, produce a short summary (2-3 paragraphs) that a non-medical professional can easily understand.

Comprehensive Diagnostic Report:
--- START REPORT ---
{comprehensive_report}
--- END REPORT ---

Instructions:
- Summarize the main findings clearly and concisely.
- List any specific, actionable recommendations mentioned in the report (like tests, referrals, lifestyle changes).
- Tell the patient the suggested next steps based *only* on the report content.
- Use simple, empathetic language. Avoid jargon.
- Do NOT add information not present in the report above.
"""
            try:
                messages = [{"role": "user", "content": prompt}]
                if hasattr(gemini_client_instance, 'chat'):
                     response_data = gemini_client_instance.chat(messages)
                     patient_summary_report = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse patient summary response.")
                elif hasattr(gemini_client_instance, 'generate_content'):
                     response = await gemini_client_instance.generate_content(prompt) # Adjust if sync
                     patient_summary_report = response.text
                else:
                     patient_summary_report = "Error: Gemini client misconfigured for patient summary."

                logger.info("Patient-friendly summary generated.")
                st.success("Patient-friendly summary generated.")

            except Exception as patient_summary_error:
                st.error(f"Patient-friendly summary generation failed: {patient_summary_error}")
                logger.error(f"Patient summary generation failed: {patient_summary_error}\n{traceback.format_exc()}")
                patient_summary_report = f"Patient summary generation failed: {patient_summary_error}"
    else:
        st.warning("Skipping patient-friendly summary because the comprehensive report was not generated successfully.")

    analysis_results['patient_summary_report'] = patient_summary_report

    st.balloons()
    st.success("Analysis Pipeline Completed!")

    return analysis_results

# --- Streamlit UI Layout ---

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")

    # Input Query
    query = st.text_area("1. Enter Patient Symptoms/Issue:", height=150, key="query_input")

    # URLs
    st.subheader("Web Search Options")
    include_urls_str = st.text_area("Include Specific URLs (one per line, optional):", height=50, key="include_urls")
    omit_urls_str = st.text_area("Omit URLs Containing (one per line, optional):", height=50, key="omit_urls")
    search_depth = st.selectbox("Search Depth:", ["basic", "advanced"], index=1, key="search_depth")
    search_breadth = st.number_input("Search Breadth (results per query):", min_value=3, max_value=20, value=7, key="search_breadth") # Reduced default

    # File Uploads
    st.subheader("Reference Files (Optional)")
    uploaded_files = st.file_uploader(
        "Upload Local Files (PDF, DOCX, CSV, XLSX, TXT):",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'csv', 'xlsx', 'xls', 'txt'],
        key="file_uploader"
    )
    uploaded_wearable_file = st.file_uploader(
        "Upload Wearable Data CSV (Optional):",
        accept_multiple_files=False,
        type=['csv'],
        key="wearable_uploader"
    )
    uploaded_survey_file = st.file_uploader(
        "Upload Survey Data CSV (for Diagnosis Prediction):",
        accept_multiple_files=False,
        type=['csv'],
        key="survey_uploader"
    )

    # Advanced Options
    st.subheader("Advanced Options")
    use_faiss = st.checkbox("Use FAISS for RAG Matching (requires FAISS install)", value=True, key="use_faiss")

    # Submit Button
    st.divider()
    submit_button = st.button("Run Diagnostic Analysis", type="primary", key="submit_button")

# --- Main Area for Outputs ---
st.header("Analysis Results")

# Initialize session state variables
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = ""
if 'entities' not in st.session_state:
    st.session_state.entities = ""
if 'prediction' not in st.session_state:
    st.session_state.prediction = {}
if 'explanation' not in st.session_state:
    st.session_state.explanation = ""
if 'classifier_trained' not in st.session_state:
    st.session_state.classifier_trained = False
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'model' not in st.session_state:
    st.session_state.model = None


# --- Train Classifier ---
# Train the classifier if survey data is uploaded, happens outside the main button click
# to allow caching and prevent retraining on every run.
survey_df = None
if uploaded_survey_file:
    try:
        survey_df = pd.read_csv(uploaded_survey_file)
        st.sidebar.success(f"Loaded survey data: {uploaded_survey_file.name}")
        # Train or load cached model
        vectorizer, model = train_symptom_classifier(survey_df)
        if vectorizer and model:
             st.session_state.classifier_trained = True
             st.session_state.vectorizer = vectorizer
             st.session_state.model = model
             logger.info("Classifier ready.")
        else:
             st.session_state.classifier_trained = False
             st.sidebar.warning("Classifier training failed or yielded no model.")

    except Exception as e:
        st.sidebar.error(f"Error loading survey data: {e}")
        logger.error(f"Survey data loading error: {e}")
        st.session_state.classifier_trained = False
else:
    # Clear classifier state if file is removed
    st.session_state.classifier_trained = False
    st.session_state.vectorizer = None
    st.session_state.model = None


# --- Handle Analysis on Button Click ---
if submit_button and query:
    st.session_state.analysis_complete = False # Reset flag
    st.session_state.results = {} # Clear previous results

    # Prepare inputs
    include_urls_list = [url.strip() for url in include_urls_str.split('\n') if url.strip()]
    omit_urls_list = [url.strip() for url in omit_urls_str.split('\n') if url.strip()]

    # Process uploaded files
    additional_files_content = {}
    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} uploaded files...")
        for file in uploaded_files:
            with st.spinner(f"Extracting text from {file.name}..."):
                content = extract_text_from_file(file)
                if content:
                    additional_files_content[file.name] = content
                else:
                    st.warning(f"Could not extract text from {file.name}.")

    # Process wearable data
    wearable_summary = None
    if uploaded_wearable_file:
        with st.spinner("Processing wearable data..."):
             wearable_summary = read_wearable_data(uploaded_wearable_file)
             if wearable_summary:
                 st.write("Wearable data summary generated.")


    # --- Run Initial Gemini Analysis (Sentiment, Entities) ---
    st.markdown("### Preliminary Analysis")
    with st.spinner("Analyzing sentiment and extracting entities..."):
         # Run async functions using asyncio.run or Streamlit's event loop management
         try:
             # Create placeholder tasks first
             sentiment_task = asyncio.ensure_future(analyze_sentiment_gemini(query, gemini_client))
             entities_task = asyncio.ensure_future(extract_medical_entities_gemini(query, gemini_client))

             # Await tasks (this might need adjustment based on Streamlit version)
             # If running in Streamlit's main thread, use asyncio.run() carefully
             # Or structure differently if Streamlit handles the loop
             # For simplicity, let's assume direct await works or use asyncio.run
             loop = asyncio.get_event_loop()
             st.session_state.sentiment = loop.run_until_complete(sentiment_task)
             st.session_state.entities = loop.run_until_complete(entities_task)

             # Alternative using asyncio.run (might block if not careful)
             # st.session_state.sentiment = asyncio.run(analyze_sentiment_gemini(query, gemini_client))
             # st.session_state.entities = asyncio.run(extract_medical_entities_gemini(query, gemini_client))

         except Exception as initial_analysis_err:
              st.error(f"Error during initial sentiment/entity analysis: {initial_analysis_err}")
              logger.error(f"Initial Gemini analysis failed: {initial_analysis_err}")
              st.session_state.sentiment = "Error"
              st.session_state.entities = "Error"

    # Display Sentiment & Entities
    with st.expander("Sentiment Analysis (Gemini)", expanded=False):
        st.write(st.session_state.sentiment)
    with st.expander("Extracted Medical Entities (Gemini)", expanded=False):
        st.write(st.session_state.entities)

    # --- Run Diagnosis Prediction & Explanation (if model trained) ---
    if st.session_state.classifier_trained:
        st.markdown("### Diagnosis Prediction (Experimental)")
        with st.spinner("Predicting diagnosis and generating explanation..."):
            pred_diag, pred_proba = predict_diagnosis(query, st.session_state.vectorizer, st.session_state.model)
            st.session_state.prediction = {"diagnosis": pred_diag, "probabilities": pred_proba}

            # Generate SHAP explanation
            st.session_state.explanation = explain_diagnosis_shap(query, st.session_state.vectorizer, st.session_state.model)

        st.write(f"**Predicted Diagnosis:** {st.session_state.prediction.get('diagnosis', 'N/A')}")
        with st.expander("Prediction Probabilities", expanded=False):
             st.json(st.session_state.prediction.get('probabilities', {}))
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
        st.info("Diagnosis prediction skipped (Survey data for training not provided or failed to load).")
        # Save query history without prediction
        save_query_history(query, "N/A - Model not trained", st.session_state.sentiment, st.session_state.entities)


    # --- Run the Main Analysis Pipeline ---
    try:
        # Instantiate SubjectAnalyzer here if needed, passing the gemini client
        # Ensure AnalysisConfig is defined or imported
        analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
        subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config)

        # Run the main async pipeline
        # Use asyncio.run() to execute the async function from the sync Streamlit context
        analysis_results = asyncio.run(run_analysis_pipeline(
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
        ))
        st.session_state.results = analysis_results
        st.session_state.analysis_complete = True

    except Exception as pipeline_error:
        st.error(f"An error occurred during the main analysis pipeline: {pipeline_error}")
        logger.error(f"Main pipeline execution failed: {pipeline_error}\n{traceback.format_exc()}")
        st.session_state.analysis_complete = False


# --- Display Results After Analysis ---
if st.session_state.analysis_complete:
    st.divider()
    st.header("Generated Reports")

    results = st.session_state.results
    comp_report = results.get('comprehensive_report', 'Not generated.')
    patient_report = results.get('patient_summary_report', 'Not generated.')

    # Display Reports
    st.subheader("Comprehensive Diagnostic Report")
    st.markdown(comp_report) # Display markdown

    st.subheader("Patient-Friendly Summary")
    st.markdown(patient_report)

    # Download Buttons
    st.divider()
    st.subheader("Download Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Comprehensive Report (.md)",
            data=comp_report,
            file_name=safe_filename(query, "comprehensive_report.md"),
            mime="text/markdown",
        )
    with col2:
        st.download_button(
            label="Download Patient Summary (.md)",
            data=patient_report,
            file_name=safe_filename(query, "patient_summary.md"),
            mime="text/markdown",
        )

    # Display Citations (Optional)
    citations = results.get('citations', [])
    if citations:
        with st.expander("View Citations", expanded=False):
            for cit in citations:
                st.write(f"- {cit}")

    # Display Visualizations (Optional)
    st.divider()
    st.header("Query Trend Visualizations (Based on `query_history.csv`)")
    if st.button("Generate Trend Visualizations"):
        with st.spinner("Generating visualizations..."):
            figs = visualize_query_trends()
            if figs:
                for name, fig in figs.items():
                    st.pyplot(fig)
                    # Add download button for the plot
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format="png", bbox_inches='tight')
                    st.download_button(
                        label=f"Download {name.replace('_', ' ')} plot (.png)",
                        data=img_buffer,
                        file_name=f"{name}.png",
                        mime="image/png"
                    )
                    plt.close(fig) # Close the figure to free memory
            else:
                st.info("No visualizations could be generated.")


elif submit_button and not query:
    st.warning("Please enter patient symptoms or issue.")

# Add a footer or additional information
st.divider()
st.caption("Medical Diagnostic Agent - Experimental Use Only")
```

**To Run This App:**

1.  **Save the code:** Save the code above as a Python file (e.g., `streamlit_app.py`).
2.  **Install Dependencies:** Make sure you have all the required libraries installed. Create a `requirements.txt` file:
    ```txt
    streamlit
    google-generativeai
    python-dotenv # For local secrets if needed
    rich
    PyPDF2
    python-docx
    pandas
    openpyxl # For Excel file reading
    supabase
    scikit-learn
    # nltk # Removed
    # spacy # Removed
    shap >= 0.41.0 # Ensure recent version for LinearExplainer updates
    faiss-cpu # Or faiss-gpu if you have GPU support
    matplotlib
    seaborn
    # tavily-python # Assuming your custom clients handle Tavily
    # Add paths or installation instructions for your custom modules:
    # web_agent @ git+https://github.com/your_repo/web_agent.git@main#egg=web_agent
    # subject_analyzer @ git+https://github.com/your_repo/subject_analyzer.git@main#egg=subject_analyzer
    # OR if they are local directories:
    # ./web_agent
    # ./subject_analyzer
    ```
    * **Important:** You need to make your custom `web_agent` and `subject_analyzer` packages installable (e.g., via `pip install -e .` if they have `setup.py` or `pyproject.toml`) or ensure their directories are in the Python path relative to `streamlit_app.py`. The example above shows how you might include them in `requirements.txt` if they are local directories or Git repositories. Adjust as needed.
    * Install using: `pip install -r requirements.txt`
    * If you removed NLTK/spaCy data downloads, ensure they are not called anywhere. The provided code replaces their functions with Gemini calls.
3.  **Configure Secrets:**
    * **Streamlit Cloud:** Add your `GEMINI_API_KEY`, `TAVILY_API_KEY`, `SUPABASE_URL`, and `SUPABASE_KEY` to the Streamlit Cloud secrets manager.
    * **Local:** Create a file `.streamlit/secrets.toml` in the same directory as your app with the following structure:
        ```toml
        [api_keys]
        GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
        TAVILY_API_KEY = "YOUR_TAVILY_API_KEY"

        [supabase]
        SUPABASE_URL = "YOUR_SUPABASE_URL"
        SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY" # Use the anon key

        # Optional: Specify models
        # [models]
        # GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
        # GOOGLE_EMBEDDING_MODEL = "text-embedding-004"
        ```
4.  **Supabase Setup:** Ensure your Supabase project has the `embeddings` table and the `match_chunks` RPC function set up correctly, compatible with the embedding dimensions of your chosen Google model (`text-embedding-004` has 768 dimensions).
5.  **Run Streamlit:** Open your terminal in the directory where you saved the file and run:
    ```bash
    streamlit run streamlit_app.py
    ```

This will launch the interactive Streamlit application in your browser. You can then input symptoms, upload files, configure options, and run the analys
