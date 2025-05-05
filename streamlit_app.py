# streamlit_app.py
# Optimized for Streamlit Cloud Deployment

import streamlit as st
import os
import sys
import asyncio # Note: Streamlit runs synchronously, asyncio usage might be limited or require workarounds
import logging
import csv
import tempfile
from datetime import datetime
# Removed: from dotenv import load_dotenv (Not needed for Streamlit Cloud secrets)
from rich.console import Console # Can be used for internal logging if needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from io import BytesIO

# --- Streamlit Cloud Deployment Note ---
# This app expects API keys to be set via Streamlit Secrets.
# Create a .streamlit/secrets.toml file locally (DO NOT COMMIT TO GIT):
#
# [secrets]
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
# TAVILY_API_KEY = "YOUR_TAVILY_API_KEY"
# SUPABASE_URL = "YOUR_SUPABASE_URL"
# SUPABASE_KEY = "YOUR_SUPABASE_KEY"
# # Optional:
# # GEMINI_MODEL_NAME = "gemini-..."
# # GOOGLE_EMBEDDING_MODEL = "text-embedding-..."
#
# When deploying, Streamlit Cloud will prompt you to copy these secrets.
# ---------------------------------------

# Configure logging (optional for Streamlit, but can be useful for debugging in logs)
logging.basicConfig(
    level=logging.INFO, # Use INFO level for deployment
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout # Log to stdout for Streamlit Cloud logs
)

# --- Dependency Installation Helpers & Caching ---
# Use @st.cache_resource for functions that download/load models or data
# to avoid re-doing it on every script rerun within the same session.

@st.cache_resource
def install_nltk_data():
    """Downloads required NLTK data if not present."""
    try:
        import nltk
        # Check if already downloaded to avoid unnecessary calls
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
            st.toast("Downloading NLTK VADER lexicon...", icon="⏳")
            nltk.download('vader_lexicon', quiet=True)
            st.toast("NLTK VADER lexicon downloaded.", icon="✅")
        return True # Indicate success or already present
    except ImportError:
         st.error("NLTK library not found. Please add 'nltk' to requirements.txt")
         logging.error("NLTK library not found.")
         return False
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")
        logging.error(f"Failed to download NLTK data: {e}")
        return False

@st.cache_resource
def install_spacy_model(model_name="en_core_web_sm"):
    """Downloads and loads a spaCy model if not present."""
    try:
        import spacy
        try:
            nlp = spacy.load(model_name)
            logging.info(f"spaCy model '{model_name}' already available.")
            return nlp # Return the loaded model
        except OSError:
            st.toast(f"Downloading spaCy model '{model_name}'...", icon="⏳")
            spacy.cli.download(model_name)
            st.toast(f"spaCy model '{model_name}' downloaded.", icon="✅")
            # Need to reload spacy after download in some environments
            import importlib
            importlib.reload(spacy)
            nlp = spacy.load(model_name)
            return nlp # Return the loaded model
    except ImportError:
        st.error("spaCy library not found. Please add 'spacy' to requirements.txt")
        logging.error("spaCy library not found.")
        return None
    except Exception as e:
        st.error(f"Failed to download or load spaCy model '{model_name}': {e}")
        logging.error(f"Failed to download or load spaCy model '{model_name}': {e}")
        return None

# Attempt to download data/models when the app starts or cache is cleared
nltk_ready = install_nltk_data()
spacy_nlp = install_spacy_model() # Load the default model

# --- Simplified File Extraction ---
# (Adapted from original script, removing console prints)
def extract_text_from_file(uploaded_file):
    """Extracts text content from various uploaded file types."""
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext == '.pdf':
            try:
                import PyPDF2
            except ImportError:
                st.error("PyPDF2 is required for PDF extraction. Add 'PyPDF2' to requirements.txt.")
                return ""
            text = ""
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif ext == '.docx':
            try:
                import docx
            except ImportError:
                st.error("python-docx is required for DOCX extraction. Add 'python-docx' to requirements.txt.")
                return ""
            doc = docx.Document(uploaded_file)
            return "\n".join(para.text for para in doc.paragraphs)
        elif ext == '.csv':
            try:
                # Read CSV, handle potential encoding issues
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0) # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                return df.to_csv(index=False)
            except ImportError:
                st.error("pandas is required for CSV extraction. Add 'pandas' to requirements.txt.")
                return ""
            except Exception as e:
                 st.warning(f"Could not read CSV file {name}: {e}")
                 return ""
        elif ext in ['.xls', '.xlsx']:
            try:
                # Ensure openpyxl is installed for xlsx
                df = pd.read_excel(uploaded_file, engine='openpyxl' if ext == '.xlsx' else None)
                return df.to_csv(index=False)
            except ImportError:
                st.error("pandas and openpyxl (for .xlsx) are required. Add them to requirements.txt.")
                return ""
            except Exception as ex:
                 st.warning(f"Could not read Excel file {name}. Ensure 'openpyxl' is in requirements.txt: {ex}")
                 return ""
        elif ext == '.txt':
            # Read as text, handling potential encoding issues
            try:
                return uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                 try:
                     uploaded_file.seek(0) # Reset file pointer
                     return uploaded_file.read().decode("latin-1")
                 except Exception as e:
                     st.warning(f"Could not decode text file {name}: {e}")
                     return ""
        else:
            st.warning(f"Unsupported file type: {ext} for file {name}")
            return ""
    except Exception as ex:
        st.error(f"Error extracting file {name}: {ex}")
        logging.error(f"File extraction error for {name}: {ex}")
        return ""


# --- Data Science / Analytics & Visualization Functions ---

@st.cache_data # Cache the trained model based on survey data content hash
def train_symptom_classifier(survey_df):
    """
    Trains a simple symptom-to-diagnosis classifier using the survey data DataFrame.
    Uses dummy data if survey_df is None or invalid.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if survey_df is None or survey_df.empty:
        st.info("No valid survey data provided for training classifier. Using dummy data.")
        logging.info("Training classifier with dummy data.")
        # Fallback to dummy data
        data = [
            ("fever cough sore throat", "Flu"), ("headache nausea sensitivity to light", "Migraine"),
            ("chest pain shortness of breath", "Heart Attack"), ("joint pain stiffness", "Arthritis"),
            ("abdominal pain diarrhea vomiting", "Gastroenteritis")
        ]
        texts, labels = zip(*data)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, labels)
        return vectorizer, model

    try:
        symptoms_col = 'What are the current symptoms or health issues you are facing'
        history_col = 'Medical Health History'

        if symptoms_col not in survey_df.columns or history_col not in survey_df.columns:
             st.error(f"Survey data missing required columns: '{symptoms_col}' or '{history_col}'. Using dummy classifier.")
             logging.error("Survey data missing required columns for classifier.")
             return train_symptom_classifier(None) # Recurse with None to get dummy

        symptoms = survey_df[symptoms_col].fillna("").astype(str).tolist()
        labels = survey_df[history_col].fillna("None").astype(str).tolist()
        processed_labels = [label.split(',')[0].strip() for label in labels]

        # Filter out entries with no symptoms or no relevant medical history
        combined_data = [(s, l) for s, l in zip(symptoms, processed_labels) if s and l != 'None']
        if not combined_data:
            st.warning("No valid data pairs (symptom, history) found in survey report after filtering. Using dummy classifier.")
            logging.warning("No valid data pairs found for classifier training.")
            return train_symptom_classifier(None) # Recurse with None

        texts, labels = zip(*combined_data)
        logging.info(f"Training classifier with {len(texts)} samples from survey data.")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=1000) # Increase max_iter if convergence issues
        model.fit(X, labels)
        return vectorizer, model

    except Exception as e:
        st.error(f"An error occurred during classifier training: {e}. Using dummy classifier.")
        logging.error(f"Classifier training error: {e}")
        return train_symptom_classifier(None) # Fallback to dummy

def predict_diagnosis(query, vectorizer, model):
    """Predicts a diagnosis from the patient query."""
    if not query or vectorizer is None or model is None:
        logging.warning("Prediction skipped: Missing query, vectorizer, or model.")
        return "Not available", {}
    try:
        X_query = vectorizer.transform([query])
        pred = model.predict(X_query)[0]
        proba = model.predict_proba(X_query)[0]
        prob_dict = dict(zip(model.classes_, proba))
        # Sort probabilities descending for display
        sorted_proba = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        return pred, sorted_proba
    except Exception as e:
        st.error(f"Error during diagnosis prediction: {e}")
        logging.error(f"Diagnosis prediction error: {e}")
        return "Error", {}

def analyze_sentiment(query):
    """Performs sentiment analysis using NLTK VADER."""
    global nltk_ready # Check if NLTK setup was successful
    if not query:
        return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0}
    if not nltk_ready:
         st.warning("NLTK VADER lexicon not available. Sentiment analysis skipped.")
         return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0}
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(query)
        return score
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        logging.error(f"Sentiment analysis error: {e}")
        return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0}

def extract_medical_entities(query):
    """Extracts medical entities using spaCy."""
    global spacy_nlp # Use the globally loaded spaCy model
    if not query:
        return []
    if spacy_nlp is None:
         st.warning("spaCy model not loaded. Entity extraction skipped.")
         return []
    try:
        doc = spacy_nlp(query)
        # Expanded list - customize as needed
        common_med_terms = {
            "flu", "migraine", "heart attack", "arthritis", "gastroenteritis", "diabetes",
            "hypertension", "asthma", "cancer", "stroke", "anxiety", "depression",
            "fever", "cough", "pain", "diarrhea", "vomiting", "headache", "fatigue",
            "nausea", "chest pain", "shortness of breath", "sore throat", "dizziness",
            "rash", "swelling", "numbness", "allergy", "infection", "inflammation"
        }
        entities = set()
        for ent in doc.ents:
            # Simple check: lowercase text in common terms or if label indicates medical condition
            if ent.text.lower() in common_med_terms or ent.label_ in ["DISEASE", "SYMPTOM", "CONDITION", "PROBLEM", "MEDICAL_CONDITION"]: # Example labels
                entities.add(ent.text)
        # Also check simple noun chunks for keywords if entities are sparse
        for chunk in doc.noun_chunks:
             # Check root text of the chunk
             if chunk.root.text.lower() in common_med_terms:
                 entities.add(chunk.text) # Add the full chunk text
             elif chunk.text.lower() in common_med_terms: # Check full chunk text
                  entities.add(chunk.text)

        return sorted(list(entities)) # Return sorted list
    except Exception as e:
        st.error(f"Error during entity extraction: {e}")
        logging.error(f"Entity extraction error: {e}")
        return []

# --- Query History (Stored in Streamlit Cloud ephemeral filesystem) ---
QUERY_HISTORY_FILE = "query_history.csv"

@st.cache_data # Cache the DataFrame reading
def get_query_history_df():
    """Loads query history from CSV into a pandas DataFrame."""
    if os.path.exists(QUERY_HISTORY_FILE):
        try:
            return pd.read_csv(QUERY_HISTORY_FILE)
        except pd.errors.EmptyDataError:
             return pd.DataFrame(columns=["timestamp", "query", "predicted_diagnosis", "sentiment_compound", "entities"])
        except Exception as e:
            st.error(f"Error reading query history file '{QUERY_HISTORY_FILE}': {e}")
            return pd.DataFrame() # Return empty df on error
    else:
        # Return an empty DataFrame with expected columns if file doesn't exist
        return pd.DataFrame(columns=["timestamp", "query", "predicted_diagnosis", "sentiment_compound", "entities"])

def save_query_history(query, diagnosis, sentiment, entities):
    """Saves the query details to a CSV file."""
    # Note: Filesystem on Streamlit Cloud is ephemeral, history persists only for the session duration.
    # For persistent history, consider saving to an external database (like Supabase).
    file_exists = os.path.isfile(QUERY_HISTORY_FILE)
    try:
        with open(QUERY_HISTORY_FILE, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.path.getsize(QUERY_HISTORY_FILE) == 0:
                writer.writerow(["timestamp", "query", "predicted_diagnosis", "sentiment_compound", "entities"])
            writer.writerow([datetime.now().isoformat(), query, diagnosis, sentiment.get("compound", 0.0), ", ".join(entities)])
        # Update the cached history after saving
        get_query_history_df.clear() # Clear cache for get_query_history_df
    except Exception as e:
        st.error(f"Failed to save query history: {e}")
        logging.error(f"Failed to save query history: {e}")


# --- Visualization Functions ---
@st.cache_data # Cache plots based on the history dataframe content hash
def generate_visualization_figures(_df_history): # Use _ prefix to indicate input drives caching
    """Generates visualization figures from query history DataFrame."""
    figures = {}
    # Make a copy to avoid modifying the cached DataFrame
    df_history = _df_history.copy()

    if df_history is None or df_history.empty:
        logging.info("Query history is empty. Cannot generate visualizations.")
        return figures

    try:
        # Ensure timestamp is datetime type
        if 'timestamp' in df_history.columns and not pd.api.types.is_datetime64_any_dtype(df_history['timestamp']):
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], errors='coerce')
            df_history.dropna(subset=['timestamp'], inplace=True) # Drop rows where conversion failed

        # --- Bar plot for predicted diagnosis frequency ---
        if 'predicted_diagnosis' in df_history.columns:
            # Treat 'Not available' or 'Error' as a separate category or ignore
            valid_diagnoses = df_history[~df_history['predicted_diagnosis'].isin(['Not available', 'Error'])]
            diag_counts = valid_diagnoses['predicted_diagnosis'].value_counts()
            if not diag_counts.empty:
                fig_diag, ax_diag = plt.subplots(figsize=(8, max(6, len(diag_counts) * 0.5))) # Adjust height
                sns.barplot(x=diag_counts.values, y=diag_counts.index, palette="viridis", ax=ax_diag, orient='h') # Horizontal
                ax_diag.set_title("Frequency of Predicted Diagnoses")
                ax_diag.set_xlabel("Count")
                ax_diag.set_ylabel("Diagnosis")
                plt.tight_layout()
                figures['diagnosis_frequency'] = fig_diag
            else:
                 logging.info("No valid diagnosis data to plot.")
        else:
             logging.warning("Column 'predicted_diagnosis' not found in history for plotting.")


        # --- Line plot for sentiment compound over time ---
        if 'timestamp' in df_history.columns and 'sentiment_compound' in df_history.columns:
            df_sorted = df_history.sort_values("timestamp")
            if not df_sorted.empty and len(df_sorted) > 1: # Need at least 2 points for a line plot
                fig_sent, ax_sent = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df_sorted, x="timestamp", y="sentiment_compound", marker="o", ax=ax_sent)
                ax_sent.set_title("Sentiment Compound Score Over Time")
                ax_sent.set_xlabel("Timestamp")
                ax_sent.set_ylabel("Sentiment Compound Score (-1 to 1)")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                figures['sentiment_over_time'] = fig_sent
            elif not df_sorted.empty:
                 logging.info("Only one data point for sentiment, cannot plot line chart.")
            else:
                logging.info("No sentiment data to plot over time.")
        else:
             logging.warning("Columns 'timestamp' or 'sentiment_compound' not found for sentiment plot.")


        # --- Pie chart for common medical entities ---
        if 'entities' in df_history.columns:
            entity_list = []
            for entities_str in df_history['entities'].dropna().astype(str):
                if entities_str.strip():
                    entities = [e.strip() for e in entities_str.split(",") if e.strip()]
                    entity_list.extend(entities)

            if entity_list:
                entity_counts = Counter(entity_list)
                # Plot top N entities for clarity if too many
                top_n = 10
                top_entities = entity_counts.most_common(top_n)
                other_count = sum(count for entity, count in entity_counts.items() if entity not in dict(top_entities))

                labels = [entity for entity, count in top_entities]
                sizes = [count for entity, count in top_entities]

                if other_count > 0:
                    labels.append(f'Other ({len(entity_counts) - top_n})')
                    sizes.append(other_count)

                fig_ent, ax_ent = plt.subplots(figsize=(8, 8))
                ax_ent.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, pctdistance=0.85)
                ax_ent.set_title(f"Distribution of Top {len(labels)} Extracted Medical Entities")
                plt.tight_layout()
                figures['entities_distribution'] = fig_ent
            else:
                 logging.info("No entity data to plot.")
        else:
             logging.warning("Column 'entities' not found for entity plot.")


    except Exception as e:
        st.error(f"Error generating visualizations: {e}")
        logging.error(f"Error generating visualizations: {e}")
        figures = {} # Clear figures if error occurs
    finally:
        plt.close('all') # Close all matplotlib figures

    return figures

def display_visualizations(figures):
    """Displays generated figures in Streamlit and provides download buttons."""
    if not figures:
        st.info("No visualizations were generated or history is empty.")
        return

    st.subheader("Query Trend Visualizations")

    for name, fig in figures.items():
        st.pyplot(fig)
        # Provide download button
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight') # Use tight bbox
        buf.seek(0)
        st.download_button(
            label=f"Download {name.replace('_', ' ').title()} Plot",
            data=buf,
            file_name=f"{name}.png",
            mime="image/png",
            key=f"download_{name}" # Unique key for each download button
        )
        st.divider()


# --- FAISS Indexing Functions (Optional) ---
@st.cache_resource # Cache the FAISS index for the session
def build_faiss_index(embeddings):
    """Builds a FAISS index from a list of embeddings."""
    if not embeddings or not isinstance(embeddings[0], (np.ndarray, list)):
         st.warning("Invalid or empty embeddings provided for FAISS.")
         return None
    try:
        import faiss
        # Ensure embeddings are numpy float32 arrays
        embeddings_array = np.array(embeddings).astype('float32')
        if embeddings_array.ndim != 2:
             st.error(f"Embeddings have incorrect dimensions ({embeddings_array.ndim}D) for FAISS. Expected 2D.")
             return None
        dim = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dim)  # Using L2 distance
        index.add(embeddings_array)
        logging.info(f"FAISS index built successfully with {index.ntotal} vectors of dimension {dim}.")
        return index
    except ImportError:
        st.error("FAISS library not installed. Add 'faiss-cpu' to requirements.txt")
        return None
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        logging.error(f"Error building FAISS index: {e}")
        return None

def search_faiss(index, query_embedding, chunks, k):
    """Searches the FAISS index."""
    if index is None or query_embedding is None or not chunks:
        logging.warning("FAISS search skipped: Index, query embedding, or chunks missing.")
        return []
    try:
        import faiss
        import numpy as np
        query_vec = np.array(query_embedding).reshape(1, -1).astype('float32')
        # Adjust k if it's larger than the number of items in the index
        k_actual = min(k, index.ntotal)
        if k_actual <= 0:
             st.warning("FAISS index is empty or k=0, cannot search.")
             return []
        distances, indices = index.search(query_vec, k_actual)
        # Ensure indices are valid and within the bounds of the chunks list
        valid_indices = [i for i in indices[0] if 0 <= i < len(chunks)]
        matched_chunks = [chunks[i] for i in valid_indices]
        logging.info(f"FAISS retrieved {len(matched_chunks)} chunks (k={k_actual}).")
        return matched_chunks
    except ImportError:
        st.error("FAISS library not installed.")
        return []
    except Exception as e:
        st.error(f"Error searching FAISS index: {e}")
        logging.error(f"Error searching FAISS index: {e}")
        return []


# --- Wearable Data Function ---
@st.cache_data # Cache wearable data summary based on file content hash
def read_wearable_data(wearable_df):
    """Reads wearable data from DataFrame and returns a summary string."""
    if wearable_df is None or wearable_df.empty:
        return None # Indicate no data
    summary_lines = []
    logging.info(f"Processing wearable data with columns: {wearable_df.columns.tolist()}")
    try:
        # Convert relevant columns to numeric, coercing errors
        if "heart_rate" in wearable_df.columns:
            hr_numeric = pd.to_numeric(wearable_df["heart_rate"], errors='coerce')
            avg_hr = hr_numeric.mean()
            if pd.notna(avg_hr):
                summary_lines.append(f"- Average Heart Rate: {avg_hr:.1f} bpm (from {hr_numeric.count()} readings)")
        if "steps" in wearable_df.columns:
            steps_numeric = pd.to_numeric(wearable_df["steps"], errors='coerce')
            total_steps = steps_numeric.sum()
            if pd.notna(total_steps) and total_steps > 0:
                 summary_lines.append(f"- Total Steps Recorded: {int(total_steps)} (over {steps_numeric.count()} entries)")
        if "sleep_hours" in wearable_df.columns:
             sleep_numeric = pd.to_numeric(wearable_df["sleep_hours"], errors='coerce')
             avg_sleep = sleep_numeric.mean()
             if pd.notna(avg_sleep):
                 summary_lines.append(f"- Average Sleep: {avg_sleep:.1f} hours (from {sleep_numeric.count()} nights)")
        # Add more fields as needed (e.g., spo2, temperature)

        if not summary_lines:
             logging.warning("No relevant columns (e.g., 'heart_rate', 'steps', 'sleep_hours') found or contained valid numeric data in wearable data.")
             return "No relevant summary could be extracted from wearable data."

        logging.info("Wearable data summary generated.")
        return "\n".join(summary_lines)

    except Exception as e:
        st.error(f"Error processing wearable data: {e}")
        logging.error(f"Error processing wearable data: {e}")
        return "Error processing wearable data."


# --- SHAP Explanation Function ---
@st.cache_data # Cache explanation based on query and model/vectorizer state
def explain_diagnosis(_query, _vectorizer, _model): # Use _ prefix for cache inputs
    """Provides diagnosis explanation using SHAP (if possible)."""
    if not _query or _vectorizer is None or _model is None:
        return "Explanation not available (missing model, vectorizer, or query)."
    if not nltk_ready or spacy_nlp is None: # Check dependencies needed by SHAP indirectly
         return "Explanation requires NLTK/spaCy models to be loaded."

    try:
        import shap
        # Check if the vectorizer is fitted and model has classes
        if not hasattr(_vectorizer, 'vocabulary_') or not hasattr(_model, 'classes_'):
             logging.warning("SHAP: Vectorizer or model not ready.")
             return "Explanation not available (model or vectorizer not fitted)."

        # --- Use LinearExplainer for Logistic Regression ---
        # It's generally faster and more direct for linear models.
        try:
            # Create background data: Use feature names as a simple proxy
            # Better: Use a small sample of actual training texts if available
            num_background_samples = min(50, len(_vectorizer.get_feature_names_out()))
            if num_background_samples == 0:
                 logging.warning("SHAP: No features in vectorizer for background data.")
                 return "Explanation not available (vectorizer has no features)."

            background_feature_texts = np.random.choice(_vectorizer.get_feature_names_out(), num_background_samples, replace=False)
            background_sparse = _vectorizer.transform(background_feature_texts)

            if background_sparse.shape[0] == 0:
                 logging.warning("SHAP: Background data transformation resulted in empty set.")
                 return "Explanation not available (empty background data)."

            # Initialize explainer
            explainer = shap.LinearExplainer(_model, background_sparse, feature_perturbation="interventional")

            # Prepare the query data (sparse matrix)
            X_query = _vectorizer.transform([_query])

            # Get SHAP values
            shap_values = explainer.shap_values(X_query) # Pass sparse query data

        except Exception as linear_explainer_err:
             logging.warning(f"SHAP LinearExplainer failed ({linear_explainer_err}), explanation unavailable.")
             return f"Explanation generation failed (LinearExplainer error: {linear_explainer_err})."


        # --- Process SHAP values ---
        # shap_values can be a list (one per class) or a single array
        if isinstance(shap_values, list):
            # For multi-class, find the SHAP values for the predicted class
            predicted_class_index = np.where(_model.classes_ == _model.predict(X_query)[0])[0][0]
            shap_values_for_class = shap_values[predicted_class_index]
        else:
            # For binary or single output models
            shap_values_for_class = shap_values[0] # Assuming shape (1, n_features)

        # Get feature names
        feature_names = _vectorizer.get_feature_names_out()

        # Map SHAP values to feature names (handle sparse matrix indexing)
        non_zero_indices = X_query.indices
        # Ensure indices are within bounds of shap_values_for_class
        valid_indices = [idx for idx in non_zero_indices if idx < len(shap_values_for_class)]
        if not valid_indices:
             logging.info("SHAP: Query vector has no features overlapping with SHAP values.")
             return "Explanation: Query features did not significantly influence the prediction according to SHAP."

        shap_values_dense = shap_values_for_class[valid_indices]
        feature_names_dense = feature_names[valid_indices]

        # Sort features by absolute SHAP value
        abs_shap = np.abs(shap_values_dense)
        top_indices_sorted = np.argsort(abs_shap)[::-1]

        # Get top N features
        n_features_to_show = 7 # Show a few more features
        explanation = "**Top Contributing Features for Prediction:**\n\n"
        explanation += "| Feature        | SHAP Value | Influence                                  |\n"
        explanation += "|----------------|------------|--------------------------------------------|\n"

        for i in range(min(n_features_to_show, len(top_indices_sorted))):
            idx = top_indices_sorted[i]
            feature = feature_names_dense[idx]
            shap_val = shap_values_dense[idx]
            influence = "Increases likelihood" if shap_val > 0 else "Decreases likelihood"
            explanation += f"| `{feature}` | {shap_val:+.3f} | {influence} |\n"


        if not top_indices_sorted.size > 0:
             explanation = "No significant features found influencing this specific prediction via SHAP."

        explanation += "\n*Note: SHAP values indicate the influence of each feature on the model's prediction for this specific query.*"
        logging.info("SHAP explanation generated successfully.")
        return explanation

    except ImportError:
        logging.warning("SHAP library not installed.")
        return "Explanation requires SHAP library. Add 'shap' to requirements.txt"
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")
        logging.error(f"SHAP explanation error: {e}", exc_info=True)
        return f"Explanation not available due to an error: {e}"


# --- Core Agent Logic ---

# Helper to get API key safely from Streamlit Secrets ONLY
def get_secret(key_name):
    """Retrieves a secret value from Streamlit secrets. Returns None if not found."""
    if hasattr(st, 'secrets') and key_name in st.secrets:
        return st.secrets[key_name]
    else:
        # For Streamlit Cloud deployment, keys MUST be in st.secrets
        logging.warning(f"Secret '{key_name}' not found in st.secrets.")
        return None

# Helper to initialize clients (cached) - Optimized for Cloud
@st.cache_resource(ttl=3600) # Cache clients for an hour
def initialize_clients():
    """Initializes API clients using secrets from st.secrets."""
    clients = {}
    errors = []
    missing_secrets = [] # Track missing keys specifically

    # --- Get API Keys Securely ---
    GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
    TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
    SUPABASE_URL = get_secret("SUPABASE_URL")
    SUPABASE_KEY = get_secret("SUPABASE_KEY")

    # --- Gemini Client ---
    if GEMINI_API_KEY:
        try:
            # Ensure google-generativeai is installed
            import google.generativeai as genai
            from subject_analyzer.src.services.gemini_client import GeminiClient # Assuming this exists
            from subject_analyzer.src.models.analysis_models import AnalysisConfig # Assuming this exists

            # Configure Gemini API
            genai.configure(api_key=GEMINI_API_KEY)

            # Model names can also be secrets
            GEMINI_MODEL_NAME = get_secret("GEMINI_MODEL_NAME") or "gemini-1.5-flash" # Default model
            GOOGLE_EMBEDDING_MODEL = get_secret("GOOGLE_EMBEDDING_MODEL") or "text-embedding-004" # Default embedding

            analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
            # Pass the embedding model name to the client if it uses it
            clients['gemini'] = GeminiClient(api_key=GEMINI_API_KEY, config=analysis_config, embedding_model=GOOGLE_EMBEDDING_MODEL)
            clients['gemini_embedding_model'] = GOOGLE_EMBEDDING_MODEL # Store separately for clarity
            logging.info(f"Gemini client initialized (Model: {GEMINI_MODEL_NAME}, Embedding: {GOOGLE_EMBEDDING_MODEL}).")
        except ImportError as ie:
             # Check if it's the google library or your custom modules
             if 'google.generativeai' in str(ie):
                 errors.append("Google AI library not found. Add 'google-generativeai' to requirements.txt")
             elif 'subject_analyzer' in str(ie):
                 errors.append("GeminiClient/AnalysisConfig components not found. Check subject_analyzer module.")
             else:
                 errors.append(f"Import error during Gemini client init: {ie}")
        except Exception as e:
            errors.append(f"Failed to initialize Gemini client: {e}")
    else:
        missing_secrets.append("GEMINI_API_KEY")


    # --- Tavily Client ---
    if TAVILY_API_KEY:
        try:
            # Ensure Tavily library and your modules are installed
            from tavily import TavilyClient as TavilyApiClient # Rename to avoid conflict if needed
            from web_agent.src.services.web_search import WebSearchService # Assuming these exist
            from web_agent.src.models.search_models import SearchConfig
            from subject_analyzer.src.services.tavily_extractor import TavilyExtractor

            search_config = SearchConfig() # Assuming default config is okay
            # Use the imported TavilyApiClient
            tavily_api_client = TavilyApiClient(api_key=TAVILY_API_KEY)
            clients['extractor'] = TavilyExtractor(api_key=TAVILY_API_KEY) # Your custom extractor
            # Your WebSearchService might need the TavilyApiClient instance
            clients['search_service'] = WebSearchService(tavily_api_client, search_config) # Pass the API client instance
            logging.info("Tavily clients initialized.")
        except ImportError as ie:
            if 'tavily' in str(ie):
                 errors.append("Tavily library not found. Add 'tavily-python' to requirements.txt")
            elif 'web_agent' in str(ie) or 'subject_analyzer' in str(ie):
                 errors.append("Tavily/WebAgent components not found. Check web_agent/subject_analyzer modules.")
            else:
                 errors.append(f"Import error during Tavily client init: {ie}")
        except Exception as e:
            errors.append(f"Failed to initialize Tavily clients: {e}")
    else:
        missing_secrets.append("TAVILY_API_KEY")


    # --- Supabase Client ---
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            # Ensure Supabase library is installed
            from supabase import create_client, Client
            clients['supabase'] = create_client(SUPABASE_URL, SUPABASE_KEY)
            # Optional: Test connection lightly? Be careful with read/write operations here.
            # Example: Check table existence (read-only)
            # try:
            #     clients['supabase'].table('embeddings').select('id', head=True, count='exact').execute()
            #     logging.info("Supabase connection tested successfully.")
            # except Exception as db_test_err:
            #     errors.append(f"Supabase connection test failed: {db_test_err}")
            logging.info("Supabase client initialized.")
        except ImportError:
            errors.append("Supabase library not found. Add 'supabase' to requirements.txt")
        except Exception as e:
            errors.append(f"Failed to initialize Supabase client: {e}")
    else:
        if not SUPABASE_URL: missing_secrets.append("SUPABASE_URL")
        if not SUPABASE_KEY: missing_secrets.append("SUPABASE_KEY")


    # --- Subject Analyzer ---
    if 'gemini' in clients:
         try:
            from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer # Assuming this exists
            if hasattr(clients['gemini'], 'config'):
                analysis_config = clients['gemini'].config
            else: # Fallback config if needed (shouldn't happen if Gemini init succeeded)
                 from subject_analyzer.src.models.analysis_models import AnalysisConfig
                 GEMINI_MODEL_NAME = get_secret("GEMINI_MODEL_NAME") or "gemini-1.5-flash"
                 analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)

            clients['subject_analyzer'] = SubjectAnalyzer(llm_client=clients['gemini'], config=analysis_config)
            logging.info("Subject Analyzer initialized.")
         except ImportError:
             errors.append("SubjectAnalyzer components not found. Check subject_analyzer module.")
         except Exception as e:
             errors.append(f"Failed to initialize Subject Analyzer: {e}")
    elif "GEMINI_API_KEY" not in missing_secrets:
         # Only add error if Gemini key was present but client init failed
         errors.append("Subject Analyzer requires Gemini client, which failed to initialize.")


    # Store errors and missing keys for feedback
    st.session_state.client_initialization_errors = errors
    st.session_state.missing_secrets = missing_secrets

    # Determine overall success
    clients_ok = not missing_secrets and not errors
    if not clients_ok:
         log_msg = "Client initialization failed."
         if missing_secrets: log_msg += f" Missing secrets: {', '.join(missing_secrets)}."
         if errors: log_msg += f" Errors: {'; '.join(errors)}."
         logging.error(log_msg)
    else:
         logging.info("All required clients initialized successfully.")


    # Return only successfully initialized clients and the status flag
    return {k: v for k, v in clients.items() if v is not None}, clients_ok


# --- Main Diagnostic Process Function ---
def run_diagnostic_process(query, include_urls, omit_urls, additional_files_content, search_depth, search_breadth, use_faiss, feedback=None):
    """
    Performs the core diagnostic steps: analysis, search, extraction, RAG, report generation.
    Returns report strings and citations. Relies on initialized clients.
    """
    # Reset status flags for this run
    st.session_state.log_messages = ["Starting diagnostic process..."]
    st.session_state.status = "Running..."
    st.session_state.error = None
    st.session_state.analysis_complete = False
    st.session_state.search_complete = False
    st.session_state.rag_complete = False
    st.session_state.report_generated = False

    # 1. Initialize clients (or get from cache)
    clients, clients_ok = initialize_clients() # Get status flag

    # --- Early Exit if Clients Failed ---
    if not clients_ok:
        error_msg = "ERROR: Cannot proceed due to configuration issues.\n\n"
        if st.session_state.get('missing_secrets'):
            error_msg += f"**Missing Secrets:** Please ensure the following are set in Streamlit Cloud secrets:\n`{', '.join(st.session_state['missing_secrets'])}`\n\n"
        if st.session_state.get('client_initialization_errors'):
             error_msg += "**Initialization Errors:**\n" + "\n".join(f"- {e}" for e in st.session_state['client_initialization_errors']) + "\n\nPlease check logs and requirements.txt."

        st.session_state.error = error_msg # Store detailed error
        st.session_state.status = "Configuration Error"
        logging.error(f"Stopping diagnostic process due to client initialization failure. Details: {error_msg}")
        # Return None for all expected outputs
        return None, None, None

    # --- Get Specific Clients (Now guaranteed to exist if clients_ok is True) ---
    try:
        gemini_client = clients['gemini']
        search_service = clients['search_service']
        extractor = clients['extractor']
        subject_analyzer = clients['subject_analyzer']
        supabase = clients['supabase']
        gemini_embedding_model = clients['gemini_embedding_model'] # Get embedding model name
    except KeyError as ke:
        # This should ideally not happen if clients_ok is True, but as a safeguard:
        st.session_state.error = f"Internal Error: Required client '{ke}' missing after successful initialization check. Please report this."
        st.session_state.status = "Error"
        logging.error(st.session_state.error)
        return None, None, None


    st.session_state.log_messages.append("API clients ready.")

    # Get current date for analysis context
    current_date = datetime.today().strftime("%Y-%m-%d")

    # Store results in session state to avoid recomputing on reruns if not necessary
    # Initialize or update task data
    if 'current_task' not in st.session_state or st.session_state.current_task.get("original_query") != query:
        st.session_state.current_task = {
            "original_query": query,
            "current_query": query,
            "feedback_history": [],
            "analysis": {},
            "search_results": {},
            "extracted_content": {},
            "comprehensive_report": None,
            "patient_summary": None,
            "citations": []
        }
    task = st.session_state.current_task

    # Apply feedback if provided
    if feedback:
        task["feedback_history"].append({
            "query": task["current_query"],
            "feedback": feedback,
            "time": datetime.now().isoformat()
        })
        task["current_query"] = f"{task['original_query']} - Additional context from user: {feedback}"
        st.info("Re-running analysis with feedback.")
        st.session_state.log_messages.append(f"Re-analyzing based on feedback: {feedback}")
        # Reset downstream steps for re-analysis
        st.session_state.analysis_complete = False
        st.session_state.search_complete = False
        st.session_state.rag_complete = False
        st.session_state.report_generated = False
        task['analysis'] = {} # Clear previous analysis


    # 2. Analyze Subject
    if not st.session_state.analysis_complete:
        st.session_state.log_messages.append(f"Analyzing query (Model: {subject_analyzer.config.model_name})...")
        try:
            analysis_query = f"{task['current_query']} (Analysis requested on {current_date})"
            # Assuming analyze method exists and works synchronously or handles async internally
            task['analysis'] = subject_analyzer.analyze(analysis_query)
            st.session_state.log_messages.append("Subject analysis successful.")
            st.session_state.analysis_complete = True
        except Exception as e:
            st.session_state.error = f"Subject analysis failed: {e}"
            st.session_state.status = "Error"
            logging.error(f"Subject analysis failed: {e}", exc_info=True)
            st.session_state.log_messages.append(f"[Error] Subject analysis failed: {e}")
            return task.get('comprehensive_report'), task.get('patient_summary'), task.get('citations') # Return current state

    # Display analysis results (moved to main layout)

    # 3. Search and Extract
    if not st.session_state.search_complete:
        st.session_state.log_messages.append("Starting web search and content extraction...")
        task['search_results'] = {}
        task['extracted_content'] = {}
        all_urls_found = [] # Collect all URLs to extract together

        try:
            if not include_urls: # Perform web search
                topics_to_search = [task['analysis'].get("main_subject", task['current_query'])]
                topics_to_search += task['analysis'].get("What_needs_to_be_researched", [])
                topics_to_search = list(set(filter(None, topics_to_search))) # Unique, non-empty

                if not topics_to_search:
                    st.warning("No specific topics identified for web search. Searching based on main query.")
                    topics_to_search = [task['current_query']]

                st.session_state.log_messages.append(f"Searching web for topics: {', '.join(topics_to_search)}")

                search_progress = st.progress(0, text=f"Searching web (Topic 1/{len(topics_to_search)})...")
                for i, topic in enumerate(topics_to_search):
                    status_text = f"Searching web (Topic {i+1}/{len(topics_to_search)}: '{topic}')..."
                    search_progress.progress((i+1)/len(topics_to_search), text=status_text)
                    st.session_state.log_messages.append(f"Searching: '{topic}' (Depth: {search_depth}, Breadth: {search_breadth})")
                    try:
                        # Assuming search_subject works synchronously
                        response = search_service.search_subject(
                            topic, "medical diagnosis information",
                            search_depth=search_depth,
                            results=search_breadth
                        )
                        results = response.get("results", [])
                        filtered_results = [
                            res for res in results
                            if res.get("url") and not any(omit.lower() in res.get("url").lower() for omit in omit_urls)
                        ]
                        task['search_results'][topic] = filtered_results
                        all_urls_found.extend([res.get("url") for res in filtered_results])
                        st.session_state.log_messages.append(f"Found {len(filtered_results)} results for '{topic}'.")
                    except Exception as e:
                        st.warning(f"Web search failed for topic '{topic}': {e}")
                        logging.warning(f"Web search failed for topic '{topic}': {e}")
                        task['search_results'][topic] = []
                search_progress.empty()

            else: # Use user-provided URLs
                st.session_state.log_messages.append(f"Using {len(include_urls)} user-provided URLs.")
                filtered_urls = [url for url in include_urls if not any(omit.lower() in url.lower() for omit in omit_urls)]
                all_urls_found.extend(filtered_urls)
                task['search_results']['user_provided'] = [{"title": "User Provided", "url": url, "score": "N/A"} for url in filtered_urls]


            # --- Extract content from all found URLs ---
            unique_urls = list(set(all_urls_found))
            if unique_urls:
                # Limit number of URLs to avoid excessive cost/time
                MAX_URLS_TO_EXTRACT = 20
                urls_to_extract = unique_urls[:MAX_URLS_TO_EXTRACT]
                if len(unique_urls) > MAX_URLS_TO_EXTRACT:
                     st.warning(f"Limiting content extraction to the first {MAX_URLS_TO_EXTRACT} relevant URLs found ({len(unique_urls)} total).")
                     st.session_state.log_messages.append(f"[Warning] Limiting extraction to {len(urls_to_extract)} URLs.")

                st.session_state.log_messages.append(f"Extracting content from {len(urls_to_extract)} unique URLs...")
                extract_progress = st.progress(0, text="Starting content extraction...")
                try:
                    # Assuming extract works synchronously
                    extraction_response = extractor.extract(
                        urls=urls_to_extract,
                        extract_depth="advanced", # Keep advanced for more content
                        include_images=False
                    )
                    extracted_items = extraction_response.get("results", [])
                    task['extracted_content']['combined_sources'] = extracted_items # Store all extracted
                    extract_progress.progress(1.0, text=f"Extraction complete ({len(extracted_items)} URLs processed).")
                    st.session_state.log_messages.append(f"Extracted content from {len(extracted_items)} URLs.")
                    failed_extractions = sum(1 for item in extracted_items if item.get("error"))
                    if failed_extractions:
                        st.warning(f"Failed to extract content from {failed_extractions} URLs.")
                        st.session_state.log_messages.append(f"[Warning] Failed extraction for {failed_extractions} URLs.")
                except Exception as e:
                    st.error(f"Content extraction failed: {e}")
                    st.session_state.log_messages.append(f"[Error] Content extraction failed: {e}")
                    logging.error(f"Content extraction failed: {e}", exc_info=True)
                    # Continue if possible, RAG will have less context
                finally:
                     extract_progress.empty()
            else:
                st.info("No valid URLs found from web search or user input to extract content.")
                st.session_state.log_messages.append("No URLs available for content extraction.")

            st.session_state.search_complete = True

        except Exception as search_extract_err:
             st.session_state.error = f"Error during Search/Extraction phase: {search_extract_err}"
             st.session_state.status = "Error"
             logging.error(st.session_state.error, exc_info=True)
             st.session_state.log_messages.append(f"[Error] Search/Extraction phase failed: {search_extract_err}")
             # Allow process to potentially continue to report generation, but RAG will likely fail
             st.session_state.search_complete = True # Mark step as 'done' even if failed


    # 4. RAG Analysis
    if not st.session_state.rag_complete:
        st.session_state.log_messages.append("Aggregating content and performing RAG analysis...")
        rag_status = st.info("Preparing content for RAG...")

        # --- Aggregate Content ---
        full_content = ""
        citations_list = []
        content_source_map = {} # url/filename -> text content

        # Process extracted content (now stored under 'combined_sources')
        extracted_items = task.get('extracted_content', {}).get('combined_sources', [])
        for i, item in enumerate(extracted_items):
            url = item.get("url", f"source_{i}")
            title = item.get("title", "Source")
            content = item.get("text") or item.get("raw_content", "")
            if content and len(content) > 50: # Basic filter
                clean_content = content.strip()
                # Truncate very long content sections to keep context manageable
                MAX_CONTENT_LEN = 5000
                if len(clean_content) > MAX_CONTENT_LEN:
                     clean_content = clean_content[:MAX_CONTENT_LEN] + "..."
                     logging.info(f"Truncated content from {url}")

                full_content += f"\n\n=== Context from: {title} ({url}) ===\n{clean_content}\n"
                citations_list.append(f"{title}: {url}")
                content_source_map[url] = clean_content


        # Process additional uploaded files
        if additional_files_content:
            st.session_state.log_messages.append(f"Including content from {len(additional_files_content)} uploaded file(s).")
            for filename, content in additional_files_content.items():
                 if content and len(content) > 50:
                     clean_content = content.strip()
                     # Truncate if needed
                     MAX_CONTENT_LEN = 5000
                     if len(clean_content) > MAX_CONTENT_LEN:
                          clean_content = clean_content[:MAX_CONTENT_LEN] + "..."
                          logging.info(f"Truncated content from file {filename}")
                     full_content += f"\n\n=== Context from Uploaded File: {filename} ===\n{clean_content}\n"
                     citations_list.append(f"Uploaded File: {filename}")
                     content_source_map[filename] = clean_content

        # --- Add Wearable Data ---
        wearable_summary = read_wearable_data(st.session_state.get("wearable_df"))
        if wearable_summary and "Error" not in wearable_summary:
             st.session_state.log_messages.append("Including wearable data summary.")
             full_content += f"\n\n=== Wearable Data Summary ===\n{wearable_summary}\n"
             citations_list.append("Wearable Data Summary (Uploaded)")
             content_source_map["wearable_data"] = wearable_summary
        elif wearable_summary: # If there was an error reading it
             st.warning(f"Could not include wearable data: {wearable_summary}")


        if not full_content.strip():
            st.warning("No content available from web search, user URLs, or files for RAG analysis.")
            st.session_state.log_messages.append("[Warning] No content found for RAG.")
            st.session_state.rag_complete = True
            task['comprehensive_report'] = "Error: No content available to generate the report."
            task['citations'] = []
            rag_status.empty()
            # Skip RAG steps, proceed to final report generation (which will show the error)
        else:
            # --- Chunking ---
            rag_status.info("Chunking aggregated content...")
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter # Ensure langchain is in requirements
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700, # Smaller chunks might be better for embedding models
                    chunk_overlap=70, # Increased overlap
                    length_function=len,
                    is_separator_regex=False,
                )
                chunks = text_splitter.split_text(full_content)
                st.session_state.log_messages.append(f"Content split into {len(chunks)} chunks.")
                if not chunks:
                    raise ValueError("Text chunking resulted in zero chunks.")
            except ImportError:
                 st.error("Langchain library not found. Add 'langchain' to requirements.txt for chunking.")
                 st.session_state.rag_complete = True # Mark as done, but failed
                 task['comprehensive_report'] = "Error: Langchain needed for chunking."
                 task['citations'] = citations_list
                 rag_status.empty()
                 chunks = [] # Ensure chunks is empty list
            except Exception as chunk_err:
                 st.error(f"Error during text chunking: {chunk_err}")
                 st.session_state.rag_complete = True
                 task['comprehensive_report'] = f"Error: Failed to chunk content ({chunk_err})."
                 task['citations'] = citations_list
                 rag_status.empty()
                 chunks = []

            if chunks: # Proceed only if chunking was successful
                # --- Embedding Generation ---
                st.session_state.log_messages.append(f"Generating embeddings (Model: {gemini_embedding_model})...")
                rag_status.info(f"Generating embeddings for {len(chunks)} chunks...")
                try:
                    # Use the Gemini client's embedding capability directly if available
                    # Otherwise, use the google.generativeai library directly
                    import google.generativeai as genai

                    # Gemini API batch embedding limits (check current limits, e.g., 100)
                    batch_size_embed = 100
                    all_embeddings = []
                    num_batches = (len(chunks) + batch_size_embed - 1) // batch_size_embed

                    embed_progress = st.progress(0, text=f"Generating embeddings (Batch 1/{num_batches})...")

                    for i in range(0, len(chunks), batch_size_embed):
                        batch_texts = chunks[i:i+batch_size_embed]
                        # Call batch embedding API
                        embedding_response = genai.embed_content(
                            model=gemini_embedding_model, # e.g., "models/text-embedding-004"
                            content=batch_texts,
                            task_type="RETRIEVAL_DOCUMENT" # Specify task type
                        )
                        # Check response structure
                        if 'embedding' not in embedding_response or not isinstance(embedding_response['embedding'], list):
                             # Handle potential list of embeddings if batch returns that way
                             if isinstance(embedding_response.get('embeddings'), list):
                                 batch_embeddings = [e['values'] for e in embedding_response['embeddings']] # Adjust based on actual API response
                             else:
                                 raise ValueError(f"Unexpected embedding response structure: {embedding_response}")
                        else:
                             # Assuming single embedding for single text, adapt if batch returns list
                             # This part needs verification based on batch call response
                             # Let's assume it returns a list of embeddings in 'embedding' key for batch
                             batch_embeddings = embedding_response['embedding'] # Adjust if structure is different


                        all_embeddings.extend(batch_embeddings)

                        # Update progress
                        progress_val = min(1.0, (i + batch_size_embed) / len(chunks))
                        batch_num = (i // batch_size_embed) + 1
                        embed_progress.progress(progress_val, text=f"Generating embeddings (Batch {batch_num}/{num_batches})...")

                    embed_progress.empty() # Clear progress bar

                    if len(all_embeddings) != len(chunks):
                         st.warning(f"Mismatch between chunks ({len(chunks)}) and generated embeddings ({len(all_embeddings)}). RAG results may be affected.")
                         st.session_state.log_messages.append(f"[Warning] Embedding count mismatch: {len(chunks)} chunks, {len(all_embeddings)} embeddings.")
                         # Attempt to truncate chunks list if embeddings are fewer
                         if len(all_embeddings) < len(chunks):
                              chunks = chunks[:len(all_embeddings)]

                    st.session_state.log_messages.append(f"Generated {len(all_embeddings)} embeddings.")
                    chunk_embeddings = all_embeddings # Use the generated embeddings

                except Exception as e:
                    st.session_state.error = f"Failed to generate embeddings: {e}"
                    st.session_state.status = "Error"
                    logging.error(f"Embedding generation failed: {e}", exc_info=True)
                    st.session_state.log_messages.append(f"[Error] Embedding generation failed: {e}")
                    if 'embed_progress' in locals(): embed_progress.empty()
                    rag_status.empty()
                    # Stop RAG process here
                    st.session_state.rag_complete = True # Mark as done, but failed
                    task['comprehensive_report'] = f"Error: Embedding generation failed ({e})."
                    task['citations'] = citations_list
                    chunk_embeddings = None # Signal failure downstream


                # --- Storing/Indexing & Retrieval ---
                if chunk_embeddings: # Proceed only if embeddings were generated
                    matched_chunks = []
                    num_chunks_to_retrieve = 20 # Retrieve more chunks for LLM context

                    rag_status.info("Indexing and retrieving relevant chunks...")

                    if use_faiss:
                        st.session_state.log_messages.append("Building/Using FAISS index...")
                        # Attempt to build index (will be cached if embeddings haven't changed implicitly)
                        faiss_index = build_faiss_index(chunk_embeddings)
                        if faiss_index:
                            st.session_state.log_messages.append(f"Searching FAISS index for top {num_chunks_to_retrieve} chunks...")
                            # Generate query embedding
                            try:
                                rag_query = f"Patient query: {task['original_query']}\nIdentified issue: {task['analysis'].get('main_subject', 'Not specified')}\nKey symptoms/details: {task['current_query']}"
                                query_embedding_response = genai.embed_content(
                                     model=gemini_embedding_model,
                                     content=rag_query,
                                     task_type="RETRIEVAL_QUERY"
                                 )
                                query_embedding = query_embedding_response['embedding'] # Adjust key if needed
                                query_embedding_np = np.array(query_embedding)
                                st.session_state.log_messages.append("Query embedding generated for FAISS search.")
                                # Search FAISS
                                matched_chunks = search_faiss(faiss_index, query_embedding_np, chunks, k=num_chunks_to_retrieve)
                            except Exception as q_embed_err:
                                 st.error(f"Failed to generate query embedding for FAISS: {q_embed_err}")
                                 st.session_state.log_messages.append(f"[Error] Query embedding failed: {q_embed_err}")
                                 # Fallback to Supabase if configured? Or fail RAG. For now, fail RAG.
                                 matched_chunks = []
                        else:
                            st.warning("FAISS index build failed. Cannot use FAISS for retrieval.")
                            st.session_state.log_messages.append("[Warning] FAISS index build failed.")
                            # Optionally fallback to Supabase here if desired and configured

                    # If not using FAISS or if FAISS failed, try Supabase (if client available)
                    if not matched_chunks and 'supabase' in clients:
                        st.session_state.log_messages.append(f"Using Supabase to find top {num_chunks_to_retrieve} chunks...")
                        try:
                             # Generate query embedding (if not already done for FAISS)
                             if 'query_embedding' not in locals():
                                 rag_query = f"Patient query: {task['original_query']}\nIdentified issue: {task['analysis'].get('main_subject', 'Not specified')}\nKey symptoms/details: {task['current_query']}"
                                 query_embedding_response = genai.embed_content(
                                     model=gemini_embedding_model,
                                     content=rag_query,
                                     task_type="RETRIEVAL_QUERY"
                                 )
                                 query_embedding = query_embedding_response['embedding']
                                 st.session_state.log_messages.append("Query embedding generated for Supabase search.")

                             query_embedding_list = list(query_embedding) # Supabase typically needs lists

                             # --- Store embeddings in Supabase before matching (if not using FAISS primarily) ---
                             # This assumes you want Supabase to be the persistent store
                             # Consider if this should happen *before* the FAISS check if Supabase is primary
                             if not use_faiss: # Only store if Supabase is the intended method
                                 st.session_state.log_messages.append("Storing embeddings in Supabase...")
                                 # Clear previous? Be careful. Maybe use upsert or unique IDs.
                                 # supabase.table("embeddings").delete().eq("source", "streamlit_rag_session").execute() # Example cleanup
                                 data_to_insert = []
                                 for i, chunk in enumerate(chunks):
                                     if i < len(chunk_embeddings):
                                         embedding_vector = list(chunk_embeddings[i]) # Ensure list
                                         data_to_insert.append({
                                             "chunk": chunk, "embedding": embedding_vector,
                                             "source": "streamlit_rag_session", # Identify source/session
                                             # Add metadata like query_id, timestamp if needed
                                         })
                                 # Insert in batches
                                 batch_size_db = 100
                                 supabase_errors = 0
                                 db_progress = st.progress(0, text=f"Storing embeddings in Supabase...")
                                 for i in range(0, len(data_to_insert), batch_size_db):
                                     batch = data_to_insert[i:i+batch_size_db]
                                     try: supabase.table("embeddings").insert(batch).execute()
                                     except Exception as e: supabase_errors += 1; logging.warning(f"Supabase insert batch error: {e}")
                                     db_progress.progress(min(1.0, (i + batch_size_db) / len(data_to_insert)))
                                 db_progress.empty()
                                 if supabase_errors == 0: st.session_state.log_messages.append("Stored embeddings in Supabase.")
                                 else: st.warning(f"Stored embeddings in Supabase with {supabase_errors} batch errors.")


                             # --- Match using Supabase RPC ---
                             st.session_state.log_messages.append("Matching chunks via Supabase RPC...")
                             match_response = supabase.rpc(
                                 "match_chunks", # Assumes RPC function exists
                                 {"query_embedding": query_embedding_list, "match_count": num_chunks_to_retrieve, "match_threshold": 0.7} # Example params
                             ).execute()

                             if match_response.data:
                                 matched_chunks = [row["chunk"] for row in match_response.data]
                                 st.session_state.log_messages.append(f"Retrieved {len(matched_chunks)} chunks from Supabase.")
                             else:
                                 st.warning("Supabase RPC 'match_chunks' returned no matches.")
                                 st.session_state.log_messages.append("Supabase RAG retrieval returned no matches.")

                        except Exception as e:
                            st.error(f"Supabase query embedding or chunk matching failed: {e}")
                            st.session_state.log_messages.append(f"[Error] Supabase RAG retrieval failed: {e}")
                            logging.error(f"Supabase RAG retrieval failed: {e}", exc_info=True)
                            matched_chunks = [] # Ensure empty list on failure
                    elif not matched_chunks:
                         st.warning("No suitable RAG method (FAISS or Supabase) was available or successful.")
                         st.session_state.log_messages.append("[Warning] RAG retrieval could not be performed.")


                    # --- Generate Comprehensive Report using LLM ---
                    rag_status.info("Generating comprehensive report...")
                    st.session_state.log_messages.append("Generating comprehensive diagnostic report via RAG...")
                    aggregated_relevant = "\n\n---\n\n".join(matched_chunks) if matched_chunks else "No specific relevant content chunks found."

                    # Construct the prompt for the LLM
                    prompt = f"""You are an expert medical diagnostic assistant AI. Generate a comprehensive diagnostic report based ONLY on the provided context.

                    **Patient's Initial Query:**
                    {task['original_query']}

                    **Agent's Refined Query (if any):**
                    {task['current_query']}

                    **Agent's Initial Analysis Summary:**
                    {task.get('analysis', 'Not available')}

                    **Relevant Context Extracted (from Web Search, User Files, Wearable Data):**
                    ```context
                    {aggregated_relevant}
                    ```
                    ---
                    **Instructions:**
                    1.  **Synthesize:** Review all provided information.
                    2.  **Address Query:** Directly address the patient's query using insights from the context.
                    3.  **Analyze:** Provide medical analysis based *only* on the relevant context. Discuss potential conditions, symptom explanations, and factors mentioned.
                    4.  **Recommendations:** Based *only* on the context, suggest general, actionable next steps (e.g., consult a professional, tests mentioned in context). **Do NOT invent recommendations or provide specific medical advice beyond the context.** Clearly state if context is insufficient.
                    5.  **Structure:** Use clear headings (e.g., Patient Query, Analysis, Potential Factors, Recommendations, Context Limitations).
                    6.  **Citations:** Mention the types of sources used (e.g., "Based on information from web sources...").
                    7.  **Format:** Use Markdown.
                    8.  **Disclaimer:** Include a disclaimer: "This AI-generated report is based on provided context and is NOT a substitute for professional medical advice. Consult a healthcare provider for any health concerns."

                    **Generate the Comprehensive Diagnostic Report:**
                    """

                    messages = [
                        {"role": "system", "content": "You are an expert medical diagnostic assistant AI generating reports based *only* on provided context. You do not provide external medical advice."},
                        {"role": "user", "content": prompt}
                    ]

                    try:
                        # Use the Gemini client's chat capability
                        # Assuming gemini_client has a chat method compatible with OpenAI's structure
                        response = gemini_client.chat(messages, temperature=0.4) # Use chat method
                        comprehensive_report = response.get("choices", [{}])[0].get("message", {}).get("content", "")

                        if not comprehensive_report.strip():
                             comprehensive_report = "Error: LLM failed to generate the comprehensive report (empty response)."
                             st.error(comprehensive_report)
                             st.session_state.log_messages.append("[Error] LLM report generation returned empty.")
                        else:
                             st.session_state.log_messages.append("Comprehensive report generated successfully.")

                        task['comprehensive_report'] = comprehensive_report
                        task['citations'] = citations_list # Save the list of source descriptions

                    except Exception as e:
                        st.session_state.error = f"Comprehensive report generation failed: {e}"
                        st.session_state.status = "Error"
                        logging.error(f"Comprehensive report generation failed: {e}", exc_info=True)
                        task['comprehensive_report'] = f"Error: Failed to generate report via RAG - {e}"
                        task['citations'] = citations_list
                        st.session_state.log_messages.append(f"[Error] RAG report generation failed: {e}")

                    st.session_state.rag_complete = True # Mark RAG step as done
                    rag_status.empty()


    # 5. Generate Patient-Friendly Summary
    # Run only if RAG completed (even if it failed, to show error) and report step hasn't run yet
    if st.session_state.rag_complete and not st.session_state.report_generated:
        # Check if comprehensive report exists and is not an error message itself
        comp_report_content = task.get('comprehensive_report')
        if comp_report_content and not comp_report_content.startswith("Error:"):
            st.session_state.log_messages.append("Generating patient-friendly summary...")
            summary_status = st.info("Generating patient-friendly summary...")

            patient_prompt = f"""You are a helpful medical assistant AI. Create a simplified, patient-friendly summary of the following comprehensive diagnostic report.

            **Comprehensive Diagnostic Report:**
            ```markdown
            {comp_report_content}
            ```
            ---
            **Instructions:**
            1.  **Simplify:** Explain main findings/analysis in simple, non-medical terms. Avoid jargon.
            2.  **Key Takeaways:** Highlight important points for the patient *based on the report*.
            3.  **Actionable Steps:** Clearly list specific next steps *mentioned in the comprehensive report*.
            4.  **Clarity:** Use short sentences, clear language.
            5.  **Tone:** Empathetic, supportive, neutral.
            6.  **Scope:** Base summary *strictly* on the comprehensive report. Do NOT add external info/advice.
            7.  **Disclaimer:** Reiterate it's informational and not a replacement for professional consultation.
            8.  **Format:** Use Markdown with bullet points.

            **Generate the Patient-Friendly Summary:**
            """
            messages = [
                {"role": "system", "content": "You are an AI assistant creating simple, patient-friendly summaries of medical reports based *only* on the provided text. You are not a medical professional."},
                {"role": "user", "content": patient_prompt}
            ]

            try:
                response = gemini_client.chat(messages, temperature=0.2) # Lower temp
                patient_summary = response.get("choices", [{}])[0].get("message", {}).get("content", "")

                if not patient_summary.strip():
                     patient_summary = "Error: LLM failed to generate the patient-friendly summary (empty response)."
                     st.error(patient_summary)
                     st.session_state.log_messages.append("[Error] Patient summary generation returned empty.")
                else:
                     st.session_state.log_messages.append("Patient-friendly summary generated.")

                task['patient_summary'] = patient_summary

            except Exception as e:
                st.error(f"Patient-friendly summary generation failed: {e}")
                logging.error(f"Patient summary generation failed: {e}", exc_info=True)
                task['patient_summary'] = f"Error: Failed to generate patient summary - {e}"
                st.session_state.log_messages.append(f"[Error] Patient summary generation failed: {e}")
            finally:
                 summary_status.empty()
        else:
             # If comprehensive report had an error, skip patient summary or set error
             task['patient_summary'] = "Not generated due to issues with the comprehensive report."
             st.session_state.log_messages.append("Skipped patient summary due to comprehensive report error.")

        st.session_state.report_generated = True # Mark final report step as done
        st.session_state.status = "Completed" if not st.session_state.error else "Completed with Errors"


    # Return final results from session state task dictionary
    return task.get('comprehensive_report'), task.get('patient_summary'), task.get('citations')


# --- Streamlit App Layout ---

st.set_page_config(page_title="Medical Health Help Assistant", layout="wide", initial_sidebar_state="expanded")

# --- Initialize Session State ---
# Check and initialize state variables if they don't exist
required_state_vars = {
    'log_messages': [], 'status': "Idle", 'error': None, 'current_task': {},
    'analysis_feedback': "", 'survey_df': None, 'wearable_df': None,
    'vectorizer': None, 'clf_model': None, 'analysis_complete': False,
    'search_complete': False, 'rag_complete': False, 'report_generated': False,
    'client_initialization_errors': [], 'missing_secrets': [],
    'uploaded_survey_filename': None, 'uploaded_wearable_filename': None,
    'ref_file_contents': {}
}
for key, default_value in required_state_vars.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# --- Render UI ---

st.title("🩺 Medical Health Help Assistant")
st.caption("AI-powered analysis based on your symptoms and provided context. This is not a substitute for professional medical advice.")

# --- Sidebar for Configuration Info & Options ---
with st.sidebar:
    st.header("Configuration Info")
    st.info(
        """
        **API Keys:** Configured via Streamlit Secrets. Ensure `GEMINI_API_KEY`,
        `TAVILY_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY` are set in the
        Streamlit Cloud app settings.
        """
    )
    # Display any persistent client errors or missing secrets
    if st.session_state.missing_secrets:
         st.error(f"**Missing Secrets:** {', '.join(st.session_state.missing_secrets)}")
    if st.session_state.client_initialization_errors:
         st.error(f"**Client Errors:** {'; '.join(st.session_state.client_initialization_errors)}")

    st.divider()
    st.header("Analysis Options")

    # Search Parameters
    st.subheader("Search Parameters")
    search_depth_options = ["basic", "advanced"]
    search_depth = st.selectbox(
        "Search Depth", search_depth_options,
        index=search_depth_options.index(st.session_state.get("search_depth_select", "advanced")), # Persist selection
        key="search_depth_select"
    )
    search_breadth = st.number_input(
        "Search Breadth (Results per Query)", min_value=3, max_value=20,
        value=st.session_state.get("search_breadth_input", 10), # Persist selection
        step=1, key="search_breadth_input"
    )
    use_faiss = st.checkbox(
        "Use FAISS for RAG (Faster In-Memory Search)",
        value=st.session_state.get("use_faiss_check", True), # Persist selection
        key="use_faiss_check",
        help="If unchecked or fails, will attempt to use Supabase vector search (if configured)."
    )

    st.divider()
    st.header("Optional Context Data")

    # Survey Data for Classifier Training
    st.subheader("Symptom Classifier Data")
    uploaded_survey_file = st.file_uploader(
        "Upload Symptom Survey (CSV)", type=['csv'], key="survey_uploader",
        help="CSV with columns like 'What are the current symptoms...' and 'Medical Health History' to train the initial diagnosis predictor."
    )
    if uploaded_survey_file is not None:
        # Process only if file changes
        if st.session_state.get("uploaded_survey_filename") != uploaded_survey_file.name:
            try:
                df = pd.read_csv(uploaded_survey_file)
                # Basic validation
                required_cols = ['What are the current symptoms or health issues you are facing', 'Medical Health History']
                if all(col in df.columns for col in required_cols):
                    st.session_state.survey_df = df
                    st.session_state.uploaded_survey_filename = uploaded_survey_file.name
                    st.success(f"Loaded survey data for classifier: '{uploaded_survey_file.name}'")
                    # Clear cached model when survey data changes
                    train_symptom_classifier.clear()
                    explain_diagnosis.clear() # Explanation depends on model
                    # Reset model state
                    st.session_state.vectorizer = None
                    st.session_state.clf_model = None
                else:
                    st.error(f"Survey CSV missing required columns. Needed: {required_cols}")
                    st.session_state.survey_df = None # Ensure invalid data isn't used
                    st.session_state.uploaded_survey_filename = None

            except Exception as e:
                st.error(f"Error reading survey CSV: {e}")
                st.session_state.survey_df = None
                st.session_state.uploaded_survey_filename = None
                train_symptom_classifier.clear() # Clear cache on error too
                explain_diagnosis.clear()
                st.session_state.vectorizer = None
                st.session_state.clf_model = None

    elif st.session_state.get("uploaded_survey_filename") is not None:
         # File removed by user
         st.session_state.survey_df = None
         st.session_state.uploaded_survey_filename = None
         st.info("Survey data removed. Diagnosis prediction will use dummy data.")
         train_symptom_classifier.clear()
         explain_diagnosis.clear()
         st.session_state.vectorizer = None
         st.session_state.clf_model = None


    # Wearable Data
    st.subheader("Wearable Data")
    uploaded_wearable_file = st.file_uploader(
        "Upload Wearable Data (CSV)", type=['csv'], key="wearable_uploader",
        help="CSV with columns like 'heart_rate', 'steps', 'sleep_hours', etc."
    )
    if uploaded_wearable_file is not None:
         if st.session_state.get("uploaded_wearable_filename") != uploaded_wearable_file.name:
            try:
                st.session_state.wearable_df = pd.read_csv(uploaded_wearable_file)
                st.session_state.uploaded_wearable_filename = uploaded_wearable_file.name
                st.success(f"Loaded wearable data: '{uploaded_wearable_file.name}'")
                read_wearable_data.clear() # Clear cache for summary
            except Exception as e:
                st.error(f"Error reading wearable CSV: {e}")
                st.session_state.wearable_df = None
                st.session_state.uploaded_wearable_filename = None
                read_wearable_data.clear()
    elif st.session_state.get("uploaded_wearable_filename") is not None:
        # File removed
        st.session_state.wearable_df = None
        st.session_state.uploaded_wearable_filename = None
        st.info("Wearable data removed.")
        read_wearable_data.clear()


    # Additional Reference Files
    st.subheader("Reference Documents")
    uploaded_ref_files = st.file_uploader(
        "Upload Context Files (PDF, DOCX, TXT, CSV)",
        type=['pdf', 'docx', 'txt', 'csv'],
        accept_multiple_files=True,
        key="ref_uploader"
    )
    # Process and store content of uploaded files in session state immediately
    # This avoids re-processing during the main run unless files change
    current_ref_filenames = {f.name for f in uploaded_ref_files} if uploaded_ref_files else set()
    previous_ref_filenames = set(st.session_state.ref_file_contents.keys())

    # If files changed, update the content store
    if current_ref_filenames != previous_ref_filenames:
        st.session_state.ref_file_contents = {} # Reset content store
        if uploaded_ref_files:
            st.info("Processing reference files...") # Temporary message
            for file in uploaded_ref_files:
                with st.spinner(f"Extracting {file.name}..."):
                    content = extract_text_from_file(file)
                    if content:
                        st.session_state.ref_file_contents[file.name] = content
                        logging.info(f"Extracted content from {file.name}")
                    else:
                        logging.warning(f"Extraction failed or empty for {file.name}")
            st.success(f"Processed {len(st.session_state.ref_file_contents)} reference file(s).")
        else:
             st.info("Reference files removed.") # Message when files are cleared


    # Display currently loaded reference files
    if st.session_state.ref_file_contents:
         st.markdown("**Loaded Reference Files:**")
         for fname in st.session_state.ref_file_contents.keys():
             st.caption(f"- {fname}")


    # --- Disclaimer ---
    st.divider()
    st.warning(
         "**Disclaimer:** This application provides AI-generated analysis for informational purposes only. "
         "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
         "Always seek the advice of your physician or other qualified health provider."
     )


# --- Main Area Layout ---

# Input Section
st.header("1. Describe Your Symptoms")
user_query = st.text_area(
    "Enter your current symptoms, duration, severity, and any other relevant details:",
    height=150,
    key="user_query_input",
    placeholder="e.g., Persistent cough for 2 weeks, mild fever, fatigue, especially in the evenings. No shortness of breath."
    )

col1, col2 = st.columns(2)
with col1:
    include_urls_input = st.text_area(
        "Include Specific URLs (Optional, one per line):",
        height=75, key="include_urls_input",
        help="Provide URLs (e.g., from trusted medical sites) to prioritize in the analysis."
        )
with col2:
    omit_urls_input = st.text_area(
        "Omit Specific URLs (Optional, one per line):",
        height=75, key="omit_urls_input",
        help="Provide URLs to exclude from the web search results."
        )

include_urls = [url.strip() for url in include_urls_input.splitlines() if url.strip()]
omit_urls = [url.strip() for url in omit_urls_input.splitlines() if url.strip()]

# Start Button
start_button = st.button("Start Diagnosis Process", type="primary", use_container_width=True, key="start_button")

st.divider()

# --- Processing and Output Area ---
# This section runs when the start button is clicked

# Placeholder for dynamic status updates
status_placeholder = st.empty()
progress_placeholder = st.empty()

if start_button:
    # --- Pre-checks ---
    if not user_query:
        st.warning("⚠️ Please enter your medical symptoms description above.")
        st.stop() # Stop execution if no query

    # Check for client initialization issues *before* running
    _, clients_ok = initialize_clients() # Check if clients are okay
    if not clients_ok:
         error_msg = "Configuration Error!\n\n"
         if st.session_state.get('missing_secrets'):
             error_msg += f"**Missing Secrets:** Please ensure the following are set in the Streamlit Cloud app settings:\n`{', '.join(st.session_state['missing_secrets'])}`\n\n"
         if st.session_state.get('client_initialization_errors'):
              error_msg += "**Initialization Errors:**\n" + "\n".join(f"- {e}" for e in st.session_state['client_initialization_errors'])
         st.error(error_msg)
         st.stop() # Stop execution if config is bad

    # --- Reset state for a new run ---
    st.session_state.status = "Initializing..."
    st.session_state.error = None
    st.session_state.current_task = {} # Clear previous task data
    st.session_state.analysis_complete = False
    st.session_state.search_complete = False
    st.session_state.rag_complete = False
    st.session_state.report_generated = False
    st.session_state.analysis_feedback = "" # Clear previous feedback
    st.session_state.log_messages = ["New diagnosis process started."] # Reset logs

    # --- Initial Processing (Sentiment, Entities, Prediction) ---
    st.header("2. Initial Analysis")
    with st.spinner("Performing initial analysis (Sentiment, Entities, Prediction)..."):
        sentiment_score = analyze_sentiment(user_query)
        medical_entities = extract_medical_entities(user_query)

        # Train or load classifier (uses cached function)
        if st.session_state.vectorizer is None or st.session_state.clf_model is None:
             st.session_state.vectorizer, st.session_state.clf_model = train_symptom_classifier(st.session_state.get("survey_df"))

        predicted_diag, diag_proba = predict_diagnosis(user_query, st.session_state.vectorizer, st.session_state.clf_model)
        explanation = explain_diagnosis(user_query, st.session_state.vectorizer, st.session_state.clf_model)

        # Save to history (ephemeral on Streamlit Cloud unless external DB is used)
        save_query_history(user_query, predicted_diag, sentiment_score, medical_entities)

    # Display Initial Insights
    cols_initial = st.columns(3)
    with cols_initial[0]:
        compound_score = sentiment_score.get('compound', 0.0)
        delta_color = "off"
        if compound_score > 0.1: delta_color = "normal"
        elif compound_score < -0.1: delta_color = "inverse"
        st.metric("Sentiment (Compound)", f"{compound_score:.2f}", delta_color=delta_color,
                  help=f"Score range: -1 (Negative) to +1 (Positive)\nPositive: {sentiment_score.get('pos', 0.0):.2f}, Neutral: {sentiment_score.get('neu', 0.0):.2f}, Negative: {sentiment_score.get('neg', 0.0):.2f}")
    with cols_initial[1]:
        st.markdown("**Predicted Diagnosis:**")
        st.markdown(f"#### {predicted_diag}")
        if diag_proba:
            top_prob = next(iter(diag_proba.values())) # Get first (highest) probability
            st.caption(f"(Confidence: {top_prob:.1%})")
            with st.expander("View Details"):
                st.json({k: f"{v:.3f}" for k, v in diag_proba.items()}) # Show probabilities nicely
        else:
             st.caption("(Probabilities unavailable)")
    with cols_initial[2]:
        st.markdown("**Extracted Medical Entities:**")
        if medical_entities:
             st.markdown(", ".join(f"`{e}`" for e in medical_entities))
        else:
             st.info("None found.")

    with st.expander("View Diagnosis Explanation (SHAP Analysis)", expanded=False):
        st.markdown(explanation)

    st.divider()

    # --- Run the main diagnostic process ---
    st.header("3. In-Depth Analysis & Report Generation")
    status_placeholder.info("Starting in-depth analysis...")
    progress_bar = progress_placeholder.progress(0)

    # Wrap the main process in a try-except block for unexpected errors
    comprehensive_report, patient_summary, citations = None, None, []
    try:
        # Run the main diagnostic function
        comprehensive_report, patient_summary, citations = run_diagnostic_process(
            query=user_query,
            include_urls=include_urls,
            omit_urls=omit_urls,
            additional_files_content=st.session_state.ref_file_contents, # Use processed content
            search_depth=search_depth,
            search_breadth=search_breadth,
            use_faiss=use_faiss,
            feedback=None # Initial run has no feedback yet
        )

        # Update progress and status based on state flags set within the function
        # These updates happen *after* the function call returns in this structure
        final_status_message = "Process completed."
        progress_value = 0
        if st.session_state.analysis_complete: progress_value = 25; final_status_message = "Analysis complete."
        if st.session_state.search_complete: progress_value = 50; final_status_message = "Search & Extraction complete."
        if st.session_state.rag_complete: progress_value = 75; final_status_message = "RAG complete."
        if st.session_state.report_generated: progress_value = 100; final_status_message = "Reports generated."

        if st.session_state.error:
             final_status_message = f"Process finished with errors: {st.session_state.error}"
             status_placeholder.error(final_status_message)
             progress_placeholder.empty() # Remove progress bar on error
        elif st.session_state.status == "Completed":
             status_placeholder.success("Diagnostic process completed successfully!")
             progress_placeholder.empty() # Remove progress bar on success
        else: # Still running or unexpected state
             status_placeholder.info(f"Current Status: {st.session_state.status}")
             progress_placeholder.progress(progress_value)


    except Exception as e:
        st.session_state.error = f"An unexpected error occurred during the main process: {e}"
        st.session_state.status = "Error"
        logging.error(f"Unexpected error in main process: {e}", exc_info=True)
        status_placeholder.error(f"An unexpected critical error occurred: {e}")
        progress_placeholder.empty()
        st.stop() # Stop execution on critical failure


    # --- Feedback Section (Appears ONLY after initial analysis is shown) ---
    # Note: For simplicity in Streamlit's execution model, feedback might be better handled
    # by having the user *edit* the initial query and re-running, rather than a separate feedback loop.
    # The current `run_diagnostic_process` supports feedback, but triggering the re-run
    # requires careful state management. We'll keep it simple for now.
    # If feedback is crucial, consider restructuring the flow.


# --- Display Final Reports (if generated) ---
# This section should always check if the reports exist in session state
if st.session_state.get('report_generated', False) and 'current_task' in st.session_state:
    st.header("4. Diagnostic Reports")
    task = st.session_state.current_task
    comp_report = task.get('comprehensive_report')
    pat_summary = task.get('patient_summary')
    citations = task.get('citations', [])

    if not comp_report and not pat_summary:
         # Handle case where process finished but reports are missing/empty (e.g., due to earlier errors)
         st.warning("Report generation was skipped or failed due to earlier errors.")
         if st.session_state.error:
             st.error(f"Details: {st.session_state.error}")
    else:
        # Generate a simple summary report dynamically
        summary_report = f"# Basic Summary Report\n\n"
        summary_report += f"**Patient Query:** {task.get('original_query', 'N/A')}\n\n"
        if comp_report and not comp_report.startswith("Error:"):
            summary_report += f"**Key Findings (from Comprehensive Report):**\n"
            # Extract first few non-empty lines as a basic summary
            summary_lines = [line for line in comp_report.splitlines() if line.strip() and not line.startswith("#")][:10] # Limit lines
            summary_report += "\n".join(summary_lines) + ("..." if len(comp_report.splitlines()) > 15 else "")
        elif comp_report: # If comp_report is an error message
             summary_report += f"**Comprehensive Report Status:**\n{comp_report}\n"
        else:
             summary_report += "**Comprehensive Report Status:** Not generated.\n"

        summary_report += f"\n\n**Sources Considered:** {len(citations)} sources (Web, User Files, Wearable Data)."
        summary_report += "\n\n---\n*This is a basic summary. Refer to other tabs for details.*"


        tab1, tab2, tab3 = st.tabs(["**Patient-Friendly Summary** ✨", "**Comprehensive Report** 📄", "Basic Summary 📝"])

        with tab1:
            if pat_summary and not pat_summary.startswith("Error:"):
                st.markdown(pat_summary)
                st.download_button(
                    label="Download Patient Summary", data=pat_summary.encode('utf-8'),
                    file_name="patient_summary_report.md", mime="text/markdown", key="dl_patient"
                )
            elif pat_summary: # If it's an error message
                 st.error(pat_summary)
            else:
                 st.info("Patient-friendly summary was not generated.")

        with tab2:
            if comp_report and not comp_report.startswith("Error:"):
                st.markdown(comp_report)
                with st.expander("View Sources/Citations Used"):
                     if citations:
                         for cit in citations: st.markdown(f"- {cit}")
                     else:
                         st.info("No specific sources listed.")
                st.download_button(
                    label="Download Comprehensive Report", data=comp_report.encode('utf-8'),
                    file_name="comprehensive_diagnostic_report.md", mime="text/markdown", key="dl_comp"
                )
            elif comp_report: # If it's an error message
                 st.error(comp_report)
            else:
                 st.info("Comprehensive report was not generated.")

        with tab3:
            st.markdown(summary_report)
            st.download_button(
                label="Download Basic Summary Report", data=summary_report.encode('utf-8'),
                file_name="basic_summary_report.md", mime="text/markdown", key="dl_basic"
            )


# --- Optional: Visualization Section ---
st.divider()
if get_query_history_df().shape[0] > 0: # Only show if history exists
    st.header("Query History & Trends")
    show_viz = st.checkbox("Show Query Trend Visualizations", key="show_viz_check")

    if show_viz:
        with st.spinner("Loading query history and generating visualizations..."):
            history_df = get_query_history_df()
            if not history_df.empty:
                st.dataframe(history_df.tail(10)) # Show last 10 queries
                figures = generate_visualization_figures(history_df) # Uses cached function
                if figures:
                     display_visualizations(figures)
                else:
                     st.info("Could not generate visualizations from the available history.")
            else:
                st.info("No query history found to display or visualize.")
else:
     st.caption("Query history is currently empty.")


# --- Log Display Expander ---
st.divider()
with st.expander("View Processing Logs"):
    log_text = "\n".join(st.session_state.get('log_messages', ["No logs yet."]))
    st.code(log_text, language='log')

# --- Final Status Display ---
# This repeats the status shown after the process finishes, ensuring it's visible at the bottom
if st.session_state.status.startswith("Completed") and not st.session_state.error:
    st.success("Process finished successfully.")
elif st.session_state.error:
    st.error(f"Process finished with an error: {st.session_state.error}")
elif st.session_state.status not in ["Idle", "Initializing..."]: # Show if running or other intermediate state
    st.info(f"Current Status: {st.session_state.status}")

