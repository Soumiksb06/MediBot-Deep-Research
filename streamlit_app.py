# streamlit_app.py
import streamlit as st
import os
import sys
import asyncio
import logging
import csv
import tempfile
from datetime import datetime
from dotenv import load_dotenv # Still useful for local .env file potentially
from rich.console import Console # Can be used for internal logging if needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Import necessary components from the original script ---
# Note: Assuming the original script's structure allows importing these.
# You might need to adjust paths or copy/paste necessary classes/functions
# if they are not structured as importable modules.

# Configure logging (optional for Streamlit, but can be useful)
logging.basicConfig(
    filename="medical_agent_streamlit.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log_capture_string = "" # To display logs in Streamlit if needed

# --- Dependency Installation Helpers ---
def install_nltk_data():
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        st.toast("NLTK VADER lexicon downloaded.", icon="✅")
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")
        logging.error(f"Failed to download NLTK data: {e}")

def install_spacy_model():
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            st.toast("spaCy model 'en_core_web_sm' already available.", icon="✅")
        except OSError:
            st.info("Downloading spaCy model 'en_core_web_sm'...")
            spacy.cli.download("en_core_web_sm")
            st.toast("spaCy model 'en_core_web_sm' downloaded.", icon="✅")
            # Need to reload spacy after download in some environments
            import importlib
            importlib.reload(spacy)
    except ImportError:
        st.error("spaCy not installed. Please install it: pip install spacy")
        logging.error("spaCy not installed.")
    except Exception as e:
        st.error(f"Failed to download or load spaCy model: {e}")
        logging.error(f"Failed to download or load spaCy model: {e}")

# Run dependency downloads at the start
# Consider using @st.cache_resource for these if they don't change
# install_nltk_data()
# install_spacy_model()

# --- Simplified File Extraction ---
# (Adapted from original script, removing console prints)
def extract_text_from_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext == '.pdf':
            try:
                import PyPDF2
            except ImportError:
                st.error("PyPDF2 is required for PDF extraction. Install with 'pip install PyPDF2'.")
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
                st.error("python-docx is required for DOCX extraction. Install with 'pip install python-docx'.")
                return ""
            doc = docx.Document(uploaded_file)
            return "\n".join(para.text for para in doc.paragraphs)
        elif ext == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(uploaded_file)
                return df.to_csv(index=False)
            except ImportError:
                st.error("pandas is required for CSV extraction. Install with 'pip install pandas'.")
                return ""
        elif ext in ['.xls', '.xlsx']:
            try:
                import pandas as pd
                df = pd.read_excel(uploaded_file)
                return df.to_csv(index=False)
            except ImportError:
                st.error("pandas is required for Excel extraction. Install with 'pip install pandas'.")
                return ""
            except Exception as ex: # Handle specific excel errors if needed
                 st.warning(f"Could not read Excel file {uploaded_file.name}. Make sure 'openpyxl' or 'xlrd' is installed: {ex}")
                 return ""
        elif ext == '.txt':
             # Read as text, handling potential encoding issues
            try:
                return uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                 try:
                     return uploaded_file.read().decode("latin-1")
                 except Exception as e:
                     st.warning(f"Could not decode text file {uploaded_file.name}: {e}")
                     return ""
        else:
            st.warning(f"Unsupported file type: {ext} for file {uploaded_file.name}")
            return ""
    except Exception as ex:
        st.error(f"Error extracting file {uploaded_file.name}: {ex}")
        logging.error(f"File extraction error for {uploaded_file.name}: {ex}")
        return ""

# --- Data Science / Analytics & Visualization Functions ---
# (Adapted from original script, using Streamlit elements where needed)

@st.cache_data # Cache the trained model based on survey data content
def train_symptom_classifier(survey_df):
    """Trains a simple symptom-to-diagnosis classifier using the survey data."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if survey_df is None or survey_df.empty:
        st.warning("No survey data provided for training. Using dummy data.")
        # Fallback to dummy data
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

    try:
        # Ensure these column names match exactly your CSV headers
        symptoms_col = 'What are the current symptoms or health issues you are facing'
        history_col = 'Medical Health History'

        if symptoms_col not in survey_df.columns or history_col not in survey_df.columns:
             st.error(f"Survey data missing required columns: '{symptoms_col}' or '{history_col}'")
             # Optionally fall back to dummy data here as well
             return None, None # Indicate failure

        symptoms = survey_df[symptoms_col].fillna("").tolist()
        labels = survey_df[history_col].fillna("None").tolist()
        processed_labels = [label.split(',')[0].strip() if isinstance(label, str) else 'None' for label in labels]

        # Filter out entries with no symptoms or no relevant medical history
        combined_data = [(s, l) for s, l in zip(symptoms, processed_labels) if s and l != 'None']
        if not combined_data:
            st.warning("No valid data for training found in the survey report after filtering.")
            # Optionally fall back to dummy data
            return None, None # Indicate failure

        texts, labels = zip(*combined_data)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=1000) # Increase max_iter if convergence issues
        model.fit(X, labels)
        return vectorizer, model

    except Exception as e:
        st.error(f"An error occurred during classifier training: {e}")
        logging.error(f"Classifier training error: {e}")
        return None, None # Indicate failure

def predict_diagnosis(query, vectorizer, model):
    """Predicts a diagnosis from the patient query."""
    if not query or vectorizer is None or model is None:
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
    """Performs sentiment analysis on the query using NLTK's VADER."""
    if not query:
        return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0} # Default neutral
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        # Ensure VADER lexicon is downloaded (call install_nltk_data() earlier)
        try:
            sia = SentimentIntensityAnalyzer()
            score = sia.polarity_scores(query)
            return score
        except LookupError:
             st.warning("NLTK VADER lexicon not found. Attempting download...")
             install_nltk_data()
             try:
                 sia = SentimentIntensityAnalyzer()
                 score = sia.polarity_scores(query)
                 return score
             except Exception as e:
                  st.error(f"Failed to perform sentiment analysis after download attempt: {e}")
                  return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0} # Default neutral on failure

    except ImportError:
         st.error("NLTK not installed. Please install it: pip install nltk")
         return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0} # Default neutral
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        logging.error(f"Sentiment analysis error: {e}")
        return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0} # Default neutral

def extract_medical_entities(query):
    """Extracts medical entities from the query using spaCy."""
    if not query:
        return []
    try:
        import spacy
        # Ensure model is downloaded (call install_spacy_model() earlier)
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("spaCy model 'en_core_web_sm' not found. Attempting download...")
            install_spacy_model()
            try:
                 # Reload spacy or model may not be found immediately after download
                 import importlib
                 importlib.reload(spacy)
                 nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                  st.error(f"Failed to load spaCy model after download attempt: {e}")
                  return []

        doc = nlp(query)
        # Expanded list - customize as needed
        common_med_terms = {
            "flu", "migraine", "heart attack", "arthritis", "gastroenteritis", "diabetes",
            "hypertension", "asthma", "cancer", "stroke", "anxiety", "depression",
            "fever", "cough", "pain", "diarrhea", "vomiting", "headache", "fatigue",
            "nausea", "chest pain", "shortness of breath", "sore throat", "dizziness",
            "rash", "swelling", "numbness", "allergy", "infection", "inflammation"
        }
        # Consider adding entity types like PROBLEM, TREATMENT, TEST if using a medical NER model
        entities = set()
        for ent in doc.ents:
            # Simple check: lowercase text in common terms or if label indicates medical condition
            # More advanced: Use a biomedical NER model for better accuracy (e.g., scispaCy)
            if ent.text.lower() in common_med_terms or ent.label_ in ["DISEASE", "SYMPTOM", "CONDITION", "PROBLEM"]: # Example labels
                entities.add(ent.text)
        # Also check simple noun chunks for keywords if entities are sparse
        for chunk in doc.noun_chunks:
             if chunk.text.lower() in common_med_terms:
                 entities.add(chunk.text)

        return list(entities)
    except ImportError:
        st.error("spaCy not installed. Please install it: pip install spacy")
        return []
    except Exception as e:
        st.error(f"Error during entity extraction: {e}")
        logging.error(f"Entity extraction error: {e}")
        return []

def get_query_history_df():
    """Loads query history from CSV into a pandas DataFrame."""
    filename = "query_history.csv"
    if os.path.exists(filename):
        try:
            return pd.read_csv(filename)
        except Exception as e:
            st.error(f"Error reading query history file '{filename}': {e}")
            return pd.DataFrame() # Return empty df on error
    else:
        # Return an empty DataFrame with expected columns if file doesn't exist
        return pd.DataFrame(columns=["timestamp", "query", "predicted_diagnosis", "sentiment_compound", "entities"])


def save_query_history(query, diagnosis, sentiment, entities):
    """Saves the query details to a CSV file for trend analysis."""
    filename = "query_history.csv"
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["timestamp", "query", "predicted_diagnosis", "sentiment_compound", "entities"])
            writer.writerow([datetime.now().isoformat(), query, diagnosis, sentiment.get("compound", 0.0), ", ".join(entities)])
        # Update the cached history after saving
        st.cache_data.clear() # Clear cache for get_query_history_df
    except Exception as e:
        st.error(f"Failed to save query history: {e}")
        logging.error(f"Failed to save query history: {e}")


# --- Visualization Functions ---
# (Adapted from original, saving figures to BytesIO for Streamlit display/download)
@st.cache_data # Cache plots based on the history dataframe
def generate_visualization_figures(df_history):
    """Generates visualization figures from query history."""
    figures = {}
    if df_history is None or df_history.empty:
        st.warning("Query history is empty. Cannot generate visualizations.")
        return figures

    try:
        # Convert timestamp if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df_history['timestamp']):
             df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])

        # --- Bar plot for predicted diagnosis frequency ---
        if 'predicted_diagnosis' in df_history.columns:
            diag_counts = df_history['predicted_diagnosis'].value_counts()
            if not diag_counts.empty:
                fig_diag, ax_diag = plt.subplots(figsize=(8, 6))
                sns.barplot(x=diag_counts.index, y=diag_counts.values, palette="viridis", ax=ax_diag)
                ax_diag.set_title("Frequency of Predicted Diagnoses")
                ax_diag.set_xlabel("Diagnosis")
                ax_diag.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                figures['diagnosis_frequency'] = fig_diag
            else:
                 st.info("No diagnosis data to plot.")
        else:
             st.warning("Column 'predicted_diagnosis' not found in history.")


        # --- Line plot for sentiment compound over time ---
        if 'timestamp' in df_history.columns and 'sentiment_compound' in df_history.columns:
            df_sorted = df_history.sort_values("timestamp")
            if not df_sorted.empty:
                fig_sent, ax_sent = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df_sorted, x="timestamp", y="sentiment_compound", marker="o", ax=ax_sent)
                ax_sent.set_title("Sentiment Compound Score Over Time")
                ax_sent.set_xlabel("Timestamp")
                ax_sent.set_ylabel("Sentiment Compound Score")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                figures['sentiment_over_time'] = fig_sent
            else:
                st.info("No sentiment data to plot over time.")
        else:
             st.warning("Columns 'timestamp' or 'sentiment_compound' not found in history.")


        # --- Pie chart for common medical entities ---
        if 'entities' in df_history.columns:
            entity_list = []
            for entities_str in df_history['entities'].dropna():
                if isinstance(entities_str, str) and entities_str.strip():
                    entities = [e.strip() for e in entities_str.split(",") if e.strip()]
                    entity_list.extend(entities)

            if entity_list:
                entity_counts = Counter(entity_list)
                labels = list(entity_counts.keys())
                sizes = list(entity_counts.values())
                fig_ent, ax_ent = plt.subplots(figsize=(8, 8))
                ax_ent.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
                ax_ent.set_title("Distribution of Extracted Medical Entities")
                plt.tight_layout()
                figures['entities_distribution'] = fig_ent
            else:
                 st.info("No entity data to plot.")
        else:
             st.warning("Column 'entities' not found in history.")


    except Exception as e:
        st.error(f"Error generating visualizations: {e}")
        logging.error(f"Error generating visualizations: {e}")
        # Clear figures if error occurs during generation
        figures = {}
    finally:
        plt.close('all') # Close all matplotlib figures to prevent memory leaks

    return figures

def display_visualizations(figures):
    """Displays generated figures in Streamlit and provides download buttons."""
    from io import BytesIO

    if not figures:
        st.info("No visualizations were generated.")
        return

    st.subheader("Query Trend Visualizations")

    for name, fig in figures.items():
        st.pyplot(fig)
        # Provide download button
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label=f"Download {name.replace('_', ' ').title()} Plot",
            data=buf,
            file_name=f"{name}.png",
            mime="image/png"
        )
        st.divider()


# --- FAISS Indexing Functions (Optional) ---
@st.cache_resource # Cache the FAISS index
def build_faiss_index(embeddings):
    """Builds a FAISS index from a list of embeddings."""
    if not embeddings or not isinstance(embeddings[0], (np.ndarray, list)):
         st.warning("Invalid or empty embeddings provided for FAISS.")
         return None
    try:
        import faiss
        # Ensure embeddings are numpy float32 arrays
        embeddings_array = np.array(embeddings).astype('float32')
        dim = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dim)  # Using L2 distance
        index.add(embeddings_array)
        st.success(f"FAISS index built successfully with {len(embeddings)} vectors.")
        return index
    except ImportError:
        st.error("FAISS library not installed. Please install it: pip install faiss-cpu")
        return None
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        logging.error(f"Error building FAISS index: {e}")
        return None

def search_faiss(index, query_embedding, chunks, k):
    """Searches the FAISS index."""
    if index is None or query_embedding is None or not chunks:
        return []
    try:
        import faiss
        import numpy as np
        query_vec = np.array(query_embedding).reshape(1, -1).astype('float32')
        # Adjust k if it's larger than the number of items in the index
        k_actual = min(k, index.ntotal)
        if k_actual == 0:
             st.warning("FAISS index is empty, cannot search.")
             return []
        distances, indices = index.search(query_vec, k_actual)
        # Ensure indices are valid
        valid_indices = [i for i in indices[0] if 0 <= i < len(chunks)]
        matched_chunks = [chunks[i] for i in valid_indices]
        st.info(f"FAISS retrieved {len(matched_chunks)} chunks.")
        return matched_chunks
    except ImportError:
        st.error("FAISS library not installed.")
        return []
    except Exception as e:
        st.error(f"Error searching FAISS index: {e}")
        logging.error(f"Error searching FAISS index: {e}")
        return []


# --- Wearable Data Function ---
@st.cache_data # Cache wearable data summary based on file content
def read_wearable_data(wearable_df):
    """Reads wearable data from DataFrame and returns a summary string."""
    if wearable_df is None or wearable_df.empty:
        return None # Indicate no data
    summary_lines = []
    try:
        if "heart_rate" in wearable_df.columns:
            avg_hr = pd.to_numeric(wearable_df["heart_rate"], errors='coerce').mean()
            if pd.notna(avg_hr):
                summary_lines.append(f"- Average Heart Rate: {avg_hr:.1f} bpm")
        if "steps" in wearable_df.columns:
            total_steps = pd.to_numeric(wearable_df["steps"], errors='coerce').sum()
            if pd.notna(total_steps):
                 summary_lines.append(f"- Total Steps Recorded: {int(total_steps)}")
        # Add more fields as needed (e.g., sleep, spo2)
        if "sleep_hours" in wearable_df.columns:
             avg_sleep = pd.to_numeric(wearable_df["sleep_hours"], errors='coerce').mean()
             if pd.notna(avg_sleep):
                 summary_lines.append(f"- Average Sleep: {avg_sleep:.1f} hours")

        if not summary_lines:
             return "No relevant columns (e.g., 'heart_rate', 'steps') found in wearable data."

        return "\n".join(summary_lines)

    except Exception as e:
        st.error(f"Error processing wearable data: {e}")
        logging.error(f"Error processing wearable data: {e}")
        return "Error processing wearable data."


# --- SHAP Explanation Function ---
@st.cache_data # Cache explanation based on query, vectorizer, model
def explain_diagnosis(query, vectorizer, model):
    """Provides a simple explanation for the predicted diagnosis using SHAP."""
    if not query or vectorizer is None or model is None:
        return "Explanation not available (missing model or query)."
    try:
        import shap
        # Check if the vectorizer is fitted and model has classes
        if not hasattr(vectorizer, 'vocabulary_') or not hasattr(model, 'classes_'):
             return "Explanation not available (model or vectorizer not ready)."

        # Prepare the query data
        X_query = vectorizer.transform([query])#.toarray() # Use sparse matrix directly if explainer supports it

        # Create SHAP explainer
        # Option 1: LinearExplainer (good for linear models like Logistic Regression)
        # Need background data - using feature names might be less ideal than actual data samples
        # If using LinearExplainer, it often expects dense data, hence .toarray() might be needed
        # background_data = vectorizer.transform(vectorizer.get_feature_names_out()).toarray() # Using features might be abstract
        # If possible, use a sample of the *original* training text data transformed
        # sample_texts = ["sample symptom text 1", "sample 2"] # Replace with actual samples if available
        # background_data_transformed = vectorizer.transform(sample_texts).toarray()
        # explainer = shap.LinearExplainer(model, background_data_transformed, feature_perturbation="interventional")

        # Option 2: KernelExplainer (model-agnostic, but can be slower)
        # KernelExplainer often works better with a summary background dataset
        # It needs a function that takes data and returns model probabilities
        def predict_proba_func(data):
            # Check if data is sparse or dense based on what vectorizer.transform returns
            # data_transformed = vectorizer.transform(data) # if data is text
            # Assuming data is already transformed (e.g., sparse matrix)
             # KernelExplainer typically gives dense arrays
            if isinstance(data, np.ndarray):
                data_sparse = data # Needs check if it's already sparse
            else: # Fallback or specific check
                data_sparse = data # Assume correct format

            # Ensure predict_proba gets the correct format
            # If predict_proba_func receives dense data from KernelExplainer, transform it
            # If it receives text, transform it. Adjust based on explainer.
            # Example: If KernelExplainer gives dense numpy array:
            # data_transformed = scipy.sparse.csr_matrix(data) # if needed

            # Let's assume KernelExplainer provides data needing transformation
            # This part is tricky and depends on the exact shap/sklearn versions
            try:
                # Try transforming assuming input is list/array of strings
                if isinstance(data[0], str):
                     data_transformed = vectorizer.transform(data)
                else: # Assume already numerical/dense needing sparse conversion?
                     # This depends heavily on what KernelExplainer actually passes.
                     # It might pass dense arrays corresponding to perturbations.
                     # Let's *assume* it passes dense and we need sparse for LR model:
                     from scipy.sparse import csr_matrix
                     data_transformed = csr_matrix(data)

                return model.predict_proba(data_transformed)

            except Exception as transform_err:
                 # Fallback or error logging
                 st.warning(f"Error in SHAP predict function: {transform_err}")
                 # Return dummy probabilities matching the number of classes
                 num_classes = len(model.classes_) if hasattr(model, 'classes_') else 2
                 return np.ones((len(data), num_classes)) / num_classes


        # Need some background data for KernelExplainer summary
        # Using a few feature names transformed as a proxy - *not ideal*
        # Better: use a sample of actual training texts transformed if possible
        num_background_samples = min(50, len(vectorizer.get_feature_names_out())) # Limit background size
        background_feature_texts = np.random.choice(vectorizer.get_feature_names_out(), num_background_samples, replace=False)
        # Transform these texts to be the background data format expected by predict_proba_func
        # If predict_proba_func expects sparse, transform here. If dense, use .toarray()
        background_data_shap = vectorizer.transform(background_feature_texts) # Keep sparse?


        # Check if background data is suitable
        if background_data_shap.shape[0] == 0:
             return "Explanation not available (could not create background data)."

        # Let KernelExplainer handle sparsity if possible, otherwise provide dense summary
        # summary_background = shap.kmeans(background_data_shap.toarray(), 10) # Example: summarize with kmeans
        # explainer = shap.KernelExplainer(predict_proba_func, summary_background)

        # Simpler approach: try explaining directly on the instance
        # Using KernelExplainer without explicit background summary (might be slow)
        # It needs the data in a format predict_proba_func can handle
        # Let's assume predict_proba_func handles dense array input from explainer
        # We pass the query transformed to dense array for explanation
        # explainer = shap.KernelExplainer(predict_proba_func, np.zeros((1, X_query.shape[1]))) # Dummy background


        # --- Trying LinearExplainer again, assuming it handles sparse input ---
        # LinearExplainer is often preferred for Logistic Regression if compatible
        try:
            # Provide sparse background data directly if supported
            background_sparse = vectorizer.transform(background_feature_texts)
            if background_sparse.shape[0] > 0:
                explainer = shap.LinearExplainer(model, background_sparse, feature_perturbation="interventional")
                shap_values = explainer.shap_values(X_query) # Pass sparse query data
            else:
                 return "Explanation not available (empty background data)."
        except Exception as linear_explainer_err:
             st.warning(f"LinearExplainer failed ({linear_explainer_err}), falling back or skipping SHAP.")
             # Fallback: Maybe try KernelExplainer here or just return unavailable
             return "Explanation generation failed (LinearExplainer error)."


        # --- Process SHAP values ---
        # shap_values can be a list (one per class) or a single array
        if isinstance(shap_values, list):
            # For multi-class, find the SHAP values for the predicted class
            predicted_class_index = np.where(model.classes_ == model.predict(X_query)[0])[0][0]
            shap_values_for_class = shap_values[predicted_class_index]
        else:
            # For binary or single output models
            shap_values_for_class = shap_values[0] # Assuming first row if shape is (1, n_features)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Map SHAP values to feature names (need to handle sparse matrix indexing)
        # Get the indices of the non-zero features in the query vector
        non_zero_indices = X_query.indices
        # Get the corresponding SHAP values and feature names
        shap_values_dense = shap_values_for_class[non_zero_indices]
        feature_names_dense = feature_names[non_zero_indices]

        # Sort features by absolute SHAP value
        abs_shap = np.abs(shap_values_dense)
        # Get indices that would sort abs_shap in descending order
        top_indices_sorted = np.argsort(abs_shap)[::-1]

        # Get top N features
        n_features_to_show = 5
        explanation = "**Top Contributing Features for Prediction:**\n"
        for i in range(min(n_features_to_show, len(top_indices_sorted))):
            idx = top_indices_sorted[i]
            feature = feature_names_dense[idx]
            shap_val = shap_values_dense[idx]
            explanation += f"- `{feature}`: {'+' if shap_val > 0 else ''}{shap_val:.3f} (influence towards predicted class)\n"

        if not top_indices_sorted.size > 0:
             explanation += "\nNo significant features found for this query."

        return explanation

    except ImportError:
        return "Explanation requires SHAP library. Install with 'pip install shap'"
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")
        logging.error(f"SHAP explanation error: {e}")
        # Provide more context if possible
        return f"Explanation not available due to an error: {e}"


# --- Core Agent Logic ---
# (Refactoring MedicalTask and main loop into functions)

# Helper to get API key safely
def get_api_key(key_name, secrets):
    if key_name in secrets:
        return secrets[key_name]
    else:
        # Fallback to environment variable if secrets not found (useful for local testing)
        load_dotenv()
        return os.getenv(key_name)

# Helper to initialize clients (cached)
@st.cache_resource(ttl=3600) # Cache clients for an hour
def initialize_clients():
    clients = {}
    errors = []

    # --- Get API Keys ---
    # Use st.secrets for deployment, fallback to sidebar/env for local
    GEMINI_API_KEY = st.session_state.get("GEMINI_API_KEY")
    TAVILY_API_KEY = st.session_state.get("TAVILY_API_KEY")
    SUPABASE_URL = st.session_state.get("SUPABASE_URL")
    SUPABASE_KEY = st.session_state.get("SUPABASE_KEY")

    # --- Gemini Client ---
    if GEMINI_API_KEY:
        try:
            from subject_analyzer.src.services.gemini_client import GeminiClient
            from subject_analyzer.src.models.analysis_models import AnalysisConfig
            # Model names from secrets or defaults
            GEMINI_MODEL_NAME = st.secrets.get("GEMINI_MODEL_NAME", "gemini-1.5-flash") # Use a known stable model
            analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
            clients['gemini'] = GeminiClient(api_key=GEMINI_API_KEY, config=analysis_config)
            # Test connection? (Optional)
            # clients['gemini'].client.models.list() # Example test
            st.toast("Gemini client initialized.", icon="✅")
        except ImportError:
            errors.append("GeminiClient components not found. Check installation/paths.")
        except Exception as e:
            errors.append(f"Failed to initialize Gemini client: {e}")
    else:
        errors.append("Gemini API Key is missing.")

    # --- Tavily Client ---
    if TAVILY_API_KEY:
        try:
            from web_agent.src.services.web_search import WebSearchService
            from web_agent.src.models.search_models import SearchConfig
            from subject_analyzer.src.services.tavily_client import TavilyClient
            from subject_analyzer.src.services.tavily_extractor import TavilyExtractor

            search_config = SearchConfig()
            search_client = TavilyClient(api_key=TAVILY_API_KEY)
            clients['extractor'] = TavilyExtractor(api_key=TAVILY_API_KEY)
            clients['search_service'] = WebSearchService(search_client, search_config)
            st.toast("Tavily clients initialized.", icon="✅")
        except ImportError:
             errors.append("Tavily/WebAgent components not found. Check installation/paths.")
        except Exception as e:
            errors.append(f"Failed to initialize Tavily clients: {e}")
    else:
        errors.append("Tavily API Key is missing.")

    # --- Supabase Client ---
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client, Client
            clients['supabase'] = create_client(SUPABASE_URL, SUPABASE_KEY)
            # Test connection (e.g., list tables)
            # clients['supabase'].table("embeddings").select("id", head=True).execute() # Example test
            st.toast("Supabase client initialized.", icon="✅")
        except ImportError:
            errors.append("Supabase library not found. Install with 'pip install supabase'")
        except Exception as e:
            errors.append(f"Failed to initialize Supabase client: {e}")
    else:
        errors.append("Supabase URL or Key is missing.")

    # --- Subject Analyzer ---
    if 'gemini' in clients:
         try:
            from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer
            # Assuming AnalysisConfig was created during Gemini client setup
            # If not, create it here
            if 'gemini' in clients and hasattr(clients['gemini'], 'config'):
                analysis_config = clients['gemini'].config
            else: # Fallback config if needed
                 from subject_analyzer.src.models.analysis_models import AnalysisConfig
                 GEMINI_MODEL_NAME = st.secrets.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")
                 analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)

            clients['subject_analyzer'] = SubjectAnalyzer(llm_client=clients['gemini'], config=analysis_config)
            st.toast("Subject Analyzer initialized.", icon="✅")
         except ImportError:
             errors.append("SubjectAnalyzer components not found. Check installation/paths.")
         except Exception as e:
             errors.append(f"Failed to initialize Subject Analyzer: {e}")
    else:
         errors.append("Subject Analyzer requires Gemini client.")


    # Display errors if any
    for error in errors:
        st.error(error)
        logging.error(error)

    # Return only successfully initialized clients
    return {k: v for k, v in clients.items() if k in clients}, len(errors) == 0


# Function to run the main diagnostic process
def run_diagnostic_process(query, include_urls, omit_urls, additional_files_content, search_depth, search_breadth, use_faiss, feedback=None):
    """
    Performs the core diagnostic steps: analysis, search, extraction, RAG, report generation.
    Returns report strings and citations.
    """
    st.session_state.log_messages = ["Starting diagnostic process..."]
    st.session_state.status = "Running..."
    st.session_state.error = None
    st.session_state.analysis_complete = False
    st.session_state.search_complete = False
    st.session_state.rag_complete = False
    st.session_state.report_generated = False


    # 1. Initialize clients (or get from cache)
    clients, clients_ok = initialize_clients()
    if not clients_ok:
        st.session_state.error = "Failed to initialize required API clients. Check sidebar configuration and API keys."
        st.session_state.status = "Error"
        logging.error(st.session_state.error)
        return None, None, None, None # Reports, citations

    # Make sure essential clients are present
    required_clients = ['gemini', 'search_service', 'extractor', 'subject_analyzer', 'supabase']
    if not all(client in clients for client in required_clients):
         missing = [client for client in required_clients if client not in clients]
         st.session_state.error = f"Missing required clients: {', '.join(missing)}. Cannot proceed."
         st.session_state.status = "Error"
         logging.error(st.session_state.error)
         return None, None, None, None

    gemini_client = clients['gemini']
    search_service = clients['search_service']
    extractor = clients['extractor']
    subject_analyzer = clients['subject_analyzer']
    supabase = clients['supabase']

    st.session_state.log_messages.append("API clients initialized.")

    # Get current date for analysis context
    current_date = datetime.today().strftime("%Y-%m-%d")

    # Store results in session state to avoid recomputing on reruns if not necessary
    if 'current_task' not in st.session_state:
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
        # Reset downstream steps if re-analyzing
        st.session_state.analysis_complete = False
        st.session_state.search_complete = False
        st.session_state.rag_complete = False
        st.session_state.report_generated = False


    # 2. Analyze Subject (unless already done for this query/feedback cycle)
    if not st.session_state.analysis_complete:
        st.session_state.log_messages.append("Analyzing patient query...")
        try:
            # The subject_analyzer uses the GeminiClient
            analysis_query = f"{task['current_query']} (Analysis requested on {current_date})"
            task['analysis'] = subject_analyzer.analyze(analysis_query)
            st.session_state.log_messages.append("Subject analysis successful.")
            st.session_state.analysis_complete = True # Mark analysis as done for this cycle
        except Exception as e:
            st.session_state.error = f"Subject analysis failed: {e}"
            st.session_state.status = "Error"
            logging.error(f"Subject analysis failed: {e}")
            st.session_state.log_messages.append(f"[Error] Subject analysis failed: {e}")
            return None, None, None, None # Stop processing

    # Display analysis results
    st.subheader("Agent's Understanding")
    st.markdown(f"**Patient Query:** {task['original_query']}")
    if task['feedback_history']:
        st.markdown(f"**Applied Feedback:** {task['feedback_history'][-1]['feedback']}") # Show last feedback
        st.markdown(f"**Refined Query for Analysis:** {task['current_query']}")

    analysis_display = task.get('analysis', {})
    if analysis_display:
        st.markdown(f"**Identified Medical Issue:** `{analysis_display.get('main_subject', 'Unknown Issue')}`")
        temporal = analysis_display.get("temporal_context", {})
        if temporal:
            st.write("**Temporal Context:**")
            st.json(temporal, expanded=False)
        else:
            st.write("**Temporal Context:** Not specified.")
        needs = analysis_display.get("What_needs_to_be_researched", [])
        st.write("**Key Aspects to Investigate:**")
        if needs:
            for item in needs: st.markdown(f"- {item}")
        else:
            st.markdown("- None identified.")
    else:
        st.warning("Analysis results are not available.")


    # Allow feedback ONLY after analysis is shown
    # Feedback mechanism handled by the main app layout now


    # 3. Search and Extract (unless already done)
    if not st.session_state.search_complete:
        st.session_state.log_messages.append("Starting web search and content extraction...")
        task['search_results'] = {}
        task['extracted_content'] = {}

        if not include_urls: # Perform web search
            topics_to_search = [task['analysis'].get("main_subject", task['current_query'])]
            topics_to_search += task['analysis'].get("What_needs_to_be_researched", [])
            topics_to_search = list(set(filter(None, topics_to_search))) # Unique, non-empty topics

            if not topics_to_search:
                st.warning("No specific topics identified for web search based on analysis. Searching based on main query.")
                topics_to_search = [task['current_query']]

            st.session_state.log_messages.append(f"Searching for topics: {', '.join(topics_to_search)}")

            all_urls_found = []
            for topic in topics_to_search:
                st.session_state.log_messages.append(f"Searching web for: '{topic}' (Depth: {search_depth}, Breadth: {search_breadth})")
                try:
                    response = search_service.search_subject(
                        topic, "medical diagnosis information", # More specific domain
                        search_depth=search_depth,
                        results=search_breadth # Use the parameter
                    )
                    results = response.get("results", [])
                    # Filter results by omit_urls and ensure URL exists
                    filtered_results = [
                        res for res in results
                        if res.get("url") and not any(omit.lower() in res.get("url").lower() for omit in omit_urls)
                    ]
                    task['search_results'][topic] = filtered_results
                    all_urls_found.extend([res.get("url") for res in filtered_results])
                    st.session_state.log_messages.append(f"Found {len(filtered_results)} relevant results for '{topic}'.")
                except Exception as e:
                    st.warning(f"Web search failed for topic '{topic}': {e}")
                    task['search_results'][topic] = []
                    logging.warning(f"Web search failed for topic '{topic}': {e}")

            # Extract content from all found URLs together
            unique_urls = list(set(all_urls_found))
            if unique_urls:
                st.session_state.log_messages.append(f"Extracting content from {len(unique_urls)} unique URLs...")
                try:
                    # Limit number of URLs sent to extractor if necessary (e.g., 20)
                    urls_to_extract = unique_urls[:20]
                    if len(unique_urls) > 20:
                         st.warning(f"Limiting content extraction to the first 20 URLs found ({len(unique_urls)} total).")
                         st.session_state.log_messages.append(f"Limiting extraction to {len(urls_to_extract)} URLs.")

                    extraction_response = extractor.extract(
                        urls=urls_to_extract,
                        extract_depth="advanced", # Keep advanced for more content
                        include_images=False
                    )
                    extracted_items = extraction_response.get("results", [])
                    # Assign extracted content back to the corresponding topic (best effort)
                    # Or store globally if mapping back is too complex
                    task['extracted_content']['web_search'] = extracted_items # Store all extracted
                    st.session_state.log_messages.append(f"Extracted content from {len(extracted_items)} URLs.")
                    failed_extractions = sum(1 for item in extracted_items if item.get("error"))
                    if failed_extractions:
                        st.warning(f"Failed to extract content from {failed_extractions} URLs.")
                        st.session_state.log_messages.append(f"[Warning] Failed extraction for {failed_extractions} URLs.")
                except Exception as e:
                    st.error(f"Content extraction failed: {e}")
                    st.session_state.log_messages.append(f"[Error] Content extraction failed: {e}")
                    logging.error(f"Content extraction failed: {e}")
            else:
                st.info("No valid URLs found from web search to extract content.")
                st.session_state.log_messages.append("No URLs found from web search.")

        else: # Use user-provided URLs
            st.session_state.log_messages.append(f"Using {len(include_urls)} user-provided URLs.")
            filtered_urls = [url for url in include_urls if not any(omit.lower() in url.lower() for omit in omit_urls)]
            st.session_state.log_messages.append(f"Extracting content from {len(filtered_urls)} filtered URLs.")
            if filtered_urls:
                try:
                    extraction_response = extractor.extract(
                        urls=filtered_urls,
                        extract_depth="advanced",
                        include_images=False
                    )
                    extracted_items = extraction_response.get("results", [])
                    task['extracted_content']['user_provided'] = extracted_items
                    task['search_results']['user_provided'] = [{"title": item.get("title", "User Provided"), "url": item.get("url"), "score": "N/A"} for item in extracted_items if item.get("url")]
                    st.session_state.log_messages.append(f"Extracted content from {len(extracted_items)} user URLs.")
                    failed_extractions = sum(1 for item in extracted_items if item.get("error"))
                    if failed_extractions:
                        st.warning(f"Failed to extract content from {failed_extractions} user URLs.")
                        st.session_state.log_messages.append(f"[Warning] Failed extraction for {failed_extractions} user URLs.")
                except Exception as e:
                    st.error(f"Extraction failed for user provided URLs: {e}")
                    st.session_state.log_messages.append(f"[Error] Extraction failed for user URLs: {e}")
                    logging.error(f"Extraction failed for user provided URLs: {e}")
            else:
                st.info("No user-provided URLs remaining after filtering.")
                st.session_state.log_messages.append("No user URLs left after filtering.")

        st.session_state.search_complete = True # Mark search/extraction as done

    # Display Search/Extraction Results (Optional, could be in expander)
    with st.expander("View Search and Extraction Details"):
        st.write("**Search Results:**")
        if task.get('search_results'):
            for topic, results in task['search_results'].items():
                st.markdown(f"**Topic:** `{topic}`")
                if results:
                    for res in results[:5]: # Show top 5 per topic
                        st.markdown(f"- [{res.get('title', 'No Title')}]({res.get('url')}) (Score: {res.get('score', 'N/A')})")
                else:
                    st.markdown("- *No results found.*")
        else:
            st.info("No search results recorded.")

        st.write("**Extracted Content Summary:**")
        if task.get('extracted_content'):
             content_count = sum(len(v) for v in task['extracted_content'].values())
             st.info(f"Found extracted content items from {content_count} sources (web/user). Details used in RAG.")
             # Optionally display snippets here if needed
        else:
            st.info("No content extracted.")


    # 4. RAG Analysis (unless already done)
    if not st.session_state.rag_complete:
        st.session_state.log_messages.append("Aggregating content and performing RAG analysis...")

        # --- Aggregate Content ---
        full_content = ""
        citations_list = []
        content_source_map = {} # url -> text content

        # Process extracted web content
        web_items = task.get('extracted_content', {}).get('web_search', [])
        for i, item in enumerate(web_items):
            url = item.get("url", f"web_source_{i}")
            title = item.get("title", "Web Source")
            content = item.get("text") or item.get("raw_content", "")
            if content and len(content) > 50: # Basic filter for meaningful content
                clean_content = content.strip()
                full_content += f"\n\n=== Content from {title} ({url}) ===\n{clean_content}\n"
                citations_list.append(f"{title}: {url}")
                content_source_map[url] = clean_content

        # Process user-provided URL content
        user_items = task.get('extracted_content', {}).get('user_provided', [])
        for i, item in enumerate(user_items):
             url = item.get("url", f"user_source_{i}")
             title = item.get("title", "User Provided Source")
             content = item.get("text") or item.get("raw_content", "")
             if content and len(content) > 50:
                 clean_content = content.strip()
                 full_content += f"\n\n=== Content from User URL: {title} ({url}) ===\n{clean_content}\n"
                 citations_list.append(f"User Provided - {title}: {url}")
                 content_source_map[url] = clean_content


        # Process additional uploaded files
        if additional_files_content:
            st.session_state.log_messages.append(f"Including content from {len(additional_files_content)} uploaded file(s).")
            for filename, content in additional_files_content.items():
                 if content and len(content) > 50:
                     full_content += f"\n\n=== Content from Uploaded File: {filename} ===\n{content}\n"
                     citations_list.append(f"Uploaded File: {filename}")
                     content_source_map[filename] = content # Use filename as key

        # --- Add Wearable Data ---
        wearable_summary = read_wearable_data(st.session_state.get("wearable_df"))
        if wearable_summary:
             st.session_state.log_messages.append("Including wearable data summary.")
             full_content += f"\n\n=== Wearable Data Summary ===\n{wearable_summary}\n"
             citations_list.append("Wearable Data Summary (provided by user)")
             content_source_map["wearable_data"] = wearable_summary


        if not full_content:
            st.warning("No content available from web search, user URLs, or files for RAG analysis.")
            st.session_state.log_messages.append("[Warning] No content found for RAG.")
            st.session_state.rag_complete = True # Mark as 'complete' (but empty)
            task['comprehensive_report'] = "Error: No content available to generate the report."
            task['citations'] = []
            # Skip directly to patient summary if possible, or just show error
            st.session_state.report_generated = True # Mark report gen as 'complete' (with error)
            return task.get('comprehensive_report'), task.get('patient_summary'), task.get('citations')


        # --- Chunking ---
        def chunk_text(text, chunk_size=700, overlap=50): # Smaller chunk, some overlap
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len,
                is_separator_regex=False,
            )
            return text_splitter.split_text(text)

        chunks = chunk_text(full_content)
        st.session_state.log_messages.append(f"Content split into {len(chunks)} chunks.")
        if not chunks:
             st.warning("Text chunking resulted in zero chunks.")
             st.session_state.log_messages.append("[Warning] Zero chunks after splitting.")
             # Handle similarly to no content
             st.session_state.rag_complete = True
             task['comprehensive_report'] = "Error: Failed to chunk content for analysis."
             task['citations'] = []
             st.session_state.report_generated = True
             return task.get('comprehensive_report'), task.get('patient_summary'), task.get('citations')


        # --- Embedding Generation ---
        st.session_state.log_messages.append("Generating embeddings for content chunks...")
        try:
            # Use the Gemini client's embedding capability
            # Model selection (use secrets or default)
            GOOGLE_EMBEDDING_MODEL = st.secrets.get("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")

            # Gemini API might have batch embedding limits (e.g., 100 per call)
            batch_size_embed = 100
            all_embeddings = []
            num_batches = (len(chunks) + batch_size_embed - 1) // batch_size_embed

            embed_progress = st.progress(0, text=f"Generating embeddings (Batch 1/{num_batches})...")

            for i in range(0, len(chunks), batch_size_embed):
                batch_texts = chunks[i:i+batch_size_embed]
                # Prepare content parts for the API
                content_parts = [{"text": text} for text in batch_texts]

                # Call Gemini embedding API
                embedding_response = gemini_client.client.models.embed_content(
                    model=GOOGLE_EMBEDDING_MODEL,
                    requests=[{'content': {'parts': [part]}} for part in content_parts], # Structure for embed_content
                    task_type="RETRIEVAL_DOCUMENT" # Specify task type for better embeddings
                )

                batch_embeddings = [e.values for e in embedding_response.embeddings]
                all_embeddings.extend(batch_embeddings)

                # Update progress
                progress_val = min(1.0, (i + batch_size_embed) / len(chunks))
                batch_num = (i // batch_size_embed) + 1
                embed_progress.progress(progress_val, text=f"Generating embeddings (Batch {batch_num}/{num_batches})...")


            if len(all_embeddings) != len(chunks):
                 st.warning(f"Mismatch between chunks ({len(chunks)}) and generated embeddings ({len(all_embeddings)}). Proceeding, but results may be affected.")
                 st.session_state.log_messages.append(f"[Warning] Embedding count mismatch: {len(chunks)} chunks, {len(all_embeddings)} embeddings.")
                 # Attempt to truncate chunks list if embeddings are fewer
                 if len(all_embeddings) < len(chunks):
                      chunks = chunks[:len(all_embeddings)]


            st.session_state.log_messages.append(f"Generated {len(all_embeddings)} embeddings.")
            embed_progress.empty() # Clear progress bar

        except Exception as e:
            st.session_state.error = f"Failed to generate embeddings: {e}"
            st.session_state.status = "Error"
            logging.error(f"Embedding generation failed: {e}")
            st.session_state.log_messages.append(f"[Error] Embedding generation failed: {e}")
            embed_progress.empty()
            return None, None, None, None


        # --- Storing/Indexing Embeddings (Supabase / FAISS) ---
        chunk_embeddings = all_embeddings # Use the generated embeddings

        if use_faiss:
            st.session_state.log_messages.append("Building FAISS index...")
            faiss_index = build_faiss_index(chunk_embeddings)
            if faiss_index is None:
                 st.warning("Could not build FAISS index. Retrieval might be slower or use Supabase fallback if implemented.")
                 st.session_state.log_messages.append("[Warning] FAISS index build failed.")
                 # Optionally implement Supabase storage here as fallback
                 st.session_state['faiss_index'] = None # Store None to indicate failure
            else:
                 st.session_state['faiss_index'] = faiss_index # Store index in session state
                 st.session_state.log_messages.append("FAISS index built.")
        else:
            # Store embeddings in Supabase (optional, can rely on FAISS in-memory)
            st.session_state.log_messages.append("Storing embeddings in Supabase (if configured)...")
            # Clear previous embeddings? (Be cautious with this)
            # try:
            #     supabase.table("embeddings").delete().eq("source", "streamlit_rag").execute()
            # except Exception as e:
            #     st.warning(f"Could not clear previous Supabase embeddings: {e}")

            data_to_insert = []
            for i, chunk in enumerate(chunks):
                if i < len(chunk_embeddings): # Ensure embedding exists
                    embedding_vector = chunk_embeddings[i]
                    # Ensure it's a list for Supabase
                    if isinstance(embedding_vector, np.ndarray):
                        embedding_vector = embedding_vector.tolist()
                    elif not isinstance(embedding_vector, list):
                         try: embedding_vector = list(embedding_vector)
                         except: continue # Skip if conversion fails

                    data_to_insert.append({
                        "chunk": chunk,
                        "embedding": embedding_vector,
                        "source": "streamlit_rag" # Identify source
                    })

            batch_size_db = 100
            supabase_errors = 0
            db_progress = st.progress(0, text=f"Storing embeddings in Supabase (Batch 1/{(len(data_to_insert) + batch_size_db - 1) // batch_size_db})...")

            for i in range(0, len(data_to_insert), batch_size_db):
                 batch = data_to_insert[i:i+batch_size_db]
                 try:
                     # Consider using upsert if chunks might be re-inserted
                     supabase.table("embeddings").insert(batch).execute()
                 except Exception as e:
                     st.warning(f"Error inserting Supabase batch {i // batch_size_db + 1}: {e}")
                     logging.warning(f"Supabase insert error: {e}")
                     supabase_errors += 1
                 # Update progress
                 progress_val = min(1.0, (i + batch_size_db) / len(data_to_insert))
                 batch_num = (i // batch_size_db) + 1
                 db_progress.progress(progress_val, text=f"Storing embeddings in Supabase (Batch {batch_num}/{(len(data_to_insert) + batch_size_db - 1) // batch_size_db})...")


            if supabase_errors == 0:
                st.session_state.log_messages.append(f"Stored {len(data_to_insert)} embeddings in Supabase.")
            else:
                 st.warning(f"Finished storing embeddings in Supabase with {supabase_errors} batch errors.")
                 st.session_state.log_messages.append(f"[Warning] Supabase storage completed with {supabase_errors} errors.")
            db_progress.empty()


        # --- Query Embedding ---
        st.session_state.log_messages.append("Generating embedding for the query...")
        # Use a more focused query for retrieval - combine original query and analysis subject
        rag_query = f"Patient query: {task['original_query']}\nIdentified issue: {task['analysis'].get('main_subject', 'Not specified')}\nKey symptoms/details: {task['current_query']}" # More context for retrieval

        try:
            # Use RETRIEVAL_QUERY task type
             query_embedding_response = gemini_client.client.models.embed_content(
                 model=GOOGLE_EMBEDDING_MODEL,
                 contents=[{'parts': [{'text': rag_query}]}], # Correct structure
                 task_type="RETRIEVAL_QUERY"
             )
             query_embedding = query_embedding_response.embeddings[0].values
             st.session_state.log_messages.append("Query embedding generated.")
        except Exception as e:
             st.session_state.error = f"Failed to get embedding for RAG query: {e}"
             st.session_state.status = "Error"
             logging.error(st.session_state.error)
             st.session_state.log_messages.append(f"[Error] Query embedding failed: {e}")
             return None, None, None, None

        # Ensure query embedding is a list for Supabase if needed, numpy for FAISS
        query_embedding_list = list(query_embedding) if not isinstance(query_embedding, list) else query_embedding
        query_embedding_np = np.array(query_embedding) if not isinstance(query_embedding, np.ndarray) else query_embedding


        # --- Retrieve Relevant Chunks ---
        st.session_state.log_messages.append("Retrieving relevant content chunks...")
        matched_chunks = []
        num_chunks_to_retrieve = 20 # Retrieve more chunks for LLM context

        if use_faiss and 'faiss_index' in st.session_state and st.session_state['faiss_index'] is not None:
            st.session_state.log_messages.append(f"Searching FAISS index for top {num_chunks_to_retrieve} chunks...")
            matched_chunks = search_faiss(st.session_state['faiss_index'], query_embedding_np, chunks, k=num_chunks_to_retrieve)
        else:
            # Fallback to Supabase RPC
            if not use_faiss:
                st.session_state.log_messages.append(f"Using Supabase to find top {num_chunks_to_retrieve} chunks...")
            else:
                st.session_state.log_messages.append(f"FAISS index not available, using Supabase to find top {num_chunks_to_retrieve} chunks...")

            try:
                match_response = supabase.rpc(
                    "match_chunks", # Assumes you have this RPC function in Supabase
                    {"query_embedding": query_embedding_list, "match_count": num_chunks_to_retrieve, "match_threshold": 0.7} # Add threshold if function supports it
                ).execute()

                if match_response.data:
                    matched_chunks = [row["chunk"] for row in match_response.data]
                    st.session_state.log_messages.append(f"Retrieved {len(matched_chunks)} chunks from Supabase.")
                else:
                    st.warning("Supabase RPC 'match_chunks' returned no matches.")
                    st.session_state.log_messages.append("Supabase RAG retrieval returned no matches.")
            except Exception as e:
                st.error(f"Supabase chunk matching failed: {e}. Trying to continue without RAG context.")
                st.session_state.log_messages.append(f"[Error] Supabase RAG retrieval failed: {e}")
                # Proceed without matched chunks, LLM will rely only on prompt
                matched_chunks = []


        if not matched_chunks:
            st.warning("No relevant content chunks found via RAG. Report generation might be less detailed.")
            st.session_state.log_messages.append("[Warning] No relevant chunks found via RAG.")


        # --- Generate Comprehensive Report using LLM ---
        st.session_state.log_messages.append("Generating comprehensive diagnostic report via RAG...")
        aggregated_relevant = "\n\n---\n\n".join(matched_chunks) if matched_chunks else "No specific relevant content found."

        # Construct a more robust prompt for the LLM
        prompt = f"""You are an expert medical diagnostic assistant AI. Your task is to generate a comprehensive diagnostic report based ONLY on the provided context.

        **Patient's Initial Query:**
        {task['original_query']}

        **Agent's Refined Query (based on feedback, if any):**
        {task['current_query']}

        **Agent's Initial Analysis:**
        {task.get('analysis', 'Not available')}

        **Relevant Context Extracted from Provided Sources (Web Search, User Files, Wearable Data):**
        {aggregated_relevant}
        ---
        **Instructions:**
        1.  **Synthesize:** Carefully review all the provided information.
        2.  **Address Query:** Directly address the patient's initial query and refined query.
        3.  **Analyze:** Provide a detailed medical analysis based *only* on the relevant context above. Discuss potential conditions, explanations for symptoms, and relevant factors mentioned in the context.
        4.  **Recommendations:** Based *only* on the context, suggest general, actionable next steps. This might include advising consultation with a specific type of healthcare professional, mentioning potential diagnostic tests discussed in the context, or lifestyle adjustments referenced. **Do NOT invent recommendations or provide specific medical advice beyond what the context supports.** State clearly if the context is insufficient for certain recommendations.
        5.  **Structure:** Organize the report clearly with headings (e.g., Patient Query, Analysis, Potential Factors, Recommendations, Context Limitations).
        6.  **Citations:** While specific chunk citations are hard, mention the types of sources used (e.g., "Based on information from web sources and uploaded files...").
        7.  **Format:** Use Markdown for clear formatting.
        8.  **Disclaimer:** Include a disclaimer stating this is AI-generated information based on provided context and is NOT a substitute for professional medical advice.

        **Generate the Comprehensive Diagnostic Report:**
        """

        messages = [
            {"role": "system", "content": "You are an expert medical diagnostic assistant AI generating reports based *only* on provided context. You do not provide external medical advice."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Use the Gemini client for report generation
            response = gemini_client.chat(messages, temperature=0.4) # Slightly higher temp for generation
            comprehensive_report = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not comprehensive_report:
                 comprehensive_report = "Error: LLM failed to generate the comprehensive report."
                 st.error(comprehensive_report)
                 st.session_state.log_messages.append("[Error] LLM report generation returned empty.")
            else:
                 st.session_state.log_messages.append("Comprehensive report generated successfully.")

            task['comprehensive_report'] = comprehensive_report
            task['citations'] = citations_list # Save the list of source descriptions

            # Optional: Clean up Supabase embeddings after use? (Consider if needed)
            # try:
            #     supabase.table("embeddings").delete().eq("source", "streamlit_rag").execute()
            #     st.session_state.log_messages.append("Cleaned Supabase embeddings.")
            # except Exception as e:
            #     st.warning(f"Could not clean Supabase embeddings: {e}")

        except Exception as e:
            st.session_state.error = f"Comprehensive report generation failed: {e}"
            st.session_state.status = "Error"
            logging.error(st.session_state.error)
            task['comprehensive_report'] = f"Error: Failed to generate report via RAG - {e}"
            task['citations'] = citations_list # Still save citations attempted
            st.session_state.log_messages.append(f"[Error] RAG report generation failed: {e}")

        st.session_state.rag_complete = True # Mark RAG step as done


    # 5. Generate Patient-Friendly Summary (unless already done)
    if st.session_state.rag_complete and not st.session_state.report_generated and task.get('comprehensive_report') and not task['comprehensive_report'].startswith("Error:"):
        st.session_state.log_messages.append("Generating patient-friendly summary...")

        # Prompt for patient summary - focusing on clarity and actionability based *only* on the main report
        patient_prompt = f"""You are a helpful medical assistant AI. Your task is to create a simplified, patient-friendly summary of the following comprehensive diagnostic report.

        **Comprehensive Diagnostic Report:**
        ```markdown
        {task['comprehensive_report']}
        ```
        ---
        **Instructions:**
        1.  **Simplify:** Explain the main findings and analysis in simple, non-medical terms. Avoid jargon.
        2.  **Key Takeaways:** Highlight the most important points the patient should understand about their situation *based on the report*.
        3.  **Actionable Steps:** Clearly list the specific, actionable next steps *mentioned in the comprehensive report*. This might include seeing a doctor, specific tests mentioned, or lifestyle changes discussed.
        4.  **Clarity:** Use short sentences and clear language.
        5.  **Tone:** Be empathetic and supportive, but maintain neutrality.
        6.  **Scope:** Base the summary *strictly* on the content of the comprehensive report provided above. Do NOT add external information or advice.
        7.  **Disclaimer:** Reiterate that this summary is for informational purposes and does not replace consultation with a healthcare professional.
        8.  **Format:** Use Markdown with bullet points for clarity.

        **Generate the Patient-Friendly Summary:**
        """
        messages = [
            {"role": "system", "content": "You are an AI assistant creating simple, patient-friendly summaries of medical reports based *only* on the provided text. You are not a medical professional."},
            {"role": "user", "content": patient_prompt}
        ]

        try:
            response = gemini_client.chat(messages, temperature=0.2) # Lower temp for factual summary
            patient_summary = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not patient_summary:
                 patient_summary = "Error: LLM failed to generate the patient-friendly summary."
                 st.error(patient_summary)
                 st.session_state.log_messages.append("[Error] Patient summary generation returned empty.")
            else:
                 st.session_state.log_messages.append("Patient-friendly summary generated.")

            task['patient_summary'] = patient_summary

        except Exception as e:
            st.error(f"Patient-friendly summary generation failed: {e}")
            logging.error(f"Patient summary generation failed: {e}")
            task['patient_summary'] = f"Error: Failed to generate patient summary - {e}"
            st.session_state.log_messages.append(f"[Error] Patient summary generation failed: {e}")

        st.session_state.report_generated = True # Mark final report step as done
        st.session_state.status = "Completed"


    # Return final results from session state task dictionary
    return task.get('comprehensive_report'), task.get('patient_summary'), task.get('citations')


# --- Streamlit App Layout ---

st.set_page_config(page_title="Medical Health Help Assistant", layout="wide")

st.title("🩺 Medical Health Help Assistant")
st.caption("AI-powered analysis based on your symptoms and provided context. Not a substitute for professional medical advice.")

# --- Initialize Session State ---
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'status' not in st.session_state:
    st.session_state.status = "Idle"
if 'error' not in st.session_state:
    st.session_state.error = None
if 'current_task' not in st.session_state:
    st.session_state.current_task = {} # Holds analysis, reports, etc.
if 'analysis_feedback' not in st.session_state:
     st.session_state.analysis_feedback = ""
if 'survey_df' not in st.session_state:
     st.session_state.survey_df = None
if 'wearable_df' not in st.session_state:
     st.session_state.wearable_df = None
if 'vectorizer' not in st.session_state:
     st.session_state.vectorizer = None
if 'clf_model' not in st.session_state:
     st.session_state.clf_model = None
# Flags to control flow and prevent re-running steps unnecessarily
if 'analysis_complete' not in st.session_state:
     st.session_state.analysis_complete = False
if 'search_complete' not in st.session_state:
     st.session_state.search_complete = False
if 'rag_complete' not in st.session_state:
     st.session_state.rag_complete = False
if 'report_generated' not in st.session_state:
     st.session_state.report_generated = False


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")

    st.subheader("API Keys (Required)")
    st.info("Use Streamlit Secrets for deployment. Enter keys here for local testing ONLY.")

    # Use secrets if available, otherwise allow input
    default_gemini = st.secrets.get("GEMINI_API_KEY", "")
    st.session_state.GEMINI_API_KEY = st.text_input("Gemini API Key", type="password", value=st.session_state.get("GEMINI_API_KEY", default_gemini))

    default_tavily = st.secrets.get("TAVILY_API_KEY", "")
    st.session_state.TAVILY_API_KEY = st.text_input("Tavily API Key", type="password", value=st.session_state.get("TAVILY_API_KEY", default_tavily))

    default_supa_url = st.secrets.get("SUPABASE_URL", "")
    st.session_state.SUPABASE_URL = st.text_input("Supabase URL", value=st.session_state.get("SUPABASE_URL", default_supa_url))

    default_supa_key = st.secrets.get("SUPABASE_KEY", "")
    st.session_state.SUPABASE_KEY = st.text_input("Supabase Key (anon or service_role)", type="password", value=st.session_state.get("SUPABASE_KEY", default_supa_key))

    st.divider()
    st.subheader("Search Parameters")
    search_depth = st.selectbox("Search Depth", ["basic", "advanced"], index=1)
    search_breadth = st.number_input("Search Breadth (Results per Query)", min_value=3, max_value=20, value=10, step=1)
    use_faiss = st.checkbox("Use FAISS for RAG (Faster if available)", value=True)

    st.divider()
    st.subheader("Optional Data Files")

    # Survey Data for Classifier Training
    uploaded_survey_file = st.file_uploader("Upload Symptom Survey Data (CSV)", type=['csv'])
    if uploaded_survey_file is not None and st.session_state.get("uploaded_survey_filename") != uploaded_survey_file.name:
         try:
             st.session_state.survey_df = pd.read_csv(uploaded_survey_file)
             st.success(f"Loaded survey data from '{uploaded_survey_file.name}'")
             st.session_state.uploaded_survey_filename = uploaded_survey_file.name
             # Clear cached model when survey data changes
             train_symptom_classifier.clear()
             # Reset model state
             st.session_state.vectorizer = None
             st.session_state.clf_model = None
         except Exception as e:
             st.error(f"Error reading survey CSV: {e}")
             st.session_state.survey_df = None
             st.session_state.uploaded_survey_filename = None
    elif uploaded_survey_file is None and st.session_state.get("uploaded_survey_filename") is not None:
         # File removed
         st.session_state.survey_df = None
         st.session_state.uploaded_survey_filename = None
         st.info("Survey data removed. Diagnosis prediction will use dummy data.")
         train_symptom_classifier.clear()
         st.session_state.vectorizer = None
         st.session_state.clf_model = None


    # Wearable Data
    uploaded_wearable_file = st.file_uploader("Upload Wearable Data (CSV)", type=['csv'])
    if uploaded_wearable_file is not None and st.session_state.get("uploaded_wearable_filename") != uploaded_wearable_file.name:
        try:
            st.session_state.wearable_df = pd.read_csv(uploaded_wearable_file)
            st.success(f"Loaded wearable data from '{uploaded_wearable_file.name}'")
            st.session_state.uploaded_wearable_filename = uploaded_wearable_file.name
            # Clear cache related to wearable data if any
            read_wearable_data.clear()
        except Exception as e:
            st.error(f"Error reading wearable CSV: {e}")
            st.session_state.wearable_df = None
            st.session_state.uploaded_wearable_filename = None
    elif uploaded_wearable_file is None and st.session_state.get("uploaded_wearable_filename") is not None:
        # File removed
        st.session_state.wearable_df = None
        st.session_state.uploaded_wearable_filename = None
        st.info("Wearable data removed.")
        read_wearable_data.clear()

    # Additional Reference Files
    uploaded_ref_files = st.file_uploader(
        "Upload Additional Context Files (PDF, DOCX, TXT, CSV)",
        type=['pdf', 'docx', 'txt', 'csv'],
        accept_multiple_files=True
    )
    # Store content of uploaded files in session state
    additional_files_content = {}
    if uploaded_ref_files:
        st.write("Processing reference files:")
        # Use a sub-key in session state to store file contents
        if 'ref_file_contents' not in st.session_state:
             st.session_state.ref_file_contents = {}
        # Check for changes
        current_filenames = {f.name for f in uploaded_ref_files}
        previous_filenames = set(st.session_state.ref_file_contents.keys())

        # Remove content of deleted files
        for fname in previous_filenames - current_filenames:
             del st.session_state.ref_file_contents[fname]

        # Add/update content of new/modified files
        for file in uploaded_ref_files:
             if file.name not in st.session_state.ref_file_contents: # Process only new files
                 with st.spinner(f"Extracting text from {file.name}..."):
                     content = extract_text_from_file(file)
                     if content:
                         st.session_state.ref_file_contents[file.name] = content
                         st.text(f"- {file.name} (extracted)")
                     else:
                         st.text(f"- {file.name} (extraction failed or empty)")
        additional_files_content = st.session_state.ref_file_contents # Use the processed content
    else:
         st.session_state.ref_file_contents = {} # Clear if no files uploaded


# --- Main Area ---

# Input Section
st.subheader("1. Describe Your Medical Symptoms")
user_query = st.text_area("Enter your current symptoms, duration, and any relevant details:", height=150, key="user_query_input")

cols_urls = st.columns(2)
with cols_urls[0]:
    include_urls_input = st.text_area("Include Specific URLs (Optional, one per line):", height=75)
with cols_urls[1]:
    omit_urls_input = st.text_area("Omit Specific URLs (Optional, one per line):", height=75)

include_urls = [url.strip() for url in include_urls_input.splitlines() if url.strip()]
omit_urls = [url.strip() for url in omit_urls_input.splitlines() if url.strip()]

# Start Button
start_button = st.button("Start Diagnosis Process", type="primary", use_container_width=True)

st.divider()


# Processing and Output Area
if start_button and not user_query:
    st.warning("Please enter your medical symptoms.")
elif start_button and user_query:
    # Reset state for a new run
    st.session_state.status = "Initializing..."
    st.session_state.error = None
    st.session_state.current_task = {} # Clear previous task data
    st.session_state.analysis_complete = False
    st.session_state.search_complete = False
    st.session_state.rag_complete = False
    st.session_state.report_generated = False
    st.session_state.analysis_feedback = "" # Clear previous feedback

    # --- Initial Processing (Sentiment, Entities, Prediction) ---
    with st.spinner("Performing initial analysis (Sentiment, Entities, Prediction)..."):
        # Check NLTK/Spacy dependencies here if not done globally
        try: install_nltk_data()
        except: pass # Ignore errors here, handled in function
        try: install_spacy_model()
        except: pass

        sentiment_score = analyze_sentiment(user_query)
        medical_entities = extract_medical_entities(user_query)

        # Train or load classifier
        if st.session_state.vectorizer is None or st.session_state.clf_model is None:
             # Pass the DataFrame from session state
             st.session_state.vectorizer, st.session_state.clf_model = train_symptom_classifier(st.session_state.get("survey_df"))

        predicted_diag, diag_proba = predict_diagnosis(user_query, st.session_state.vectorizer, st.session_state.clf_model)
        explanation = explain_diagnosis(user_query, st.session_state.vectorizer, st.session_state.clf_model)

        # Save to history
        save_query_history(user_query, predicted_diag, sentiment_score, medical_entities)

    st.subheader("2. Initial Analysis Insights")
    cols_initial = st.columns(3)
    with cols_initial[0]:
        st.metric("Sentiment", f"{sentiment_score.get('compound', 0.0):.2f}", help=f"Positive: {sentiment_score.get('pos', 0.0):.2f}, Neutral: {sentiment_score.get('neu', 0.0):.2f}, Negative: {sentiment_score.get('neg', 0.0):.2f}")
    with cols_initial[1]:
        st.markdown("**Predicted Diagnosis:**")
        st.markdown(f"**{predicted_diag}**")
        if diag_proba:
            with st.expander("View Probabilities"):
                st.json({k: f"{v:.3f}" for k, v in diag_proba.items()}) # Show probabilities nicely
    with cols_initial[2]:
        st.markdown("**Extracted Entities:**")
        if medical_entities:
             st.markdown(", ".join(f"`{e}`" for e in medical_entities))
        else:
             st.info("None found.")

    with st.expander("View Diagnosis Explanation (SHAP)"):
        st.markdown(explanation)

    st.divider()

    # --- Run the main diagnostic process ---
    st.subheader("3. In-Depth Analysis & Report Generation")
    progress_bar = st.progress(0, text="Starting diagnostic process...")
    status_text = st.empty()
    status_text.info("Initializing...")

    # Wrap the main process in a try-except block
    try:
        # Run the main diagnostic function
        comprehensive_report, patient_summary, citations = run_diagnostic_process(
            query=user_query,
            include_urls=include_urls,
            omit_urls=omit_urls,
            additional_files_content=additional_files_content, # Pass processed content
            search_depth=search_depth,
            search_breadth=search_breadth,
            use_faiss=use_faiss,
            feedback=None # Initial run has no feedback yet
        )

        # Update progress and status based on state flags set within the function
        if st.session_state.analysis_complete:
             progress_bar.progress(25, text="Analysis complete. Starting search...")
             status_text.info("Analysis complete. Starting search...")
        if st.session_state.search_complete:
             progress_bar.progress(50, text="Search & Extraction complete. Starting RAG...")
             status_text.info("Search & Extraction complete. Starting RAG...")
        if st.session_state.rag_complete:
             progress_bar.progress(75, text="RAG complete. Generating reports...")
             status_text.info("RAG complete. Generating reports...")
        if st.session_state.report_generated:
             progress_bar.progress(100, text="Process completed.")
             if st.session_state.error:
                  status_text.error(f"Process finished with errors: {st.session_state.error}")
             else:
                  status_text.success("Diagnostic process completed successfully!")


    except Exception as e:
        st.session_state.error = f"An unexpected error occurred: {e}"
        st.session_state.status = "Error"
        logging.error(f"Unexpected error in main process: {e}", exc_info=True)
        status_text.error(f"An unexpected error occurred: {e}")
        progress_bar.progress(100, text="Process failed.")


    # --- Feedback Section (Appears after initial analysis) ---
    if st.session_state.get('analysis_complete', False) and not st.session_state.get('report_generated', False): # Show feedback only after analysis, before final report is locked in
         st.subheader("4. Review Agent's Understanding & Provide Feedback (Optional)")
         feedback = st.text_area("If the agent's understanding seems incorrect or incomplete, provide clarification here:", key="feedback_input")
         rerun_button = st.button("Re-run Analysis with Feedback")

         if rerun_button and feedback:
             # Trigger re-run with feedback
             status_text.info("Re-running process with feedback...")
             progress_bar.progress(0, text="Re-starting analysis with feedback...")
             try:
                 comprehensive_report, patient_summary, citations = run_diagnostic_process(
                     query=user_query, # Use original query here
                     include_urls=include_urls,
                     omit_urls=omit_urls,
                     additional_files_content=additional_files_content,
                     search_depth=search_depth,
                     search_breadth=search_breadth,
                     use_faiss=use_faiss,
                     feedback=feedback # Pass the feedback
                 )
                 # Update progress based on state flags after re-run
                 if st.session_state.analysis_complete: progress_bar.progress(25, text="Re-Analysis complete. Starting search..."); status_text.info("...")
                 if st.session_state.search_complete: progress_bar.progress(50, text="Search & Extraction complete. Starting RAG..."); status_text.info("...")
                 if st.session_state.rag_complete: progress_bar.progress(75, text="RAG complete. Generating reports..."); status_text.info("...")
                 if st.session_state.report_generated:
                      progress_bar.progress(100, text="Process completed with feedback.")
                      if st.session_state.error: status_text.error(f"Process finished with errors: {st.session_state.error}")
                      else: status_text.success("Process completed successfully with feedback!")

             except Exception as e:
                 st.session_state.error = f"An unexpected error occurred during re-run: {e}"
                 st.session_state.status = "Error"
                 logging.error(f"Unexpected error during re-run: {e}", exc_info=True)
                 status_text.error(f"An unexpected error occurred during re-run: {e}")
                 progress_bar.progress(100, text="Re-run failed.")


# --- Display Final Reports (if generated) ---
if st.session_state.get('report_generated', False) and 'current_task' in st.session_state:
    task = st.session_state.current_task
    comp_report = task.get('comprehensive_report', 'Not generated.')
    pat_summary = task.get('patient_summary', 'Not generated.')
    citations = task.get('citations', [])

    st.subheader("Final Diagnostic Reports")

    # Generate a simple summary report (could be enhanced)
    summary_report = f"# Summary Diagnostic Report\n\n"
    summary_report += f"**Patient Query:** {task.get('original_query', 'N/A')}\n\n"
    summary_report += f"**Key Findings (from Comprehensive Report):**\n"
    # Basic summary - extract first few lines or use LLM for better summary if needed
    summary_lines = [line for line in comp_report.splitlines() if line.strip() and not line.startswith("#")]
    summary_report += "\n".join(summary_lines[:15]) + ("..." if len(summary_lines) > 15 else "")
    summary_report += f"\n\n**Sources Considered:** {len(citations)} sources (Web, User Files, Wearable Data)."
    summary_report += "\n\n---\n*This is a basic summary. Please refer to the Comprehensive Report for details.*"


    tab1, tab2, tab3 = st.tabs(["Patient-Friendly Summary", "Comprehensive Report", "Basic Summary"])

    with tab1:
        st.markdown(pat_summary)
        st.download_button(
            label="Download Patient Summary",
            data=pat_summary.encode('utf-8'),
            file_name="patient_summary_report.md",
            mime="text/markdown"
        )

    with tab2:
        st.markdown(comp_report)
        with st.expander("View Sources/Citations Used"):
             if citations:
                 for cit in citations: st.markdown(f"- {cit}")
             else:
                 st.info("No specific sources listed.")
        st.download_button(
            label="Download Comprehensive Report",
            data=comp_report.encode('utf-8'),
            file_name="comprehensive_diagnostic_report.md",
            mime="text/markdown"
        )

    with tab3:
        st.markdown(summary_report)
        st.download_button(
            label="Download Basic Summary Report",
            data=summary_report.encode('utf-8'),
            file_name="basic_summary_report.md",
            mime="text/markdown"
        )


# --- Optional: Visualization Section ---
st.divider()
st.subheader("Query History & Trends")

show_viz = st.checkbox("Show Query Trend Visualizations")

if show_viz:
    with st.spinner("Loading query history and generating visualizations..."):
        history_df = get_query_history_df()
        if not history_df.empty:
            st.dataframe(history_df.tail(10)) # Show last 10 queries
            figures = generate_visualization_figures(history_df)
            display_visualizations(figures)
        else:
            st.info("No query history found to display or visualize.")

# --- Log Display Expander ---
st.divider()
with st.expander("View Processing Logs"):
    log_text = "\n".join(st.session_state.get('log_messages', ["No logs yet."]))
    st.code(log_text, language=None) # Use st.code for simple text display

# --- Final Status Display ---
if st.session_state.status == "Completed" and not st.session_state.error:
    st.success("Process finished successfully.")
elif st.session_state.error:
    st.error(f"Process finished with an error: {st.session_state.error}")
elif st.session_state.status != "Idle":
    st.info(f"Current Status: {st.session_state.status}")


# --- Disclaimer ---
st.sidebar.divider()
st.sidebar.warning("**Disclaimer:** This application provides AI-generated analysis for informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")
