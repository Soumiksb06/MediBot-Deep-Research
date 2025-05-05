import streamlit as st
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
import json # Import json for parsing Gemini responses
import matplotlib.pyplot as plt # Import matplotlib for visualizations
import pandas as pd # Import pandas for data handling in visualizations
from collections import Counter # Import Counter for entity distribution
import sys
import Dict

# Configure logging (optional in Streamlit, but good for debugging)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Add project root to sys.path if needed - Adjust path based on your project structure
# Assuming the structure where src contains the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import agent modules
try:
    from web_agent.src.services.web_search import WebSearchService
    from web_agent.src.models.search_models import SearchConfig
    from subject_analyzer.src.services.tavily_client import TavilyClient
    from subject_analyzer.src.services.tavily_extractor import TavilyExtractor
    from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer
    from subject_analyzer.src.services.gemini_client import GeminiClient
    from subject_analyzer.src.models.analysis_models import AnalysisConfig
    # Import functions from med_agent_new.py (will be adapted or moved)
    from med_agent_new import extract_text_from_file, train_symptom_classifier, predict_diagnosis, explain_diagnosis, save_query_history, read_wearable_data, MedicalTask
except ImportError as e:
    st.error(f"Failed to import necessary modules. Please ensure all files are in the correct directory structure and dependencies are installed. Error: {e}")
    st.stop()


# ==============================
# Gemini-based replacements for NLTK and SpaCy
# ==============================

async def analyze_sentiment_gemini(gemini_client: GeminiClient, query: str) -> Dict:
    """
    Performs sentiment analysis on the query using Gemini.
    Returns a dictionary with sentiment scores (e.g., compound, positive, negative, neutral).
    """
    prompt = f"""Analyze the sentiment of the following medical query and provide a sentiment score.
    Query: "{query}"
    Respond with a JSON object containing 'compound', 'positive', 'negative', and 'neutral' scores, ranging from 0.0 to 1.0.
    Example response: {{"compound": 0.8, "positive": 0.7, "negative": 0.1, "neutral": 0.2}}
    Ensure the response is ONLY the JSON object and is valid JSON.
    """
    messages = [
        {"role": "system", "content": "You are a sentiment analysis expert and respond ONLY with valid JSON."},
        {"role": "user", "content": prompt}
    ]
    try:
        # Run synchronous chat in a thread pool to avoid blocking Streamlit's event loop
        response = await asyncio.to_thread(gemini_client.chat, messages)
        result = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Attempt to parse JSON
        try:
            sentiment_scores = json.loads(result)
            # Basic validation of keys and value types (ensure they are numbers)
            if all(k in sentiment_scores and isinstance(sentiment_scores[k], (int, float)) for k in ['compound', 'positive', 'negative', 'neutral']):
                return sentiment_scores
            else:
                st.warning("Gemini sentiment analysis returned invalid JSON structure or non-numeric scores.")
                return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
        except json.JSONDecodeError:
            st.warning(f"Gemini sentiment analysis returned non-JSON response: {result[:100]}...")
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
    except Exception as e:
        st.error(f"Error in Gemini sentiment analysis: {e}")
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0} # Return neutral on error

async def extract_medical_entities_gemini(gemini_client: GeminiClient, query: str) -> List[str]:
    """
    Extracts medical entities from the query using Gemini.
    Returns a list of extracted medical terms.
    """
    prompt = f"""Extract key medical entities (like symptoms, conditions, body parts, medications, procedures) from the following query.
    Query: "{query}"
    Respond with a JSON object containing a single key "entities" which is a list of the extracted medical terms.
    Example response: {{"entities": ["fever", "cough", "headache", "appendicitis", "CT scan"]}}
    Ensure the response is ONLY the JSON object and is valid JSON.
    """
    messages = [
        {"role": "system", "content": "You are a medical entity extraction expert and respond ONLY with valid JSON."},
        {"role": "user", "content": prompt}
    ]
    try:
        # Run synchronous chat in a thread pool
        response = await asyncio.to_thread(gemini_client.chat, messages)
        result = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Attempt to parse JSON
        try:
            entities_json = json.loads(result)
             # Basic validation of keys and value type
            if isinstance(entities_json.get("entities"), list):
                return entities_json.get("entities", [])
            else:
                st.warning("Gemini entity extraction returned invalid JSON structure.")
                return []
        except json.JSONDecodeError:
            st.warning(f"Gemini entity extraction returned non-JSON response: {result[:100]}...")
            return []
    except Exception as e:
        st.error(f"Error in Gemini entity extraction: {e}")
        return [] # Return empty list on error

def visualize_query_trends_streamlit():
    """
    Generates visualizations from the query history and returns matplotlib figures.
    """
    filename = "query_history.csv"
    if not os.path.exists(filename):
        st.info("No query history available for visualization.")
        return None, None, None
    df = pd.read_csv(filename)
    if df.empty:
        st.info("Query history is empty.")
        return None, None, None

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Bar plot for predicted diagnosis frequency
    diag_counts = df['predicted_diagnosis'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(8,6))
    # Handle cases where diag_counts might be empty
    if not diag_counts.empty:
        sns.barplot(x=diag_counts.index, y=diag_counts.values, palette="viridis", ax=ax1)
        ax1.set_title("Frequency of Predicted Diagnoses")
        ax1.set_xlabel("Diagnosis")
        ax1.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right') # Rotate labels for readability
    else:
        ax1.text(0.5, 0.5, "No diagnosis data", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title("Frequency of Predicted Diagnoses")
    plt.tight_layout()


    # Line plot for sentiment compound over time
    df = df.sort_values("timestamp")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    # Handle cases where df might be empty after sorting
    if not df.empty:
        sns.lineplot(data=df, x="timestamp", y="sentiment_compound", marker="o", ax=ax2)
        ax2.set_title("Sentiment Compound Score Over Time")
        ax2.set_xlabel("Timestamp")
        ax2.set_ylabel("Sentiment Compound Score")
        plt.xticks(rotation=45, ha='right') # Rotate labels for readability
    else:
         ax2.text(0.5, 0.5, "No sentiment data", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
         ax2.set_title("Sentiment Compound Score Over Time")
    plt.tight_layout()


    # Pie chart for common medical entities
    entity_list = []
    # Ensure 'entities' column exists and handle potential NaNs
    if 'entities' in df.columns:
        for entities_str in df['entities'].fillna(""):
            if entities_str.strip() == "":
                continue
            entities = [e.strip() for e in entities_str.split(",") if e.strip()]
            entity_list.extend(entities)

    fig3, ax3 = plt.subplots(figsize=(8,8))
    if entity_list:
        entity_counts = Counter(entity_list)
        labels = list(entity_counts.keys())
        sizes = list(entity_counts.values())
        ax3.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        ax3.set_title("Distribution of Extracted Medical Entities")
    else:
        ax3.text(0.5, 0.5, "No entity data", horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        ax3.set_title("Distribution of Extracted Medical Entities")

    plt.tight_layout()

    return fig1, fig2, fig3


# ==============================
# Streamlit App
# ==============================

st.set_page_config(page_title="Medical Health Agent", layout="wide")

st.header("Medical Health Help System")

load_dotenv()

# Initialize session state
if 'task' not in st.session_state:
    st.session_state.task = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'search_extract_done' not in st.session_state:
    st.session_state.search_extract_done = False
if 'rag_analysis_done' not in st.session_state:
    st.session_state.rag_analysis_done = False
if 'reports_saved' not in st.session_state:
    st.session_state.reports_saved = False
if 'patient_report_saved' not in st.session_state:
    st.session_state.patient_report_saved = False
if 'sentiment_entities_predicted' not in st.session_state:
    st.session_state.sentiment_entities_predicted = False
if 'full_report_content' not in st.session_state:
    st.session_state.full_report_content = None
if 'summary_report_content' not in st.session_state:
    st.session_state.summary_report_content = None
if 'patient_report_content' not in st.session_state:
    st.session_state.patient_report_content = None
if 'diagnosis_prediction' not in st.session_state:
    st.session_state.diagnosis_prediction = None
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = None
if 'entity_results' not in st.session_state:
    st.session_state.entity_results = None
if 'diagnosis_explanation' not in st.session_state:
    st.session_state.diagnosis_explanation = None
if 'wearable_summary' not in st.session_state:
    st.session_state.wearable_summary = None
if 'additional_files_content' not in st.session_state:
    st.session_state.additional_files_content = {}
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'clf_model' not in st.session_state:
    st.session_state.clf_model = None


# Load environment variables and initialize clients
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") # Use a Google embedding model

if not GEMINI_API_KEY or not TAVILY_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Please ensure GEMINI_API_KEY, TAVILY_API_KEY, SUPABASE_URL, and SUPABASE_KEY are set in your environment variables.")
    st.stop()

# Initialize clients (these can be cached)
@st.cache_resource
def initialize_clients(gemini_api_key, tavily_api_key):
    console = Console() # Use Rich console for internal logging/debugging if needed, Streamlit handles display
    analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
    gemini_client = GeminiClient(api_key=gemini_api_key, config=analysis_config)
    search_client = TavilyClient(api_key=tavily_api_key)
    extractor = TavilyExtractor(api_key=tavily_api_key)
    search_config = SearchConfig()
    search_service = WebSearchService(search_client, search_config)
    subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config)
    return gemini_client, search_service, extractor, subject_analyzer

gemini_client, search_service, extractor, subject_analyzer = initialize_clients(GEMINI_API_KEY, TAVILY_API_KEY)

# Train classifier on startup and cache it
@st.cache_resource
def get_classifier():
    st.info("Training symptom classifier...")
    try:
        vectorizer, clf_model = train_symptom_classifier()
        if vectorizer is not None and clf_model is not None:
            st.success("Symptom classifier trained.")
            return vectorizer, clf_model
        else:
             st.warning("Could not train classifier. Diagnosis prediction will be unavailable.")
             return None, None
    except Exception as e:
        st.error(f"Error training classifier: {e}")
        return None, None

# Only train the classifier once
if st.session_state.vectorizer is None or st.session_state.clf_model is None:
    st.session_state.vectorizer, st.session_state.clf_model = get_classifier()


# --- Input Section ---
st.subheader("Patient Information and Query")
original_query = st.text_area("Describe your current medical symptoms or issue:", key="original_query")

col1, col2 = st.columns(2)
with col1:
    include_urls_input = st.text_input("Enter URL(s) to include (comma-separated):", key="include_urls")
    include_urls = [url.strip() for url in include_urls_input.split(',')] if include_urls_input else []
with col2:
    omit_urls_input = st.text_input("Enter URL(s) to omit (comma-separated):", key="omit_urls")
    omit_urls = [url.strip() for url in omit_urls_input.split(',')] if omit_urls_input else []

uploaded_files = st.file_uploader("Upload additional reference files (PDF, DOCX, CSV, Excel, TXT)", type=["pdf", "docx", "csv", "xls", "xlsx", "txt"], accept_multiple_files=True, key="uploaded_files")

if uploaded_files:
    st.session_state.additional_files_content = {}
    console_temp = Console() # Use a temporary console for file extraction messages
    for uploaded_file in uploaded_files:
        # Streamlit file uploader provides file-like objects, not paths
        # Need to save temporarily or pass the file object
        # Saving temporarily is simpler for compatibility with existing extract_text_from_file
        file_path = os.path.join("./temp_uploads", uploaded_file.name)
        os.makedirs("./temp_uploads", exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            content = extract_text_from_file(file_path, console_temp) # Pass console_temp
            if content:
                st.session_state.additional_files_content[uploaded_file.name] = content
            os.remove(file_path) # Clean up temporary file
        except Exception as e:
            st.error(f"Error processing uploaded file {uploaded_file.name}: {e}")
            if os.path.exists(file_path):
                os.remove(file_path) # Ensure cleanup even on error


search_depth = st.selectbox("Search Depth:", ["basic", "advanced"], index=1, key="search_depth")
search_breadth = st.number_input("Search Breadth (number of results):", min_value=1, value=10, key="search_breadth")

# --- Process Buttons ---
st.subheader("Run Analysis Steps")

if st.button("Analyze Query and Predict Diagnosis", key="analyze_button"):
    if not original_query:
        st.warning("Please enter a medical query.")
    else:
        # Clear previous results for a new query
        st.session_state.task = MedicalTask(original_query, Console()) # Re-initialize task for new query
        st.session_state.analysis_done = False
        st.session_state.search_extract_done = False
        st.session_state.rag_analysis_done = False
        st.session_state.reports_saved = False
        st.session_state.patient_report_saved = False
        st.session_state.sentiment_entities_predicted = False
        st.session_state.full_report_content = None
        st.session_state.summary_report_content = None
        st.session_state.patient_report_content = None
        st.session_state.diagnosis_prediction = None
        st.session_state.sentiment_results = None
        st.session_state.entity_results = None
        st.session_state.diagnosis_explanation = None
        st.session_state.wearable_summary = None
        st.session_state.additional_files_content = {} # Clear file content on new query
        st.session_state.uploaded_files = [] # Clear file uploader state


        with st.spinner("Analyzing query, predicting diagnosis, sentiment, and extracting entities..."):
            try:
                current_date = datetime.today().strftime("%Y-%m-%d")
                # Subject Analysis
                asyncio.run(st.session_state.task.analyze(subject_analyzer, current_date))
                st.session_state.analysis_done = True
                st.success("Query Analysis Complete.")

                # Sentiment Analysis (Gemini)
                st.session_state.sentiment_results = asyncio.run(analyze_sentiment_gemini(gemini_client, original_query))
                st.write("Sentiment Analysis Results:", st.session_state.sentiment_results)

                # Entity Extraction (Gemini)
                st.session_state.entity_results = asyncio.run(extract_medical_entities_gemini(gemini_client, original_query))
                st.write("Extracted Medical Entities:", st.session_state.entity_results)


                # Diagnosis Prediction and Explanation (if classifier is available)
                if st.session_state.vectorizer and st.session_state.clf_model:
                    predicted_diag, diag_proba = predict_diagnosis(original_query, st.session_state.vectorizer, st.session_state.clf_model)
                    st.session_state.diagnosis_prediction = f"{predicted_diag}"
                    # Format probability dictionary nicely
                    proba_str = ", ".join([f"{k}: {v:.2f}" for k, v in diag_proba.items()])
                    st.write(f"Predicted Diagnosis: **{predicted_diag}**")
                    st.write(f"Probabilities: {proba_str}")


                    st.session_state.diagnosis_explanation = explain_diagnosis(original_query, st.session_state.vectorizer, st.session_state.clf_model)
                    st.write("Diagnosis Explanation:", st.session_state.diagnosis_explanation)
                else:
                    st.warning("Symptom classifier not available. Cannot predict diagnosis.")
                    st.session_state.diagnosis_prediction = "Classifier not available."
                    st.session_state.diagnosis_explanation = "Classifier not available."


                # Save query history
                # Ensure sentiment_results has the 'compound' key before saving
                sentiment_compound = st.session_state.sentiment_results.get("compound", 0) if st.session_state.sentiment_results else 0
                # Use the predicted diagnosis string directly, or handle the "Classifier not available" case
                diag_to_save = st.session_state.diagnosis_prediction if st.session_state.diagnosis_prediction != "Classifier not available." else "N/A"
                save_query_history(original_query, diag_to_save, sentiment_compound, st.session_state.entity_results)
                st.info("Query history saved.")

                st.session_state.sentiment_entities_predicted = True

            except Exception as e:
                st.error(f"An error occurred during query analysis: {e}")
                logging.error(f"Query analysis error: {e}", exc_info=True) # Log traceback

if st.session_state.analysis_done:
    st.subheader("Agent's Understanding")
    st.write("**Patient Query:**", st.session_state.task.original_query)
    st.write("**Identified Medical Issue:**", st.session_state.task.analysis.get('main_subject', 'Unknown Issue'))
    temporal = st.session_state.task.analysis.get("temporal_context", {})
    if temporal:
        st.write("**Temporal Context:**")
        for key, value in temporal.items():
            st.write(f"- {key.capitalize()}: {value}")
    else:
        st.write("**Temporal Context:** None")
    needs = st.session_state.task.analysis.get("What_needs_to_be_researched", [])
    st.write("**Key aspects to investigate:**", ', '.join(needs) if needs else 'None')

    # Feedback loop
    feedback = st.text_area("Provide feedback on the agent's understanding (optional):", key="feedback")
    if st.button("Update Analysis with Feedback", key="feedback_button"):
        if feedback and st.session_state.task:
            st.session_state.task.update_feedback(feedback)
            st.session_state.task.current_query = f"{original_query} - Additional context: {feedback}"
            with st.spinner("Reanalyzing query with your feedback..."):
                 try:
                    current_date = datetime.today().strftime("%Y-%m-%d")
                    asyncio.run(st.session_state.task.analyze(subject_analyzer, current_date))
                    st.session_state.analysis_done = True # Keep analysis_done as True
                    st.success("Query Analysis Updated with Feedback.")
                    # Re-display updated analysis
                    st.subheader("Agent's Understanding (Updated)")
                    st.write("**Patient Query:**", st.session_state.task.original_query)
                    st.write("**Identified Medical Issue:**", st.session_state.task.analysis.get('main_subject', 'Unknown Issue'))
                    temporal = st.session_state.task.analysis.get("temporal_context", {})
                    if temporal:
                        st.write("**Temporal Context:**")
                        for key, value in temporal.items():
                            st.write(f"- {key.capitalize()}: {value}")
                    else:
                        st.write("**Temporal Context:** None")
                    needs = st.session_state.task.analysis.get("What_needs_to_be_researched", [])
                    st.write("**Key aspects to investigate:**", ', '.join(needs) if needs else 'None')
                 except Exception as e:
                    st.error(f"An error occurred during feedback analysis: {e}")
                    logging.error(f"Feedback analysis error: {e}", exc_info=True) # Log traceback
        else:
            st.info("No feedback provided or task not initialized.")


if st.session_state.analysis_done:
    if st.button("Search and Extract Information", key="search_button"):
        if st.session_state.task:
            with st.spinner("Searching the web and extracting content..."):
                try:
                    # Pass the correct omit_urls and search parameters
                    asyncio.run(st.session_state.task.search_and_extract(search_service, extractor, omit_urls, search_depth, search_breadth))
                    st.session_state.search_extract_done = True
                    st.success("Search and Extraction Complete.")
                except Exception as e:
                    st.error(f"An error occurred during search and extraction: {e}")
                    logging.error(f"Search and extraction error: {e}", exc_info=True) # Log traceback
        else:
            st.warning("Please analyze the query first.")


if st.session_state.search_extract_done:
    st.subheader("Search Results")
    # Ensure search_results is a dictionary before iterating
    if isinstance(st.session_state.task.search_results, dict):
        for topic, results in st.session_state.task.search_results.items():
            st.write(f"**Search for: {topic}**")
            if results:
                for res in results:
                    # Ensure res is a dictionary before accessing keys
                    if isinstance(res, dict):
                         st.write(f"- [{res.get('title', 'No Title')}]({res.get('url', '#')}) (Relevance: {res.get('score', 'N/A')})")
                    else:
                         st.write(f"- Invalid search result format: {res}")
            else:
                st.write("No search results found.")
    else:
        st.write("No search results available.")


    st.subheader("Extracted Content")
    # Ensure extracted_content is a dictionary before iterating
    if isinstance(st.session_state.task.extracted_content, dict):
        for topic, items in st.session_state.task.extracted_content.items():
            st.write(f"**Extraction for: {topic}**")
            if items:
                for item in items:
                    # Ensure item is a dictionary before accessing keys
                    if isinstance(item, dict):
                        url = item.get("url", "No URL")
                        text = item.get("text") or item.get("raw_content", "")
                        with st.expander(f"Content from {url}"):
                            st.write(text)
                    else:
                        st.write(f"- Invalid extracted content format: {item}")
            else:
                st.write("No content extracted.")
    else:
        st.write("No extracted content available.")


if st.session_state.search_extract_done:
    if st.button("Perform RAG Analysis and Generate Reports", key="rag_button"):
        if st.session_state.task:
            with st.spinner("Performing RAG analysis and generating reports..."):
                try:
                    # Pass the gemini_client for RAG analysis
                    comprehensive_report, citations = asyncio.run(st.session_state.task.analyze_full_content_rag(gemini_client))
                    st.session_state.full_report_content = st.session_state.task.generate_report(st.session_state.additional_files_content, comprehensive_report, citations)
                    st.session_state.summary_report_content = st.session_state.task.generate_summary_report(comprehensive_report, citations)

                    # Read wearable data if available
                    st.session_state.wearable_summary = read_wearable_data()
                    if st.session_state.wearable_summary:
                        st.info("Wearable data integrated into the report.")


                    st.session_state.rag_analysis_done = True
                    st.success("RAG Analysis and Report Generation Complete.")
                except Exception as e:
                    st.error(f"An error occurred during RAG analysis and report generation: {e}")
                    logging.error(f"RAG analysis error: {e}", exc_info=True) # Log traceback
        else:
            st.warning("Please perform search and extraction first.")


if st.session_state.rag_analysis_done:
    st.subheader("Generated Reports")
    if st.session_state.full_report_content:
        with st.expander("Full Diagnostic Report"):
            st.markdown(st.session_state.full_report_content)
        # Sanitize filename for download
        safe_query = ''.join(c if c.isalnum() or c.isspace() else '_' for c in st.session_state.task.original_query).strip().replace(' ', '_')
        full_filename = f"{safe_query}_diagnostic_full.md" if safe_query else "diagnostic_full.md"
        st.download_button("Download Full Report (Markdown)", st.session_state.full_report_content, file_name=full_filename)

    if st.session_state.summary_report_content:
        with st.expander("Summary Diagnostic Report"):
            st.markdown(st.session_state.summary_report_content)
        # Sanitize filename for download
        safe_query = ''.join(c if c.isalnum() or c.isspace() else '_' for c in st.session_state.task.original_query).strip().replace(' ', '_')
        summary_filename = f"{safe_query}_diagnostic_summary.md" if safe_query else "diagnostic_summary.md"
        st.download_button("Download Summary Report (Markdown)", st.session_state.summary_report_content, file_name=summary_filename)

    if st.button("Generate Patient-Friendly Summary", key="patient_summary_button"):
         if st.session_state.full_report_content and st.session_state.task:
             with st.spinner("Generating patient-friendly summary..."):
                 try:
                     # Pass the gemini_client for patient summary generation
                     # Pass full report content and citations (even if not used by prompt, maintain signature)
                     patient_report = asyncio.run(st.session_state.task.generate_patient_summary_report(gemini_client, st.session_state.full_report_content, []))
                     st.session_state.patient_report_content = patient_report
                     st.session_state.patient_report_saved = True
                     st.success("Patient-Friendly Summary Generated.")
                 except Exception as e:
                    st.error(f"An error occurred during patient summary generation: {e}")
                    logging.error(f"Patient summary generation error: {e}", exc_info=True) # Log traceback
         else:
            st.warning("Please generate the full report first and ensure the task is initialized.")


if st.session_state.patient_report_saved:
    st.subheader("Patient-Friendly Summary")
    if st.session_state.patient_report_content:
         st.markdown(st.session_state.patient_report_content)
         # Sanitize filename for download
         safe_query = ''.join(c if c.isalnum() or c.isspace() else '_' for c in st.session_state.task.original_query).strip().replace(' ', '_')
         patient_filename = f"{safe_query}_patient_summary.md" if safe_query else "patient_summary.md"
         st.download_button("Download Patient Summary (Markdown)", st.session_state.patient_report_content, file_name=patient_filename)

# --- Visualizations ---
st.subheader("Query Trend Visualizations")
if st.button("Generate Visualizations", key="viz_button"):
     with st.spinner("Generating visualizations from query history..."):
         try:
             fig1, fig2, fig3 = visualize_query_trends_streamlit()
             if fig1:
                 st.pyplot(fig1)
             if fig2:
                 st.pyplot(fig2)
             if fig3:
                 st.pyplot(fig3)
             if not fig1 and not fig2 and not fig3:
                 st.info("No data available to generate visualizations.")

         except Exception as e:
            st.error(f"An error occurred during visualization generation: {e}")
            logging.error(f"Visualization error: {e}", exc_info=True) # Log traceback

