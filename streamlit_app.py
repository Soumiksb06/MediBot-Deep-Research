%%writefile streamlit_app.py
import streamlit as st
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
import json # Import json for parsing Gemini responses

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
    from med_agent_new import extract_text_from_file, train_symptom_classifier, predict_diagnosis, explain_diagnosis, save_query_history, visualize_query_trends, read_wearable_data, MedicalTask
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
    Respond with a JSON object containing 'compound', 'positive', 'negative', and 'neutral' scores.
    Example response: {{"compound": 0.8, "positive": 0.7, "negative": 0.1, "neutral": 0.2}}
    Ensure the response is ONLY the JSON object and is valid JSON.
    """
    messages = [
        {"role": "system", "content": "You are a sentiment analysis expert and respond ONLY with valid JSON."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = await asyncio.to_thread(gemini_client.chat, messages) # Run sync chat in a thread
        result = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Attempt to parse JSON
        try:
            sentiment_scores = json.loads(result)
            # Basic validation of keys
            if all(k in sentiment_scores for k in ['compound', 'positive', 'negative', 'neutral']):
                return sentiment_scores
            else:
                st.warning("Gemini sentiment analysis returned invalid JSON structure.")
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
    prompt = f"""Extract key medical entities (like symptoms, conditions, body parts, medications) from the following query.
    Query: "{query}"
    Respond with a JSON object containing a single key "entities" which is a list of the extracted medical terms.
    Example response: {{"entities": ["fever", "cough", "headache"]}}
    Ensure the response is ONLY the JSON object and is valid JSON.
    """
    messages = [
        {"role": "system", "content": "You are a medical entity extraction expert and respond ONLY with valid JSON."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = await asyncio.to_thread(gemini_client.chat, messages) # Run sync chat in a thread
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
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-exp-04-17") # Use a Google embedding model

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
        if vectorizer and clf_model:
            st.success("Symptom classifier trained.")
            return vectorizer, clf_model
        else:
             st.warning("Could not train classifier. Diagnosis prediction will be unavailable.")
             return None, None
    except Exception as e:
        st.error(f"Error training classifier: {e}")
        return None, None

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
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        content = extract_text_from_file(file_path, console_temp) # Pass console_temp
        if content:
            st.session_state.additional_files_content[uploaded_file.name] = content
        os.remove(file_path) # Clean up temporary file

search_depth = st.selectbox("Search Depth:", ["basic", "advanced"], index=1, key="search_depth")
search_breadth = st.number_input("Search Breadth (number of results):", min_value=1, value=10, key="search_breadth")

# --- Process Buttons ---
st.subheader("Run Analysis Steps")

if st.button("Analyze Query and Predict Diagnosis", key="analyze_button"):
    if not original_query:
        st.warning("Please enter a medical query.")
    else:
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
                    st.session_state.diagnosis_prediction = f"{predicted_diag} with probabilities: {diag_proba}"
                    st.write("Predicted Diagnosis:", st.session_state.diagnosis_prediction)

                    st.session_state.diagnosis_explanation = explain_diagnosis(original_query, st.session_state.vectorizer, st.session_state.clf_model)
                    st.write("Diagnosis Explanation:", st.session_state.diagnosis_explanation)
                else:
                    st.warning("Symptom classifier not available. Cannot predict diagnosis.")
                    st.session_state.diagnosis_prediction = "Classifier not available."
                    st.session_state.diagnosis_explanation = "Classifier not available."


                # Save query history
                # Ensure sentiment_results has the 'compound' key before saving
                sentiment_compound = st.session_state.sentiment_results.get("compound", 0) if st.session_state.sentiment_results else 0
                save_query_history(original_query, st.session_state.diagnosis_prediction.split(" with probabilities:")[0] if st.session_state.diagnosis_prediction and " with probabilities:" in st.session_state.diagnosis_prediction else "N/A", sentiment_compound, st.session_state.entity_results)
                st.info("Query history saved.")

                st.session_state.sentiment_entities_predicted = True

            except Exception as e:
                st.error(f"An error occurred during query analysis: {e}")

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
        if feedback:
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
        else:
            st.info("No feedback provided.")


if st.session_state.analysis_done:
    if st.button("Search and Extract Information", key="search_button"):
        with st.spinner("Searching the web and extracting content..."):
            try:
                # Pass the correct omit_urls and search parameters
                asyncio.run(st.session_state.task.search_and_extract(search_service, extractor, omit_urls, search_depth, search_breadth))
                st.session_state.search_extract_done = True
                st.success("Search and Extraction Complete.")
            except Exception as e:
                st.error(f"An error occurred during search and extraction: {e}")

if st.session_state.search_extract_done:
    st.subheader("Search Results")
    for topic, results in st.session_state.task.search_results.items():
        st.write(f"**Search for: {topic}**")
        if results:
            for res in results:
                st.write(f"- [{res.get('title', 'No Title')}]({res.get('url', '#')}) (Relevance: {res.get('score', 'N/A')})")
        else:
            st.write("No search results found.")

    st.subheader("Extracted Content")
    for topic, items in st.session_state.task.extracted_content.items():
        st.write(f"**Extraction for: {topic}**")
        if items:
            for item in items:
                url = item.get("url", "No URL")
                text = item.get("text") or item.get("raw_content", "")
                with st.expander(f"Content from {url}"):
                    st.write(text)
        else:
            st.write("No content extracted.")


if st.session_state.search_extract_done:
    if st.button("Perform RAG Analysis and Generate Reports", key="rag_button"):
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

if st.session_state.rag_analysis_done:
    st.subheader("Generated Reports")
    if st.session_state.full_report_content:
        with st.expander("Full Diagnostic Report"):
            st.markdown(st.session_state.full_report_content)
        st.download_button("Download Full Report (Markdown)", st.session_state.full_report_content, file_name=f"{st.session_state.task.original_query.replace(' ', '_')}_diagnostic_full.md")

    if st.session_state.summary_report_content:
        with st.expander("Summary Diagnostic Report"):
            st.markdown(st.session_state.summary_report_content)
        st.download_button("Download Summary Report (Markdown)", st.session_state.summary_report_content, file_name=f"{st.session_state.task.original_query.replace(' ', '_')}_diagnostic_summary.md")

    if st.button("Generate Patient-Friendly Summary", key="patient_summary_button"):
         if st.session_state.full_report_content:
             with st.spinner("Generating patient-friendly summary..."):
                 try:
                     # Pass the gemini_client for patient summary generation
                     patient_report = asyncio.run(st.session_state.task.generate_patient_summary_report(gemini_client, st.session_state.full_report_content, [])) # Citations might not be needed for prompt
                     st.session_state.patient_report_content = patient_report
                     st.session_state.patient_report_saved = True
                     st.success("Patient-Friendly Summary Generated.")
                 except Exception as e:
                    st.error(f"An error occurred during patient summary generation: {e}")
         else:
            st.warning("Please generate the full report first.")


if st.session_state.patient_report_saved:
    st.subheader("Patient-Friendly Summary")
    if st.session_state.patient_report_content:
         st.markdown(st.session_state.patient_report_content)
         st.download_button("Download Patient Summary (Markdown)", st.session_state.patient_report_content, file_name=f"{st.session_state.task.original_query.replace(' ', '_')}_patient_summary.md")

# --- Visualizations ---
st.subheader("Query Trend Visualizations")
if st.button("Generate Visualizations", key="viz_button"):
     with st.spinner("Generating visualizations from query history..."):
         try:
             # Modify visualize_query_trends to return figures or use st.pyplot directly
             # For now, it saves files, so we'll inform the user
             visualize_query_trends()
             st.info("Visualizations generated and saved as PNG files in the application directory.")
             # **Future Improvement:** Adapt visualize_query_trends to return matplotlib figures and display them with st.pyplot()
             # Example: fig = visualize_query_trends_streamlit()
             # st.pyplot(fig)
         except Exception as e:
            st.error(f"An error occurred during visualization generation: {e}")
