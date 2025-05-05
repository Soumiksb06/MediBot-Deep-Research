# streamlit_app.py
import streamlit as st
import os
import sys
import asyncio
import logging
from datetime import datetime
import pandas as pd
import base64 # To handle file downloads

# --- Add project root to sys.path if needed ---
# Assuming med_agent_new.py is in the same directory or a reachable path
# If med_agent_new.py is in a subdirectory 'scripts', use:
# script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
# sys.path.insert(0, script_dir)
# Ensure med_agent_new.py and its dependencies (like web_agent, subject_analyzer) are importable

# --- Attempt to import from med_agent_new ---
try:
    # Note: Direct import might trigger top-level code in med_agent_new.
    # It's better if med_agent_new.py is structured with functions/classes
    # and the main execution logic is within an if __name__ == "__main__": block.
    from med_agent_new import (
        MedicalTask,
        GeminiClient, # Make sure GeminiClient is importable
        TavilyClient,
        TavilyExtractor,
        WebSearchService,
        SubjectAnalyzer,
        SearchConfig,
        AnalysisConfig,
        extract_text_from_file,
        analyze_sentiment,         # Keep these as they are used in the original script
        extract_medical_entities, # Keep these as they are used in the original script
        train_symptom_classifier,
        predict_diagnosis,
        explain_diagnosis,
        save_query_history,
        visualize_query_trends,
        read_wearable_data,
        # Ensure Console is handled or replaced within imported functions for Streamlit
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"Error importing from med_agent_new.py: {e}")
    st.error("Please ensure med_agent_new.py and its dependencies are in the Python path and structured for import.")
    IMPORT_SUCCESS = False
except Exception as e:
    st.error(f"An unexpected error occurred during import: {e}")
    IMPORT_SUCCESS = False

# --- Configure Logging (Optional for Streamlit, useful for debugging) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Medical Agent Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Function for Downloads ---
def get_download_link(file_path, link_text):
    """Generates a link to download a file."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
        return href
    except FileNotFoundError:
        return f"{link_text} (File not found)"
    except Exception as e:
        return f"Error generating link for {link_text}: {e}"

# --- Streamlit UI ---
st.title("🩺 Medical Agent Assistant")
st.markdown("Describe your medical symptoms or issue, and the agent will analyze, research, and generate reports.")

# --- Initialize Session State ---
if 'task' not in st.session_state:
    st.session_state.task = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'search_extract_complete' not in st.session_state:
    st.session_state.search_extract_complete = False
if 'rag_complete' not in st.session_state:
    st.session_state.rag_complete = False
if 'patient_summary_complete' not in st.session_state:
    st.session_state.patient_summary_complete = False
if 'comprehensive_report' not in st.session_state:
    st.session_state.comprehensive_report = ""
if 'citations' not in st.session_state:
    st.session_state.citations = []
if 'patient_report' not in st.session_state:
    st.session_state.patient_report = ""
if 'additional_files_content' not in st.session_state:
    st.session_state.additional_files_content = {}
if 'full_report_md' not in st.session_state:
    st.session_state.full_report_md = ""
if 'summary_report_md' not in st.session_state:
    st.session_state.summary_report_md = ""
if 'patient_report_md' not in st.session_state:
    st.session_state.patient_report_md = ""
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

# --- Sidebar for Inputs and Configuration ---
with st.sidebar:
    st.header("Configuration")

    # --- API Keys ---
    st.subheader("API Keys")
    # Load from Streamlit secrets if available, otherwise use text input
    default_gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    default_tavily_key = st.secrets.get("TAVILY_API_KEY", "")
    default_supabase_url = st.secrets.get("SUPABASE_URL", "")
    default_supabase_key = st.secrets.get("SUPABASE_KEY", "")

    gemini_api_key = st.text_input("Gemini API Key", type="password", value=default_gemini_key)
    tavily_api_key = st.text_input("Tavily API Key", type="password", value=default_tavily_key)
    supabase_url = st.text_input("Supabase URL", value=default_supabase_url)
    supabase_key = st.text_input("Supabase Key", type="password", value=default_supabase_key)

    # Set environment variables for the imported script - **Important**
    # This is one way; alternatively, modify the script to accept keys as arguments
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    os.environ["SUPABASE_URL"] = supabase_url
    os.environ["SUPABASE_KEY"] = supabase_key
    os.environ["GOOGLE_EMBEDDING_MODEL"] = st.secrets.get("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
    os.environ["GEMINI_MODEL_NAME"] = st.secrets.get("GEMINI_MODEL_NAME", "gemini-1.5-flash") # Adjust model as needed
    os.environ["USE_FAISS"] = str(st.secrets.get("USE_FAISS", False)) # Control FAISS usage via secrets

    # --- Search Parameters ---
    st.subheader("Search Parameters")
    include_urls_input = st.text_area("Include Specific URLs (comma-separated, one per line)", height=100)
    omit_urls_input = st.text_area("Omit Specific URLs (comma-separated, one per line)", height=100)
    search_depth = st.selectbox("Search Depth", ["advanced", "basic"], index=0)
    search_breadth = st.number_input("Search Breadth (Results per query)", min_value=1, max_value=50, value=10)

    # --- File Uploads ---
    st.subheader("Reference Files")
    uploaded_files = st.file_uploader(
        "Upload Local Files (PDF, DOCX, TXT, CSV, XLSX)",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True
    )

    # --- Wearable Data ---
    st.subheader("Wearable Data")
    uploaded_wearable_file = st.file_uploader(
        "Upload Wearable Data (CSV, optional, e.g., 'wearable_data.csv')",
        type=["csv"]
    )

    # --- Classifier Data ---
    st.subheader("Symptom Classifier Data")
    uploaded_survey_file = st.file_uploader(
        "Upload Survey Data for Classifier (CSV, required, e.g., 'Survey report_Sheet1.csv')",
        type=["csv"]
    )

# --- Main Area ---
if not IMPORT_SUCCESS:
    st.warning("App cannot function without successful import from `med_agent_new.py`.")
    st.stop() # Stop execution if import failed

# --- Instantiate Clients (only if keys are provided) ---
# Use placeholders or dummy clients if keys are missing? Or raise error?
# For now, assume keys are provided or script handles missing keys gracefully.
if gemini_api_key and tavily_api_key and supabase_url and supabase_key:
    try:
        search_config = SearchConfig()
        # Make sure model names are correctly fetched or set defaults
        gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        analysis_config = AnalysisConfig(model_name=gemini_model_name, temperature=0.3)

        gemini_client = GeminiClient(api_key=gemini_api_key, config=analysis_config)
        search_client = TavilyClient(api_key=tavily_api_key)
        extractor = TavilyExtractor(api_key=tavily_api_key)
        search_service = WebSearchService(search_client, search_config)
        subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config)
        CLIENTS_INITIALIZED = True
    except Exception as e:
        st.error(f"Error initializing API clients: {e}")
        CLIENTS_INITIALIZED = False
else:
    st.warning("Please provide all API keys in the sidebar to proceed.")
    CLIENTS_INITIALIZED = False

# --- Step 1: Initial Query and Pre-analysis ---
st.header("1. Patient Query")
initial_query = st.text_area("Describe the medical symptoms or issue:", height=150, key="initial_query_input")

if st.button("Start Analysis", disabled=not CLIENTS_INITIALIZED or not initial_query):
    if not uploaded_survey_file:
         st.warning("Please upload the Survey Data CSV in the sidebar for diagnosis prediction.")
    else:
        # Reset state for new analysis
        st.session_state.task = None
        st.session_state.analysis_complete = False
        st.session_state.search_extract_complete = False
        st.session_state.rag_complete = False
        st.session_state.patient_summary_complete = False
        st.session_state.feedback_history = []
        st.session_state.additional_files_content = {}
        st.session_state.current_query = initial_query

        # --- Handle File Uploads ---
        # Save uploaded files temporarily to be read by extract_text_from_file
        temp_dir = "temp_streamlit_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        st.session_state.additional_files_content = {}
        if uploaded_files:
            st.info(f"Processing {len(uploaded_files)} uploaded reference file(s)...")
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Use a dummy console object or modify extract_text_from_file
                class DummyConsole:
                    def print(self, *args, **kwargs):
                        pass # Or log to streamlit/logging
                console = DummyConsole()
                content = extract_text_from_file(file_path, console) # Requires PyPDF2, python-docx, pandas
                if content:
                    st.session_state.additional_files_content[uploaded_file.name] = content
                else:
                    st.warning(f"Could not extract text from {uploaded_file.name}")

        # Handle wearable data file
        if uploaded_wearable_file:
            wearable_file_path = os.path.join(temp_dir, "wearable_data.csv") # Force specific name if needed by script
            with open(wearable_file_path, "wb") as f:
                f.write(uploaded_wearable_file.getbuffer())
            st.info("Wearable data file saved for processing.")

        # Handle survey data file (save it with the expected name)
        survey_file_path = os.path.join(temp_dir, "Survey report_Sheet1.csv") # Expected name
        with open(survey_file_path, "wb") as f:
             f.write(uploaded_survey_file.getbuffer())
        st.info("Survey data file saved for classifier training.")
        # Change working directory temporarily? Or modify script to accept path?
        original_cwd = os.getcwd()
        os.chdir(temp_dir) # Change CWD so the script finds the file

        with st.spinner("Performing initial analysis (Sentiment, NER, Diagnosis Prediction)..."):
            try:
                # --- Sentiment Analysis ---
                sentiment_score = analyze_sentiment(initial_query) # Requires nltk
                st.session_state.sentiment = sentiment_score

                # --- NER ---
                entities = extract_medical_entities(initial_query) # Requires spacy
                st.session_state.entities = entities

                # --- Diagnosis Prediction ---
                # Train classifier (reads 'Survey report_Sheet1.csv' from current dir)
                vectorizer, clf_model = train_symptom_classifier() # Requires scikit-learn, pandas

                if vectorizer and clf_model:
                    predicted_diag, diag_proba = predict_diagnosis(initial_query, vectorizer, clf_model)
                    explanation = explain_diagnosis(initial_query, vectorizer, clf_model) # Requires shap
                    st.session_state.predicted_diag = predicted_diag
                    st.session_state.diag_proba = diag_proba
                    st.session_state.explanation = explanation
                    # Save history (writes 'query_history.csv' to current dir)
                    save_query_history(initial_query, predicted_diag, sentiment_score, entities)
                else:
                    st.warning("Could not train/load symptom classifier. Diagnosis prediction skipped.")
                    st.session_state.predicted_diag = "N/A"
                    st.session_state.diag_proba = {}
                    st.session_state.explanation = "N/A"

                # --- Create MedicalTask ---
                 # Use a dummy console or adapt MedicalTask
                class StreamlitConsole:
                    def print(self, message="", **kwargs):
                        if "[red]" in message:
                            st.error(message.replace("[red]", "").replace("[/red]", ""))
                        elif "[yellow]" in message:
                            st.warning(message.replace("[yellow]", "").replace("[/yellow]", ""))
                        elif "[green]" in message:
                            st.success(message.replace("[green]", "").replace("[/green]", ""))
                        elif "[bold green]" in message:
                             st.success(message.replace("[bold green]", "").replace("[/bold green]", "")) # Make bold text normal success
                        else:
                            st.info(str(message)) # Display other messages as info

                st_console = StreamlitConsole()
                st.session_state.task = MedicalTask(initial_query, st_console)
                st.session_state.task.feedback_history = st.session_state.feedback_history # Link histories

                # --- Initial Subject Analysis ---
                # Requires running async code
                async def run_analysis():
                    current_date = datetime.today().strftime("%Y-%m-%d")
                    await st.session_state.task.analyze(subject_analyzer, current_date)

                # Run the async function using asyncio.run (might block)
                asyncio.run(run_analysis())
                st.session_state.analysis_complete = True
                st.rerun() # Rerun to update UI based on new state

            except FileNotFoundError as fnf_error:
                 st.error(f"File not found during initial analysis: {fnf_error}. Ensure 'Survey report_Sheet1.csv' was uploaded and accessible.")
            except ImportError as imp_error:
                 st.error(f"Missing package dependency: {imp_error}. Please install all required packages.")
            except Exception as e:
                st.error(f"An error occurred during initial analysis: {e}")
            finally:
                 os.chdir(original_cwd) # Change back CWD

# --- Display Initial Analysis Results ---
if st.session_state.analysis_complete:
    st.subheader("Preliminary Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sentiment (Compound)", f"{st.session_state.sentiment.get('compound', 0):.2f}")
        st.write(f"Pos: {st.session_state.sentiment.get('pos', 0):.2f}, Neu: {st.session_state.sentiment.get('neu', 0):.2f}, Neg: {st.session_state.sentiment.get('neg', 0):.2f}")
    with col2:
        st.write("**Medical Entities:**")
        st.write(f"{', '.join(st.session_state.entities) if st.session_state.entities else 'None found'}")
    with col3:
        st.write("**Predicted Diagnosis:**")
        st.write(f"{st.session_state.predicted_diag}")
        # Optional: Show probabilities
        # with st.expander("Show Probabilities"):
        #    st.write(st.session_state.diag_proba)
        with st.expander("Show Explanation"):
           st.text(st.session_state.explanation) # Display explanation as preformatted text

    st.subheader("Agent's Understanding (Subject Analysis)")
    analysis_data = st.session_state.task.analysis
    st.write(f"**Patient Query:** {st.session_state.task.original_query}")
    st.write(f"**Identified Medical Issue:** {analysis_data.get('main_subject', 'Unknown Issue')}")
    temporal = analysis_data.get("temporal_context", {})
    if temporal:
        for key, value in temporal.items():
            st.write(f"- **{key.capitalize()}:** {value}")
    needs = analysis_data.get("What_needs_to_be_researched", [])
    st.write(f"**Key aspects to investigate:** {', '.join(needs) if needs else 'None'}")

    # --- Optional Feedback Loop ---
    st.subheader("Refine Analysis (Optional)")
    feedback = st.text_area("Provide feedback on the agent's analysis to refine the query (leave blank if correct):", key="feedback_input")
    if st.button("Re-analyze with Feedback", disabled=not feedback):
         with st.spinner("Re-analyzing with feedback..."):
            st.session_state.task.update_feedback(feedback)
            st.session_state.task.current_query = f"{st.session_state.task.original_query} - Additional context: {feedback}"
            st.session_state.current_query = st.session_state.task.current_query # Update global state too

            # Rerun analysis
            async def run_analysis_feedback():
                current_date = datetime.today().strftime("%Y-%m-%d")
                await st.session_state.task.analyze(subject_analyzer, current_date)

            try:
                asyncio.run(run_analysis_feedback())
                st.session_state.feedback_history = st.session_state.task.feedback_history # Update history
                st.success("Re-analysis complete.")
                st.rerun() # Update UI
            except Exception as e:
                 st.error(f"Error during re-analysis: {e}")


# --- Step 2: Search and Extract ---
st.header("2. Web Search & Content Extraction")
if st.session_state.analysis_complete:
    if st.button("Perform Search & Extraction", disabled=st.session_state.search_extract_complete):
        with st.spinner("Searching the web and extracting content... This may take a while."):
            try:
                # Prepare URLs
                include_urls = [url.strip() for url in include_urls_input.split(',') if url.strip()]
                omit_urls = [url.strip() for url in omit_urls_input.split(',') if url.strip()]

                async def run_search_extract():
                    if not include_urls:
                        await st.session_state.task.search_and_extract(
                            search_service, extractor, omit_urls, search_depth, search_breadth
                        )
                    else:
                        # Handle user-provided URLs (logic from original script)
                        filtered_urls = [url for url in include_urls if not any(omit.lower() in url.lower() for omit in omit_urls)]
                        st.session_state.task.search_results = {"User Provided": [{"title": "User Provided", "url": url, "content": ""} for url in filtered_urls]}
                        st.info(f"Using user provided URLs: {filtered_urls}")
                        # Use TavilyExtractor directly
                        extraction_response = extractor.extract(
                             urls=filtered_urls,
                             extract_depth="advanced", # Or make configurable
                             include_images=False
                         )
                        st.session_state.task.extracted_content["User Provided"] = extraction_response.get("results", [])

                # Run async function
                asyncio.run(run_search_extract())
                st.session_state.search_extract_complete = True
                st.success("Search and extraction complete.")
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during search/extraction: {e}")

# --- Display Search/Extraction Results ---
if st.session_state.search_extract_complete:
    st.subheader("Search Results")
    if st.session_state.task.search_results:
        for topic, results in st.session_state.task.search_results.items():
            with st.expander(f"Search Topic: {topic} ({len(results)} results)"):
                if results:
                    for res in results:
                        st.write(f"**Title:** {res.get('title', 'N/A')}")
                        st.write(f"**URL:** {res.get('url', 'N/A')}")
                        st.write(f"**Relevance:** {res.get('score', 'N/A')}")
                        st.divider()
                else:
                    st.write("No results found for this topic.")
    else:
        st.write("No search results available.")

    st.subheader("Extracted Content Snippets")
    if st.session_state.task.extracted_content:
        for topic, items in st.session_state.task.extracted_content.items():
             with st.expander(f"Extracted Content for: {topic} ({len(items)} items)"):
                 if items:
                     for item in items:
                         st.write(f"**URL:** {item.get('url', 'N/A')}")
                         content_snippet = item.get('text') or item.get('raw_content', '')
                         st.text_area("Content Snippet:", value=content_snippet[:1000] + "..." if len(content_snippet)>1000 else content_snippet , height=150, disabled=True, key=f"extract_{item.get('url', topic)}_{items.index(item)}") # Add unique key
                         if item.get("error"):
                             st.warning(f"Extraction Error: {item.get('error')}")
                         st.divider()
                 else:
                     st.write("No content extracted for this topic.")
    else:
        st.write("No extracted content available.")


# --- Step 3: RAG Analysis & Report Generation ---
st.header("3. Comprehensive Analysis & Report Generation")
if st.session_state.search_extract_complete:
    if st.button("Generate Diagnostic Reports", disabled=st.session_state.rag_complete):
        with st.spinner("Performing RAG analysis and generating comprehensive report... This may take significant time and resources."):
            try:
                # --- Run RAG Analysis ---
                async def run_rag():
                    # Pass the already initialized gemini_client
                    report, citations = await st.session_state.task.analyze_full_content_rag(gemini_client)
                    return report, citations

                st.session_state.comprehensive_report, st.session_state.citations = asyncio.run(run_rag())

                # --- Generate Markdown Reports ---
                # Get additional file content prepared earlier
                additional_files = st.session_state.additional_files_content
                st.session_state.full_report_md = st.session_state.task.generate_report(
                    additional_files, st.session_state.comprehensive_report, st.session_state.citations
                )
                st.session_state.summary_report_md = st.session_state.task.generate_summary_report(
                    st.session_state.comprehensive_report, st.session_state.citations
                )

                st.session_state.rag_complete = True
                st.success("Comprehensive RAG analysis and report generation complete.")
                st.rerun()
            except ImportError as imp_error:
                 st.error(f"Missing package for RAG: {imp_error}. Check faiss-cpu, numpy, supabase client.")
            except Exception as e:
                st.error(f"An error occurred during RAG analysis: {e}")
                # Attempt to clean up Supabase embeddings on failure if possible
                # Note: This requires the Supabase client to be initialized
                try:
                    from supabase import create_client, Client
                    if supabase_url and supabase_key:
                        supabase: Client = create_client(supabase_url, supabase_key)
                        supabase.table("embeddings").delete().eq("source", "Aggregated content").execute()
                        st.warning("Attempted to clear Supabase embeddings after RAG failure.")
                except Exception as cleanup_err:
                    st.error(f"Error during Supabase cleanup after RAG failure: {cleanup_err}")


# --- Display Reports ---
if st.session_state.rag_complete:
    st.subheader("Generated Reports")
    tab1, tab2 = st.tabs(["Comprehensive Report", "Summary Report"])

    with tab1:
        st.markdown(st.session_state.full_report_md)
        # Add download button for the full report
        st.download_button(
            label="Download Full Report (Markdown)",
            data=st.session_state.full_report_md,
            file_name=f"{st.session_state.task.original_query[:30].replace(' ','_')}_diagnostic_full.md",
            mime="text/markdown",
        )

    with tab2:
        st.markdown(st.session_state.summary_report_md)
         # Add download button for the summary report
        st.download_button(
            label="Download Summary Report (Markdown)",
            data=st.session_state.summary_report_md,
            file_name=f"{st.session_state.task.original_query[:30].replace(' ','_')}_diagnostic_summary.md",
            mime="text/markdown",
        )


# --- Step 4: Patient-Friendly Summary ---
st.header("4. Patient-Friendly Summary")
if st.session_state.rag_complete:
     if st.button("Generate Patient-Friendly Summary", disabled=st.session_state.patient_summary_complete):
         with st.spinner("Generating patient-friendly summary..."):
             try:
                 async def run_patient_summary():
                     # Pass the already initialized gemini_client
                     report = await st.session_state.task.generate_patient_summary_report(
                         gemini_client, st.session_state.comprehensive_report, st.session_state.citations
                     )
                     return report

                 st.session_state.patient_report = asyncio.run(run_patient_summary())
                 st.session_state.patient_report_md = st.session_state.patient_report # Assuming it's already markdown
                 st.session_state.patient_summary_complete = True
                 st.success("Patient-friendly summary generated.")
                 st.rerun()
             except Exception as e:
                 st.error(f"An error occurred generating the patient summary: {e}")

# --- Display Patient Summary ---
if st.session_state.patient_summary_complete:
    st.subheader("Patient-Friendly Summary")
    st.markdown(st.session_state.patient_report_md)
    # Add download button
    st.download_button(
        label="Download Patient Summary (Markdown)",
        data=st.session_state.patient_report_md,
        file_name=f"{st.session_state.task.original_query[:30].replace(' ','_')}_patient_summary.md",
        mime="text/markdown",
    )


# --- Step 5: Visualize Trends (Optional) ---
st.header("5. Query Trends Visualization (Optional)")
if st.session_state.analysis_complete: # Can be shown after initial analysis
    if st.button("Generate Trend Visualizations"):
        with st.spinner("Generating visualizations from query history..."):
            # Ensure query_history.csv exists (it's created in the temp dir during analysis)
            temp_dir = "temp_streamlit_uploads"
            history_file = os.path.join(temp_dir, "query_history.csv")
            plot_dir = os.path.join(temp_dir, "plots") # Save plots in a sub-directory
            os.makedirs(plot_dir, exist_ok=True)

            original_cwd = os.getcwd()
            try:
                 # Change CWD so visualize_query_trends finds the history file and saves plots
                 os.chdir(temp_dir)
                 visualize_query_trends() # Requires matplotlib, seaborn, pandas. Saves PNGs locally.
                 st.success("Trend visualization plots generated.")

                 # Find generated plots and display them
                 plot_files = [f for f in os.listdir(".") if f.endswith(".png")] # Look in current (temp) dir
                 if plot_files:
                     st.subheader("Query Trend Plots")
                     for plot_file in plot_files:
                         try:
                              st.image(plot_file, caption=plot_file.replace('_', ' ').replace('.png', '').title())
                              # Provide download link for the plot
                              # Need to read the image file to provide data to st.download_button
                              with open(plot_file, "rb") as pf:
                                   st.download_button(
                                        label=f"Download {plot_file}",
                                        data=pf.read(),
                                        file_name=plot_file,
                                        mime="image/png"
                                   )
                         except Exception as img_e:
                              st.warning(f"Could not display or provide download for {plot_file}: {img_e}")
                 else:
                      st.info("No plot files found. Ensure 'query_history.csv' exists and contains data.")

            except FileNotFoundError:
                 st.error("Could not find 'query_history.csv'. Was an initial analysis run?")
            except ImportError as viz_imp_error:
                 st.error(f"Missing package for visualization: {viz_imp_error}. Install matplotlib, seaborn.")
            except Exception as e:
                 st.error(f"An error occurred during visualization: {e}")
            finally:
                 os.chdir(original_cwd) # Change back CWD


# --- Add Footer or other info ---
st.divider()
st.caption("Medical Agent Assistant v1.0")
# --- End of App ---
