import streamlit as st
import asyncio
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import base64
import csv

import nltk
# Check if vader_lexicon is already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
     # Fallback/alternative check for missing data, handle LookupError
     # This can still be problematic if DownloadError is truly missing.
     # A better approach is below using LookupError
     try:
         nltk.download('vader_lexicon', quiet=True)
     except Exception as e:
         st.warning(f"Could not download NLTK vader_lexicon: {e}")

except LookupError:
    # If LookupError is raised, the data is not found, so download it
    st.spinner("Downloading NLTK vader_lexicon...")
    try:
        nltk.download('vader_lexicon', quiet=True)
        st.success("Downloaded NLTK vader_lexicon.")
    except Exception as e:
        st.warning(f"Could not download NLTK vader_lexicon: {e}")

# You might also want to add a similar check for the spaCy model download
# which is handled within the extract_medical_entities function in the current script.

import spacy
try:
    spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")

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

# Import agent modules (assuming they are in the same directory or correctly in the path)
# If your module structure is different, you may need to adjust these imports
try:
    from web_agent.src.services.web_search import WebSearchService
    from web_agent.src.models.search_models import SearchConfig
    from subject_analyzer.src.services.tavily_client import TavilyClient
    from subject_analyzer.src.services.tavily_extractor import TavilyExtractor
    from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer
    from subject_analyzer.src.services.gemini_client import GeminiClient
    from subject_analyzer.src.models.analysis_models import AnalysisConfig
    # Assuming other necessary imports like PyPDF2, docx, etc. are handled within functions or installed
except ImportError as e:
    st.error(f"Failed to import agent modules. Make sure your module structure is correct and required libraries are installed. Error: {e}")
    # st.stop() # Don't stop the app immediately, let the user see the error

# Suppress matplotlib and seaborn warnings
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module="seaborn")


# --- Helper Functions (adapted for Streamlit) ---

def extract_text_from_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        # Move imports here to avoid issues if libraries are not installed globally
        if ext == '.pdf':
            import PyPDF2
            text = ""
            # Use uploaded_file as a file-like object
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif ext == '.docx':
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join(para.text for para in doc.paragraphs)
        elif ext in ['.csv']:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            return df.to_csv(index=False)
        elif ext in ['.xls', '.xlsx']:
            import pandas as pd
            df = pd.read_excel(uploaded_file)
            return df.to_csv(index=False)
        else:
            # Decode the bytes content of the uploaded file
            return uploaded_file.getvalue().decode("utf-8")
    except ImportError as e:
         st.error(f"Required library for file type '{ext}' not found: {e}. Please install it.")
         return ""
    except Exception as ex:
        st.warning(f"Could not extract text from {uploaded_file.name}: {ex}")
        logging.error(f"File extraction error for {uploaded_file.name}: {ex}")
        return ""

# Placeholder implementations if they are not provided elsewhere and needed for the streamlit app
def read_wearable_data():
    """Placeholder for reading wearable device data."""
    # Implement your logic to read and summarize wearable data from a file or database
    # For a Streamlit app, you might add a file uploader for wearable data CSV
    wearable_data_file = st.sidebar.file_uploader("Upload Wearable Data (CSV)", type=["csv"])
    if wearable_data_file:
        try:
            import pandas as pd
            df = pd.read_csv(wearable_data_file)
            summary = "Wearable Data Summary:\n"
            if "heart_rate" in df.columns:
                avg_hr = df["heart_rate"].mean()
                summary += f"Average Heart Rate: {avg_hr:.1f}\n"
            if "steps" in df.columns:
                total_steps = df["steps"].sum()
                summary += f"Total Steps: {total_steps}\n"
            # Add other relevant summaries
            return summary
        except ImportError:
            st.error("pandas is required for reading wearable data. Install with 'pip install pandas'.")
            return None
        except Exception as e:
            st.error(f"Error reading wearable data: {e}")
            return None
    return None

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
        # Log to the server logs, not st.error here as it might be called in a loop
        logging.error("FAISS or Numpy not installed. Cannot build FAISS index.")
        return None
    except Exception as e:
        logging.error(f"Error building FAISS index: {e}")
        return None

# Assuming search_faiss is also needed and defined elsewhere if USE_FAISS is true


# --- MedicalTask Class (adapted for Streamlit output) ---

class MedicalTask:
    def __init__(self, query):
        self.original_query = query
        self.current_query = query
        self.analysis = {}
        self.search_results = {}
        self.extracted_content = {}
        self.feedback_history = []
        self.report = "" # To store the full report content
        self.summary_report = "" # To store the summary report content
        self.patient_report = "" # To store the patient report content

    def update_feedback(self, feedback):
        self.feedback_history.append({
            "query": self.current_query,
            "feedback": feedback,
            "time": datetime.now().isoformat()
        })

    async def analyze(self, subject_analyzer, current_date: str):
        st.info(f"Analyzing patient query for diagnosis (as of {current_date})...")
        try:
            self.analysis = subject_analyzer.analyze(f"{self.current_query} (as of {current_date})")
            logging.info("Subject analysis successful.")
        except Exception as e:
            logging.error(f"Subject analysis failed: {e}")
            st.error(f"Subject analysis failed: {e}")
            raise e

        st.subheader("Agent's Understanding:")
        st.write(f"**Patient Query:** {self.original_query}")
        st.write(f"**Identified Medical Issue:** {self.analysis.get('main_subject', 'Unknown Issue')}")
        temporal = self.analysis.get("temporal_context", {})
        if temporal:
            st.write("**Temporal Context:**")
            for key, value in temporal.items():
                st.write(f"- {key.capitalize()}**: {value}")
        else:
            st.write("No temporal context provided.")
        needs = self.analysis.get("What_needs_to_be_researched", [])
        st.write("**Key aspects to investigate:**")
        st.write(f"{', '.join(needs) if needs else 'None'}")


    async def search_and_extract(self, search_service, extractor, omit_urls, search_depth, search_breadth, include_urls):
        topics = [self.analysis.get("main_subject", self.current_query)]
        topics += self.analysis.get("What_needs_to_be_researched", [])

        if not include_urls:
            st.info("Searching the web for medical information...")
            for topic in topics:
                if not topic:
                    continue
                st.write(f"Searching for information on: **{topic}**")
                detailed_query = f"{topic} medical diagnosis (Depth: {search_depth}, Breadth: {search_breadth})"
                st.write(f"Executing search for: {detailed_query}")
                try:
                    response = search_service.search_subject(
                        topic, "medical", search_depth=search_depth, results=search_breadth
                    )
                    results = response.get("results", [])
                    results = [res for res in results if res.get("url") and not any(
                        omit.lower() in res.get("url").lower() for omit in omit_urls)]
                    self.search_results[topic] = results
                    st.write(f"Found {len(results)} results for {topic}")
                    for res in results:
                        st.write(f"- [{res.get('title', 'No Title')}]({res.get('url', '#')}) (Relevance: {res.get('score', 'N/A')})")
                except Exception as e:
                    st.error(f"Search failed for {topic}: {e}")
                    self.search_results[topic] = []

                st.info(f"Extracting content for: **{topic}**...")
                try:
                    urls = [res.get("url") for res in self.search_results[topic] if res.get("url")]
                    extraction_response = extractor.extract(
                        urls=urls,
                        extract_depth="advanced",
                        include_images=False
                    )
                    extracted = extraction_response.get("results", [])
                    self.extracted_content[topic] = extracted
                    st.write(f"Extracted content from {len(extracted)} URLs for {topic}.")
                    failed = [res for res in extracted if res.get("error")]
                    if failed:
                        st.warning(f"Warning: Failed to extract {len(failed)} URLs for {topic}.")
                except Exception as e:
                    st.error(f"Extraction failed for {topic}: {e}")
        else:
            st.info("Using user provided URLs for content extraction...")
            filtered_urls = [url.strip() for url in include_urls if url.strip() and not any(omit.lower() in url.lower() for omit in omit_urls)]
            self.search_results["User Provided"] = [{"title": "User Provided", "url": url} for url in filtered_urls]
            st.write(f"Using user provided URLs: {filtered_urls}")
            if filtered_urls:
                try:
                    extraction_response = extractor.extract(
                        urls=filtered_urls,
                        extract_depth="advanced",
                        include_images=False
                    )
                    self.extracted_content["User Provided"] = extraction_response.get("results", [])
                    st.write(f"Extracted content from {len(self.extracted_content.get('User Provided', []))} user provided URLs.")
                except Exception as e:
                    st.error(f"Extraction failed for user provided URLs: {e}")
            else:
                st.warning("No valid URLs provided for inclusion after filtering.")


    async def analyze_full_content_rag(self, gemini_client):
        st.info("Aggregating extracted content for comprehensive diagnostic analysis...")

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
            # Ensure text is a string
            if not isinstance(text, str):
                logging.warning(f"Expected string for chunking, but got {type(text)}")
                return []
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


        chunks = chunk_text(full_content, chunk_size=765)
        st.write(f"Total chunks generated: {len(chunks)}")

        if not chunks:
            st.warning("No content chunks generated for RAG.")
            return "No relevant content found for analysis.", []

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
                try:
                    from google.genai import types
                except ImportError:
                    st.error("google.genai.types not available. Cannot perform embedding.")
                    raise

                # Ensure text is not empty or None
                if not text or not isinstance(text, str):
                     logging.warning("Skipping embedding for empty or invalid text.")
                     # Return a zero vector or handle as an error
                     # For now, raise an error that can be caught
                     raise ValueError("Cannot embed empty or invalid text.")


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
                st.warning(f"Embedding error: {e}")
                # Re-raise the exception if you want it to be caught by the executor
                raise

        # === End Modified Embedding Section ===

        # Get the current running loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError: # Handle the case where no loop is running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run synchronous get_embedding calls in a thread pool executor
        # Filter out any empty chunks before attempting to embed
        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        if not valid_chunks:
             st.warning("No valid chunks to embed after cleaning.")
             return "No relevant content found for analysis.", []

        st.info(f"Generating embeddings for {len(valid_chunks)} valid chunks...")
        chunk_embeddings = []
        # Process embeddings in batches or sequentially to avoid overwhelming the API
        # Using run_in_executor with asyncio.gather for potential concurrency
        try:
             chunk_embeddings = await asyncio.gather(*[loop.run_in_executor(None, get_embedding, chunk) for chunk in valid_chunks], return_exceptions=True)
             # Handle any exceptions returned by run_in_executor
             failed_embeddings = [i for i, result in enumerate(chunk_embeddings) if isinstance(result, Exception)]
             if failed_embeddings:
                 st.warning(f"Failed to generate embeddings for {len(failed_embeddings)} chunks.")
                 # Filter out the exceptions, keep only valid embeddings
                 valid_chunk_embeddings = [result for result in chunk_embeddings if not isinstance(result, Exception)]
                 valid_chunks = [valid_chunks[i] for i in range(len(valid_chunks)) if i not in failed_embeddings]
                 chunk_embeddings = valid_chunk_embeddings
             if not chunk_embeddings:
                 st.error("No embeddings were successfully generated.")
                 return "Error: Could not generate embeddings for RAG.", citations

        except Exception as e:
             st.error(f"Failed to generate embeddings for chunks: {e}")
             return "Error: Could not generate embeddings for RAG.", citations


        # Ensure numpy is imported for FAISS if needed
        try:
            import numpy as np
        except ImportError:
             st.error("Numpy is required for FAISS. Install with 'pip install numpy'.")
             # Re-raise the exception to stop execution if numpy is critical
             raise
        except Exception as e:
             st.error(f"Error importing numpy: {e}")
             raise


        from supabase import create_client, Client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")

        if not SUPABASE_URL or not SUPABASE_KEY:
            st.warning("Supabase URL or Key not found in environment variables. Skipping Supabase operations.")
            # Fallback to using all chunks if Supabase is not configured
            aggregated_relevant = full_content
            st.info("Proceeding with RAG using all extracted content (Supabase not configured).")
        else:
            try:
                supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
                try:
                    # Use upsert for robustness instead of delete all
                    # Define a unique key or assume chunk content is unique enough for this purpose
                    # Or handle potential duplicates based on your Supabase setup
                    # For simplicity, let's try inserting and handle potential errors
                    pass # Skip explicit delete

                except Exception as e:
                    st.warning(f"Warning: Could not clear previous embeddings in Supabase: {e}")

                data_to_insert = []
                for i, chunk in enumerate(valid_chunks):
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
                             st.error(f"Error converting embedding vector to list for chunk {i}. Type: {type(embedding_vector)}")
                             # Decide how to handle chunks that fail embedding conversion
                             continue # Skip inserting this chunk

                    data_to_insert.append({
                        "chunk": chunk,
                        "embedding": embedding_vector, # Use the Google embedding vector
                        "source": "Aggregated content"
                    })

                if data_to_insert:
                    st.info(f"Inserting {len(data_to_insert)} embeddings into Supabase...")
                    batch_size = 200
                    for i in range(0, len(data_to_insert), batch_size):
                        batch = data_to_insert[i:i+batch_size]
                        try:
                            # Use insert and handle conflicts if chunk is not unique
                            supabase.table("embeddings").insert(batch).execute()
                            # st.write(f"Inserted batch {i // batch_size + 1} of {((len(data_to_insert) + batch_size - 1) // batch_size)}.") # Corrected total batch calculation
                        except Exception as e:
                             st.error(f"Error inserting batch {i // batch_size + 1} into Supabase: {e}")
                             # Decide whether to continue or stop on batch insertion errors
                else:
                    st.warning("No data to insert into Supabase.")


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
                    st.error(f"Failed to get embedding for summarization prompt: {e}")
                    # Decide how to proceed if query embedding fails - likely cannot perform RAG
                    return "Error: Could not generate query embedding for RAG.", citations


                USE_FAISS = os.getenv("USE_FAISS", "False").lower() == "true"
                if USE_FAISS:
                    # Use FAISS for matching
                    # build_faiss_index requires numpy and is defined at the top level
                    if chunk_embeddings:
                        faiss_index = build_faiss_index(chunk_embeddings)
                        if faiss_index is not None:
                            k = min(155, len(valid_chunks))
                            # Ensure query_embedding is in the correct format for FAISS search (numpy array float32)
                            try:
                                query_vec = np.array(query_embedding).reshape(1, -1).astype('float32')
                                distances, indices = faiss_index.search(query_vec, k)
                                # Ensure indices are valid for the current valid_chunks list
                                matched_chunks = [valid_chunks[i] for i in indices[0] if i < len(valid_chunks)]
                            except Exception as e:
                                 st.error(f"Error during FAISS search: {e}")
                                 st.warning("Falling back to Supabase for RAG.")
                                 # Ensure query_embedding is a list for Supabase RPC
                                 match_response = supabase.rpc("match_chunks", {"query_embedding": list(query_embedding), "match_count": 155}).execute() # Increased match_count for fallback
                                 matched_chunks = [row["chunk"] for row in match_response.data] if match_response.data else []
                        else:
                            st.warning("FAISS index could not be built; falling back to Supabase.")
                            # Ensure query_embedding is a list for Supabase RPC
                            match_response = supabase.rpc("match_chunks", {"query_embedding": list(query_embedding), "match_count": 155}).execute()
                            matched_chunks = [row["chunk"] for row in match_response.data] if match_response.data else []
                    else:
                         st.warning("No valid chunk embeddings to build FAISS index; falling back to Supabase (if configured).")
                         if SUPABASE_URL and SUPABASE_KEY:
                            match_response = supabase.rpc("match_chunks", {"query_embedding": list(query_embedding), "match_count": 200}).execute()
                            matched_chunks = [row["chunk"] for row in match_response.data] if match_response.data else []
                         else:
                            matched_chunks = [] # No embeddings or Supabase

                else:
                    # Ensure query_embedding is a list for Supabase RPC
                    # Use Supabase RPC for matching
                    match_response = supabase.rpc("match_chunks", {"query_embedding": list(query_embedding), "match_count": 200}).execute()
                    matched_chunks = [row["chunk"] for row in match_response.data] if match_response.data else []

                st.write(f"Retrieved {len(matched_chunks)} relevant chunks.")
                aggregated_relevant = "\n\n".join(matched_chunks)

            except Exception as e:
                st.error(f"Error during Supabase RAG process: {e}")
                st.warning("Proceeding with RAG using all extracted content (Supabase error).")
                aggregated_relevant = full_content


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

        st.info("Performing secondary analysis via RAG...")
        try:
            response = gemini_client.chat(messages)
            comprehensive_report = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            if SUPABASE_URL and SUPABASE_KEY:
                try:
                    # Optional: clear embeddings if you want a clean state for each query
                    # supabase.table("embeddings").delete().eq("source", "Aggregated content").execute()
                    # st.info("Cleared Supabase embeddings after report generation.")
                    pass # Keep embeddings for potential future use or analysis
                except Exception as e:
                    st.error(f"Error cleaning up Supabase: {e}")

            return comprehensive_report, citations
        except Exception as e:
            st.error(f"Secondary analysis via RAG failed: {e}")
            if SUPABASE_URL and SUPABASE_KEY:
                try:
                    # Optional: clear embeddings even on failure
                    # supabase.table("embeddings").delete().eq("source", "Aggregated content").execute()
                    # st.info("Cleared Supabase embeddings after failed attempt.")
                    pass
                except Exception as cleanup_err:
                    st.error(f"Error during cleanup: {cleanup_err}")
            return "Secondary analysis failed.", citations


    def generate_report(self, additional_files, comprehensive_report, citations):
        report = f"# Medical Diagnostic Report for: {self.original_query} (Generated on {datetime.today().strftime('%Y-%m-%d')})\n\n"
        report += f"**Refined Query:** {self.current_query}\n\n"
        report += "## Agent's Understanding\n"
        for key, value in self.analysis.items():
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

        wearable_summary = read_wearable_data() # Assuming read_wearable_data is defined elsewhere and adapted for Streamlit
        if wearable_summary:
            report += "## Wearable Data Summary\n"
            report += wearable_summary + "\n"

        report += "\n---\n\n"
        report += "## Comprehensive Diagnostic Report (via Secondary Analysis)\n"
        report += comprehensive_report + "\n\n"
        report += "## Citations\n"
        for citation in citations:
            report += f"- {citation}\n"

        self.report = report
        return report

    def generate_summary_report(self, comprehensive_report, citations):
        summary = f"# Summary Diagnostic Report for: {self.original_query} (Generated on {datetime.today().strftime('%Y-%m-%d')})\n\n"
        summary += f"**Refined Query:** {self.current_query}\n\n"
        summary += "## Comprehensive Diagnostic Report\n"
        summary += comprehensive_report + "\n\n"
        summary += "## Citations\n"
        for citation in citations:
            summary += f"- {citation}\n"
        self.summary_report = summary
        return summary

    async def generate_patient_summary_report(self, gemini_client, comprehensive_report):
        st.info("Generating patient-friendly summary report with clear action steps...")
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
            response = gemini_client.chat(messages)
            simplified_report = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.patient_report = simplified_report
            return simplified_report
        except Exception as e:
            st.error(f"Failed to generate patient-friendly summary: {e}")
            self.patient_report = "Patient-friendly summary generation failed."
            return self.patient_report


# --- Data Science / Analytics & Visualization Functions (adapted for Streamlit) ---

# Ensure these functions are defined and imported correctly if they are in separate files
# Placeholder implementations if they are not provided elsewhere
def train_symptom_classifier():
    """
    Trains a simple symptom-to-diagnosis classifier using the survey data.
    Adapted to use Streamlit file uploader.
    """
    st.sidebar.subheader("Symptom Classifier Training Data")
    survey_file = st.sidebar.file_uploader("Upload Survey Report (CSV) for training", type=["csv"], key="survey_file_uploader")

    if survey_file:
        try:
            import pandas as pd
            df = pd.read_csv(survey_file)
            # Ensure these column names match exactly your CSV headers
            # You might need to adjust column names based on the uploaded CSV
            symptoms_col = 'What are the current symptoms or health issues you are facing'
            labels_col = 'Medical Health History'

            if symptoms_col not in df.columns or labels_col not in df.columns:
                 st.error(f"Survey report must contain columns '{symptoms_col}' and '{labels_col}'.")
                 return None, None

            symptoms = df[symptoms_col].fillna("").tolist()
            labels = df[labels_col].fillna("None").tolist()

            processed_labels = [label.split(',')[0].strip() if isinstance(label, str) else 'None' for label in labels]

            texts = symptoms
            labels = processed_labels

            if not texts or not labels:
                 st.warning("No valid data loaded from survey report for training.")
                 return None, None

            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)
            model = LogisticRegression()
            model.fit(X, labels)
            st.sidebar.success("Symptom classifier trained successfully!")
            return vectorizer, model

        except ImportError:
            st.error("scikit-learn is required for classifier training. Install with 'pip install scikit-learn'.")
            return None, None
        except Exception as e:
            st.error(f"An error occurred while training the classifier: {e}")
            return None, None
    else:
        st.sidebar.info("Upload a survey report CSV to train the symptom classifier.")
        return None, None


def predict_diagnosis(query, vectorizer, model):
    """
    Predicts a diagnosis from the patient query and returns the label with probability scores.
    """
    if vectorizer and model:
        try:
            X_query = vectorizer.transform([query])
            pred = model.predict(X_query)[0]
            proba = model.predict_proba(X_query)[0]
            prob_dict = dict(zip(model.classes_, proba))
            return pred, prob_dict
        except Exception as e:
            st.warning(f"Could not predict diagnosis: {e}")
            return "Prediction Failed", {}
    else:
        return "Classifier not trained", {}

def analyze_sentiment(query):
    """
    Performs sentiment analysis on the query using NLTK's VADER.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
             nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(query)
        return score
    except ImportError:
        st.warning("NLTK or VADER lexicon not found. Sentiment analysis skipped.")
        return {"compound": 0.0, "neg": 0.0, "neu": 0.0, "pos": 0.0}
    except Exception as e:
        st.warning(f"Error during sentiment analysis: {e}")
        return {"compound": 0.0, "neg": 0.0, "neu": 0.0, "pos": 0.0}


def extract_medical_entities(query):
    """
    Extracts medical entities from the query using spaCy.
    Only returns entities matching a predefined list of common medical terms.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            # Download the model if not found
            st.spinner("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(query)
        common_med_terms = {"flu", "migraine", "heart attack", "arthritis", "gastroenteritis",
                            "fever", "cough", "pain", "diarrhea", "vomiting", "headache",
                            "nausea", "chest pain", "shortness of breath", "sore throat",
                            "symptom", "issue", "condition", "medical history", "diagnosis"} # Added more terms

        entities = set()
        for ent in doc.ents:
            # Check if the entity text is a common medical term or appears to be a medical concept
            # This is a simple check; for more robust entity extraction, consider a medical NER model
            if ent.text.lower() in common_med_terms or any(term in ent.text.lower() for term in common_med_terms):
                 entities.add(ent.text)
            # Also consider checking against labels if available and relevant
            if 'clf_model' in st.session_state and st.session_state.clf_model:
                if ent.text in st.session_state.clf_model.classes_:
                    entities.add(ent.text)


        return list(entities)
    except ImportError:
        st.warning("spaCy not found. Medical entity extraction skipped.")
        return []
    except Exception as e:
        st.warning(f"Error during medical entity extraction: {e}")
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
    Generates and displays visualizations from the query history in Streamlit.
    """
    filename = "query_history.csv"
    if not os.path.exists(filename):
        st.info("No query history available for visualization.")
        return

    try:
        df = pd.read_csv(filename)
        if df.empty:
            st.info("Query history is empty.")
            return

        st.subheader("Query Trend Visualizations")

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Bar plot for predicted diagnosis frequency
        diag_counts = df['predicted_diagnosis'].value_counts()
        if not diag_counts.empty:
            st.write("#### Frequency of Predicted Diagnoses")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=diag_counts.index, y=diag_counts.values, palette="viridis", ax=ax)
            plt.xlabel("Diagnosis")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No predicted diagnoses to visualize.")

        # Line plot for sentiment compound over time
        if not df.empty:
            st.write("#### Sentiment Compound Score Over Time")
            df = df.sort_values("timestamp")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x="timestamp", y="sentiment_compound", marker="o", ax=ax)
            plt.xlabel("Timestamp")
            plt.ylabel("Sentiment Compound Score")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No sentiment data to visualize.")

        # Pie chart for common medical entities
        entity_list = []
        for entities_str in df['entities']:
            if pd.isna(entities_str) or entities_str.strip() == "":
                continue
            entities = [e.strip() for e in entities_str.split(",") if e.strip()]
            entity_list.extend(entities)

        if entity_list:
            st.write("#### Distribution of Extracted Medical Entities")
            entity_counts = Counter(entity_list)
            labels = list(entity_counts.keys())
            sizes = list(entity_counts.values())
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No medical entities to visualize.")

    except ImportError:
        st.error("matplotlib or seaborn not found. Cannot generate visualizations.")
    except Exception as e:
        st.error(f"An error occurred while generating visualizations: {e}")


def explain_diagnosis(query, vectorizer, model):
    """
    Provides a simple explanation for the predicted diagnosis using SHAP.
    """
    if vectorizer and model:
        try:
            import shap
            import numpy as np
            # Ensure the vectorizer has been fitted
            if not hasattr(vectorizer, 'vocabulary_'):
                 return "Vectorizer has not been fitted. Cannot generate explanation."

            X_query = vectorizer.transform([query]).toarray()

            # Create an explainer using a sample of the training data
            # Need to handle cases where feature_names_out might be empty or different
            feature_names = vectorizer.get_feature_names_out()
            if len(feature_names) == 0:
                return "No features found for explanation."

            # Select a representative sample from the training data (if available)
            # For simplicity, using the feature names themselves as samples is a common approach for TfidfVectorizer
            try:
                sample_texts = vectorizer.transform(feature_names).toarray()
            except Exception as e:
                 st.warning(f"Could not generate sample texts for SHAP explainer: {e}")
                 return "Explanation not available due to data sampling issue."

            # Ensure the model is a supported type for LinearExplainer
            if not hasattr(model, 'predict_proba'):
                 return "Model type not supported for SHAP explanation."


            explainer = shap.LinearExplainer(model, sample_texts, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_query)

            # Assuming a multi-class model, shap_values will be a list of arrays, one for each class
            # For a simple explanation, we might look at the SHAP values for the predicted class
            # Check if model.classes_ is available and if predicted_class_index is valid
            if hasattr(model, 'classes_') and len(model.classes_) > 0:
                predicted_class = model.predict(X_query)[0]
                try:
                    predicted_class_index = model.classes_.tolist().index(predicted_class)
                    if 0 <= predicted_class_index < len(shap_values):
                         shap_values_for_predicted_class = shap_values[predicted_class_index][0]

                         abs_shap = np.abs(shap_values_for_predicted_class)
                         top_indices = np.argsort(abs_shap)[-5:][::-1] # Get indices of top 5 features

                         explanation = "Top contributing terms to the predicted diagnosis:\n"
                         for idx in top_indices:
                              # Ensure idx is within the bounds of feature_names
                              if idx < len(feature_names):
                                 explanation += f"- {feature_names[idx]}: {shap_values_for_predicted_class[idx]:.4f}\n"
                              else:
                                 explanation += f"- Unknown feature (index {idx})\n"
                         return explanation
                    else:
                         return "Could not retrieve SHAP values for the predicted class."

                except ValueError:
                     return f"Predicted class '{predicted_class}' not found in model classes for explanation."
                except IndexError:
                     return "Index error when accessing SHAP values for the predicted class."

            else:
                 return "Model classes not available for explanation."


        except ImportError:
            st.warning("SHAP not found. Diagnosis explanation skipped.")
            return "Explanation not available (SHAP not installed)."
        except Exception as e:
            st.warning(f"Error during diagnosis explanation: {e}")
            return "Explanation not available."
    else:
        return "Classifier not trained. Cannot generate explanation."


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Medical Health Assistant")

st.title("Medical Health Assistant")
st.markdown("Get a comprehensive medical report based on your symptoms and relevant web content.")

# Load environment variables
load_dotenv()

# --- Sidebar for configuration and data upload ---
st.sidebar.title("Configuration & Data")

# API Keys - Consider using Streamlit Secrets for production
st.sidebar.subheader("API Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", key="gemini_api_key")
tavily_api_key = st.sidebar.text_input("Tavily API Key", type="password", key="tavily_api_key")
supabase_url = st.sidebar.text_input("Supabase URL", type="password", key="supabase_url")
supabase_key = st.sidebar.text_input("Supabase Key", type="password", key="supabase_key")

# Store keys in environment variables for the script to access
os.environ["GEMINI_API_KEY"] = gemini_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["SUPABASE_URL"] = supabase_url
os.environ["SUPABASE_KEY"] = supabase_key

# Gemini Model Selection
GEMINI_MODEL_NAME = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-2.5-flash-exp-04-17", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"], # Add other models as needed
    index=0,
    key="gemini_model_name"
)
os.environ["GEMINI_MODEL_NAME"] = GEMINI_MODEL_NAME # Update env var

# Analysis Config
temperature = st.sidebar.slider("Analysis Temperature", 0.0, 1.0, 0.3, 0.1, key="analysis_temperature")

# Search Configuration
st.sidebar.subheader("Web Search Configuration")
search_depth = st.sidebar.selectbox("Search Depth", ["basic", "advanced"], index=1, key="search_depth")
search_breadth = st.sidebar.slider("Search Breadth (results per query)", 1, 20, 10, key="search_breadth")

# Include/Omit URLs and File Uploads
st.sidebar.subheader("Additional Data")
include_urls_input = st.sidebar.text_area("Include URLs (comma-separated)", key="include_urls")
omit_urls_input = st.sidebar.text_area("Omit URLs (comma-separated)", key="omit_urls")
uploaded_files = st.sidebar.file_uploader("Upload Additional Reference Files", type=["pdf", "docx", "csv", "txt", "xls", "xlsx"], accept_multiple_files=True, key="uploaded_files")

# Wearable Data Upload (integrated into read_wearable_data placeholder)
# train_symptom_classifier() call is now within the sidebar

# --- Main Content Area ---

st.subheader("Patient Query")
original_query = st.text_area("Describe your current medical symptoms or issue:", key="original_query")

# Store the trained classifier in session state
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'clf_model' not in st.session_state:
    st.session_state.clf_model = None

# Train classifier if survey data is uploaded
st.session_state.vectorizer, st.session_state.clf_model = train_symptom_classifier()


# Placeholder for patient feedback loop (simplified for Streamlit)
# In a real app, you might use session state to manage a multi-turn conversation

# --- Action Button ---
if st.button("Generate Medical Report", key="generate_report_button"):
    if not original_query:
        st.warning("Please describe your symptoms or issue.")
    elif not gemini_api_key or not tavily_api_key:
         st.warning("Please enter your Gemini and Tavily API keys in the sidebar.")
    else:
        # Initialize API clients and services
        search_config = SearchConfig()
        analysis_config = AnalysisConfig(
            model_name=st.session_state.gemini_model_name,
            temperature=st.session_state.analysis_temperature
        )

        try:
            gemini_client = GeminiClient(api_key=st.session_state.gemini_api_key, config=analysis_config)
            search_client = TavilyClient(api_key=st.session_state.tavily_api_key)
            extractor = TavilyExtractor(api_key=st.session_state.tavily_api_key)
            search_service = WebSearchService(search_client, search_config)
            subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config)
        except Exception as e:
             st.error(f"Failed to initialize API clients. Check your API keys and ensure required libraries are installed. Error: {e}")
             # st.stop() # Don't stop the app immediately
             st.warning("Please ensure your API keys are correct and the agent modules are accessible.")


        # --- Processing ---
        with st.spinner("Generating report..."):
            current_date = datetime.today().strftime("%Y-%m-%d")
            task = MedicalTask(original_query)

            # Analyze query sentiment, extract medical entities, and predict diagnosis.
            sentiment_score = analyze_sentiment(original_query)
            entities = extract_medical_entities(original_query)
            st.subheader("Initial Query Analysis")
            st.write(f"**Sentiment Analysis:** {sentiment_score}")
            st.write(f"**Extracted Medical Entities:** {entities}")

            predicted_diag, diag_proba = predict_diagnosis(original_query, st.session_state.vectorizer, st.session_state.clf_model)
            st.write(f"**Predicted Diagnosis:** {predicted_diag}")
            st.write(f"**Prediction Probabilities:** {diag_proba}")
            explanation = explain_diagnosis(original_query, st.session_state.vectorizer, st.session_state.clf_model)
            st.write(f"**Diagnosis Explanation:**\n{explanation}")

            # Save query history for trend analysis.
            save_query_history(original_query, predicted_diag, sentiment_score, entities)

            # Process uploaded files
            additional_files_content = {}
            if uploaded_files:
                st.info("Processing uploaded files...")
                for file in uploaded_files:
                    file_content = extract_text_from_file(file)
                    if file_content:
                        additional_files_content[file.name] = file_content
                        st.write(f"Processed: {file.name}")

            # Perform analysis, search, and extraction
            include_urls = [url.strip() for url in include_urls_input.split(',') if url.strip()]
            omit_urls = [url.strip() for url in omit_urls_input.split(',') if url.strip()]

            try:
                # Note: In a real async Streamlit app, you might need a mechanism to run async tasks
                # For simplicity here, we're running them directly, which might block the UI
                # Consider libraries like streamlit-asyncio for better async support
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(task.analyze(subject_analyzer, current_date))
                loop.run_until_complete(task.search_and_extract(search_service, extractor, omit_urls, search_depth, search_breadth, include_urls))
                comprehensive_report, citations = loop.run_until_complete(task.analyze_full_content_rag(gemini_client))
                patient_report = loop.run_until_complete(task.generate_patient_summary_report(gemini_client, comprehensive_report))

            except Exception as e:
                 st.error(f"An error occurred during the report generation process: {e}")
                 logging.error(f"Report generation process failed: {e}", exc_info=True) # Log the traceback
                 comprehensive_report = "Report generation failed."
                 citations = []
                 patient_report = "Patient summary generation failed."


            # Generate and display reports
            task.generate_report(additional_files_content, comprehensive_report, citations)
            task.generate_summary_report(comprehensive_report, citations) # Generates and stores in task.summary_report
            # task.generate_patient_summary_report already populates task.patient_report


            st.subheader("Comprehensive Diagnostic Report")
            st.markdown(task.report) # Display the full report

            st.subheader("Summary Diagnostic Report")
            st.markdown(task.summary_report) # Display the summary report

            st.subheader("Patient-Friendly Summary Report")
            st.markdown(task.patient_report) # Display the patient summary report

            # --- Report Downloads ---
            st.subheader("Download Reports")

            def get_binary_file_downloader_html(bin_file, file_label='File'):
                # Check if bin_file is a string before encoding
                if isinstance(bin_file, str):
                    bin_to_b64 = base64.b64encode(bin_file.encode()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_to_b64}" download="{file_label}.md">Download {file_label}</a>'
                    return href
                else:
                    return f"Could not generate download link for {file_label} (invalid data type)."


            st.markdown(get_binary_file_downloader_html(task.report, 'Full_Diagnostic_Report'), unsafe_allow_html=True)
            st.markdown(get_binary_file_downloader_html(task.summary_report, 'Summary_Diagnostic_Report'), unsafe_allow_html=True)
            st.markdown(get_binary_file_downloader_html(task.patient_report, 'Patient_Summary_Report'), unsafe_allow_html=True)


        st.success("Report generation complete!")

# --- Visualization Section ---
st.subheader("Query Trend Analysis")
if st.button("Show Query Trend Visualizations"):
    visualize_query_trends()
