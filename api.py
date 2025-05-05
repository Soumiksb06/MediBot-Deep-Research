# api.py
import os
import sys
import asyncio
import logging
import csv
from datetime import datetime
from dotenv import load_dotenv
# from rich.console import Console # Not needed for API output, use logging

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Configure logging instead of rich console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to sys.path if needed (adjust path based on your project structure)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import agent modules (assuming they are in the same directory or accessible)
# Note: We will copy necessary functions/classes into this file for simplicity
# If they are in separate files/modules, ensure they are importable
# from web_agent.src.services.web_search import WebSearchService
# from web_agent.src.models.search_models import SearchConfig
# from subject_analyzer.src.services.tavily_client import TavilyClient
# from subject_analyzer.src.services.tavily_extractor import TavilyExtractor
# from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer
# from subject_analyzer.src.services.gemini_client import GeminiClient # <--- Changed line
# from subject_analyzer.src.models.analysis_models import AnalysisConfig
# from your_other_modules import read_wearable_data, build_faiss_index # Assuming these are elsewhere

# --- Copy/Paste or ensure importability of necessary functions and classes ---
# Due to the length, I'll include the necessary pieces adapted for API context.
# You might need to adjust imports based on your actual project structure.

# --- Start Copied/Adapted Functions/Classes ---

# Define necessary models and clients (adapt paths if they are in subdirectories)
# Assuming these are simplified for demonstration or you have actual modules

class SearchConfig:
    # Define SearchConfig as in your web_agent module
    pass

class AnalysisConfig:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature

# Placeholder clients - Replace with your actual client implementations
# Ensure these clients are thread-safe if initialized globally
class GeminiClient:
    def __init__(self, api_key: str, config: AnalysisConfig):
        self.api_key = api_key
        self.config = config
        # Initialize Google Generative AI client here
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.client = genai

    async def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        # Adapt this to use your actual Gemini client chat method
        try:
            model = self.client.GenerativeModel(self.config.model_name)
            # Convert messages format if needed for your client
            response = await model.generate_content_async(messages)
            # Adapt response parsing to match your client's output
            return {"choices": [{"message": {"content": response.text}}]} # Example structure
        except Exception as e:
            logger.error(f"Gemini chat failed: {e}")
            raise

    async def embed_content(self, model: str, contents: List[Any]):
         # Adapt this to use your actual Gemini client embedding method
         try:
            # Use the embed_content method from the google-generativeai client
            embedding_response = await self.client.embed_content_async( # Use the async version
                model=model,
                content=contents # Pass contents directly
            )
            # Assuming a structure like response.embedding.values
            embedding_vector = embedding_response['embedding']['values'] # Access values
            return {"embedding": embedding_vector} # Return in a dict similar to OpenAI's structure
         except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise


class TavilyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        import tavily # Assuming you installed tavily-python
        self.client = tavily.TavilyClient(api_key=self.api_key)

    def search_subject(self, query: str, topic: str, search_depth: str, results: int) -> Dict[str, Any]:
        # Adapt this to use your actual Tavily client search method
        try:
            response = self.client.search(
                query=f"{query} {topic}",
                search_depth=search_depth,
                max_results=results,
                include_images=False,
                include_raw_content=True # Request raw content for extraction
            )
            return response # Tavily client returns a dict directly
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise

class TavilyExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        import tavily # Assuming you installed tavily-python
        self.client = tavily.TavilyClient(api_key=self.api_key)

    def extract(self, urls: List[str], extract_depth: str, include_images: bool) -> Dict[str, Any]:
        # Adapt this to use your actual Tavily client extract method
        if not urls:
            return {"results": []}
        try:
            # Tavily client's search can include content, so direct extraction might not be needed
            # If you use Tavily's separate extract endpoint:
            # response = self.client.extract(url=urls[0]) # Tavily extract is usually per URL
            # Need to loop or use a method that handles multiple URLs if available

            # Assuming you get raw_content from search results and process it
            # If you need to *specifically* extract from user-provided URLs not in search:
            extracted_results = []
            for url in urls:
                 try:
                    content = self.client.get_content(url) # Use get_content for extraction
                    extracted_results.append({"url": url, "text": content, "raw_content": content})
                 except Exception as e:
                    logger.warning(f"Tavily get_content failed for {url}: {e}")
                    extracted_results.append({"url": url, "error": str(e)})

            return {"results": extracted_results}

        except Exception as e:
            logger.error(f"Tavily extraction failed: {e}")
            raise


class WebSearchService:
    def __init__(self, search_client: TavilyClient, config: SearchConfig):
        self.search_client = search_client
        self.config = config

    def search_subject(self, query: str, topic: str, search_depth: str, results: int) -> Dict[str, Any]:
         # This method should call the search_client
         return self.search_client.search_subject(query, topic, search_depth, results)


class SubjectAnalyzer:
    def __init__(self, llm_client: GeminiClient, config: AnalysisConfig):
        self.llm_client = llm_client
        self.config = config

    def analyze(self, query: str) -> Dict[str, Any]:
        # Adapt this to use your actual LLM client for analysis
        prompt = f"""Analyze the following medical query and identify:
- main_subject: The primary medical issue or symptom.
- temporal_context: Any mentioned timeframes or duration (e.g., "for 3 days", "since last week").
- What_needs_to_be_researched: Key aspects or related conditions to investigate further.
Respond as a JSON object.

Query: {query}
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            # Use the synchronous chat method if the client is synchronous, or await if it's async
            # Assuming the client is async and has an async chat method
            response = asyncio.run(self.llm_client.chat(messages))
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Attempt to parse JSON response
            import json
            try:
                analysis_result = json.loads(content)
                return analysis_result
            except json.JSONDecodeError:
                 logger.warning(f"Failed to parse JSON from analysis response: {content}")
                 return {"main_subject": query, "temporal_context": {}, "What_needs_to_be_researched": []} # Fallback
        except Exception as e:
            logger.error(f"Subject analysis LLM call failed: {e}")
            return {"main_subject": query, "temporal_context": {}, "What_needs_to_be_researched": []} # Fallback


# --- Data Science / Analytics & Visualization Functions (Adapted) ---

def train_symptom_classifier(csv_file_path: str):
    """
    Trains a simple symptom-to-diagnosis classifier using the survey data from a file.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import pandas as pd # Import pandas

    try:
        df = pd.read_csv(csv_file_path)
        symptoms = df['What are the current symptoms or health issues you are facing'].fillna("").tolist()
        labels = df['Medical Health History'].fillna("None").tolist()
        processed_labels = [label.split(',')[0].strip() if isinstance(label, str) else 'None' for label in labels]

        texts = symptoms
        labels = processed_labels

        if not texts or not labels:
             logger.warning("No data loaded from survey report for training.")
             return None, None

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression()
        model.fit(X, labels)
        logger.info("Symptom classifier trained successfully.")
        return vectorizer, model

    except FileNotFoundError:
        logger.error(f"Error: Survey report CSV not found at {csv_file_path}.")
        # Fallback to dummy data or handle appropriately - For API, better to fail or return None
        # Using dummy data as a last resort, but highlight this failure
        logger.warning("Using dummy data for classifier training due to file not found.")
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

    except KeyError as e:
        logger.error(f"Error: Missing expected column in survey report CSV: {e}")
        return None, None
    except Exception as e:
        logger.error(f"An error occurred while reading the survey report or training: {e}")
        return None, None


def predict_diagnosis(query, vectorizer, model):
    """
    Predicts a diagnosis from the patient query and returns the label with probability scores.
    Requires trained vectorizer and model.
    """
    if not vectorizer or not model:
        return "Classifier not available", {}
    try:
        X_query = vectorizer.transform([query])
        pred = model.predict(X_query)[0]
        proba = model.predict_proba(X_query)[0]
        prob_dict = dict(zip(model.classes_, proba))
        return pred, prob_dict
    except Exception as e:
        logger.error(f"Diagnosis prediction failed: {e}")
        return "Prediction Error", {}


def analyze_sentiment(query):
    """
    Performs sentiment analysis on the query using NLTK's VADER.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        # Ensure vader_lexicon is downloaded - must happen in a writable location on deployment
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
            nltk.download('vader_lexicon', quiet=True) # This might not work in all deployment envs
        except nltk.downloader.LookUpError:
             nltk.download('vader_lexicon', quiet=True)

        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(query)
        return score
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {"compound": 0.0, "neg": 0.0, "neu": 0.0, "pos": 0.0}


def extract_medical_entities(query):
    """
    Extracts medical entities from the query using spaCy.
    Only returns entities matching a predefined list of common medical terms.
    Requires spaCy model to be downloaded.
    """
    try:
        import spacy
        # Try to load the English model - must be downloaded on deployment
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy model 'en_core_web_sm' not found locally, attempting download. Error: {e}")
            try:
                 # This might fail in read-only file systems on deployment
                 spacy.cli.download("en_core_web_sm")
                 nlp = spacy.load("en_core_web_sm")
            except Exception as download_e:
                 logger.error(f"Failed to download spaCy model 'en_core_web_sm': {download_e}")
                 return [] # Cannot perform NER without model

        doc = nlp(query)
        common_med_terms = {"flu", "migraine", "heart attack", "arthritis", "gastroenteritis",
                            "fever", "cough", "pain", "diarrhea", "vomiting", "headache",
                            "nausea", "chest pain", "shortness of breath", "sore throat"}
        entities = set()
        for ent in doc.ents:
            if ent.text.lower() in common_med_terms:
                entities.add(ent.text)
        return list(entities)
    except Exception as e:
        logger.error(f"Medical entity extraction failed: {e}")
        return []


# Adapted for Supabase persistence
async def save_query_history_to_db(supabase_client, query, diagnosis, sentiment, entities):
    """
    Saves the query details to Supabase for trend analysis.
    Requires Supabase client.
    """
    try:
        data_to_insert = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "predicted_diagnosis": diagnosis,
            "sentiment_compound": sentiment.get("compound", 0),
            "entities": entities # Store as list or string depending on Supabase column type (jsonb recommended)
        }
        # Assuming a 'query_history' table exists in Supabase
        response = await supabase_client.table("query_history").insert([data_to_insert]).execute()
        # Supabase client might return response object, check for errors
        if hasattr(response, 'error') and response.error:
             logger.error(f"Failed to save query history to Supabase: {response.error}")
        else:
             logger.info("Query history saved to Supabase.")
    except Exception as e:
        logger.error(f"An error occurred while saving query history to Supabase: {e}")


# Adapted to retrieve data from DB instead of reading CSV
async def get_query_history_data(supabase_client):
    """
    Retrieves query history data from Supabase for trend analysis.
    Returns data needed for visualization plots.
    """
    try:
        # Assuming a 'query_history' table exists
        response = await supabase_client.table("query_history").select("*").order("timestamp").execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Failed to retrieve query history from Supabase: {response.error}")
            return None
        data = response.data # List of dicts
        if not data:
            logger.warning("No query history data found in Supabase.")
            return None

        # Process data for visualization
        diagnoses = [item.get('predicted_diagnosis') for item in data if item.get('predicted_diagnosis')]
        sentiments = [(item.get('timestamp'), item.get('sentiment_compound')) for item in data if item.get('timestamp') and item.get('sentiment_compound') is not None]
        all_entities = []
        for item in data:
             entities_list = item.get('entities')
             if isinstance(entities_list, list): # Assuming entities are stored as JSONB array
                 all_entities.extend(entities_list)
             elif isinstance(entities_list, str): # If stored as comma-separated string
                 all_entities.extend([e.strip() for e in entities_list.split(',') if e.strip()])


        # Return data structures suitable for a client to plot
        from collections import Counter
        diag_counts = dict(Counter(diagnoses))
        entity_counts = dict(Counter(all_entities))

        sentiment_data = [{"timestamp": ts, "compound": comp} for ts, comp in sentiments]


        return {
            "diagnosis_frequency": diag_counts,
            "sentiment_over_time": sentiment_data,
            "entities_distribution": entity_counts
        }

    except Exception as e:
        logger.error(f"An error occurred while retrieving query history data: {e}")
        return None


# Adapted to accept file-like object or bytes for file extraction
def extract_text_from_file_obj(file_obj: UploadFile, logger):
    """
    Extracts text from an uploaded file object.
    """
    # Determine file extension from filename or content type
    filename = file_obj.filename
    ext = os.path.splitext(filename)[1].lower()
    logger.info(f"Attempting to extract text from uploaded file: {filename} with extension {ext}")

    try:
        if ext == '.pdf':
            try:
                import PyPDF2
            except ImportError:
                logger.error("PyPDF2 is required for PDF extraction but not installed.")
                return ""
            text = ""
            try:
                reader = PyPDF2.PdfReader(file_obj.file) # Use file_obj.file which is a BytesIO object
                for page in reader.pages:
                    text += page.extract_text() or ""
                logger.info(f"Successfully extracted text from PDF: {filename}")
                return text
            except Exception as e:
                logger.error(f"Error extracting PDF {filename}: {e}")
                return ""

        elif ext == '.docx':
            try:
                import docx
                from io import BytesIO
            except ImportError:
                logger.error("python-docx is required for DOCX extraction but not installed.")
                return ""
            try:
                file_obj.file.seek(0) # Ensure file pointer is at the beginning
                doc = docx.Document(BytesIO(file_obj.file.read()))
                text = "\n".join(para.text for para in doc.paragraphs)
                logger.info(f"Successfully extracted text from DOCX: {filename}")
                return text
            except Exception as e:
                logger.error(f"Error extracting DOCX {filename}: {e}")
                return ""

        elif ext in ['.csv', '.xls', '.xlsx']:
            try:
                import pandas as pd
                from io import BytesIO
            except ImportError:
                logger.error("pandas is required for CSV/Excel extraction but not installed.")
                return ""
            try:
                 file_obj.file.seek(0) # Ensure file pointer is at the beginning
                 if ext == '.csv':
                     df = pd.read_csv(BytesIO(file_obj.file.read()))
                 else: # .xls, .xlsx
                     df = pd.read_excel(BytesIO(file_obj.file.read()))
                 text = df.to_csv(index=False)
                 logger.info(f"Successfully extracted data from spreadsheet: {filename}")
                 return text
            except Exception as e:
                 logger.error(f"Error extracting spreadsheet {filename}: {e}")
                 return ""

        else: # Assume text file
            try:
                file_obj.file.seek(0)
                # Need to decode bytes to string
                content_bytes = file_obj.file.read()
                try:
                    text = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                     try:
                         text = content_bytes.decode("latin-1") # Try another common encoding
                     except Exception:
                          text = content_bytes.decode(errors='ignore') # Ignore errors
                logger.info(f"Successfully extracted text from file: {filename}")
                return text
            except Exception as e:
                logger.error(f"Error extracting text from {filename}: {e}")
                return ""

    except Exception as ex:
        logger.error(f"General file extraction error for {filename}: {ex}")
        return ""


# Adapted to read from uploaded file or return None
def read_wearable_data_from_file_obj(file_obj: Optional[UploadFile], logger):
    """
    Reads wearable data from an uploaded file object (if provided) and returns a summary string.
    """
    if file_obj is None:
        return None

    filename = file_obj.filename
    logger.info(f"Attempting to read wearable data from uploaded file: {filename}")

    try:
        import pandas as pd
        from io import BytesIO
        file_obj.file.seek(0)
        df = pd.read_csv(BytesIO(file_obj.file.read()))

        summary = "Wearable Data Summary:\n"
        if "heart_rate" in df.columns:
            avg_hr = df["heart_rate"].mean()
            summary += f"Average Heart Rate: {avg_hr:.1f}\n"
        if "steps" in df.columns:
            total_steps = df["steps"].sum()
            summary += f"Total Steps: {total_steps}\n"
        # Add other relevant columns if needed
        logger.info(f"Successfully processed wearable data from {filename}")
        return summary
    except Exception as e:
        logger.error(f"Error reading or processing wearable data from {filename}: {e}")
        return f"Could not process wearable data from {filename}."


def build_faiss_index(embeddings):
    """
    Builds a FAISS index from a list of embeddings.
    Requires faiss-cpu and numpy.
    """
    try:
        import faiss
        import numpy as np
        if not embeddings:
            logger.warning("No embeddings provided to build FAISS index.")
            return None
        # Assuming embeddings are lists of floats
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        embeddings_array = np.array(embeddings).astype('float32')
        index.add(embeddings_array)
        logger.info(f"FAISS index built with {len(embeddings)} embeddings.")
        return index
    except ImportError:
        logger.error("FAISS or Numpy not installed. Cannot build FAISS index.")
        return None
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        return None


# Explanation function remains largely the same, just needs vectorizer/model
def explain_diagnosis(query, vectorizer, model):
    """
    Provides a simple explanation for the predicted diagnosis using SHAP.
    Requires trained vectorizer, model, and shap.
    """
    if not vectorizer or not model:
        return "Explanation not available: Classifier not trained."
    try:
        import shap
        import numpy as np

        # SHAP explainer setup can be resource-intensive and might need caching
        # or optimization for a production API. For simplicity, we re-create here.
        # Ensure shap is installed.

        # Generate a small sample of data from the vectorizer's vocabulary for SHAP
        # This assumes the vectorizer was trained on text data.
        # Getting feature names might be slow or memory intensive with a large vocabulary.
        feature_names = vectorizer.get_feature_names_out()
        # Create a dummy sample X for explainer training
        # Need to handle case where vocabulary is empty or too large
        if not feature_names or len(feature_names) > 1000: # Limit sample size
             logger.warning("Vectorizer vocabulary too large or empty for SHAP explanation sample.")
             return "Explanation not available: Model complexity too high or vectorizer issue."

        sample_texts = [" ".join(feature_names[:min(len(feature_names), 100)])] # Use a subset of features
        sample_X = vectorizer.transform(sample_texts).toarray()


        explainer = shap.LinearExplainer(model, sample_X, feature_perturbation="interventional")

        X_query = vectorizer.transform([query]).toarray()
        if X_query.shape[1] != sample_X.shape[1]:
             logger.error("Feature mismatch between query and explainer sample for SHAP.")
             return "Explanation not available: Feature mismatch."

        shap_values = explainer.shap_values(X_query)

        # Assuming binary or multi-class classification, shap_values can be a list of arrays
        # For simplicity, let's focus on the explanation for the predicted class
        if isinstance(shap_values, list):
            # Find the index of the predicted class
            predicted_class_idx = model.classes_.tolist().index(model.predict(X_query)[0])
            class_shap_values = shap_values[predicted_class_idx][0]
        else:
            class_shap_values = shap_values[0] # Assume shape (1, num_features)


        abs_shap = np.abs(class_shap_values)
        # Ensure we don't ask for more indices than available features
        top_indices_count = min(5, len(abs_shap))
        top_indices = np.argsort(abs_shap)[-top_indices_count:][::-1]

        explanation = f"Top features contributing to predicted diagnosis '{model.predict(X_query)[0]}':\n"
        if top_indices_count == 0:
             explanation += "No contributing features identified."
        else:
            for idx in top_indices:
                # Ensure idx is within bounds of feature_names
                if idx < len(feature_names):
                    explanation += f"- {feature_names[idx]}: {class_shap_values[idx]:.4f}\n"
                else:
                    logger.warning(f"SHAP top index {idx} out of bounds for feature names length {len(feature_names)}")


        return explanation
    except ImportError:
         logger.error("SHAP library not installed. Cannot generate explanations.")
         return "Explanation not available: SHAP library missing."
    except Exception as e:
        logger.error(f"Error during diagnosis explanation with SHAP: {e}")
        return "Explanation not available due to internal error."


# --- Main Task Class (Adapted for API) ---

class MedicalTask:
    def __init__(self, query: str, logger: logging.Logger):
        self.original_query = query
        self.current_query = query # Might be updated with feedback
        self.analysis: Dict[str, Any] = {}
        self.search_results: Dict[str, List[Dict[str, Any]]] = {}
        self.extracted_content: Dict[str, List[Dict[str, Any]]] = {}
        self.additional_files_content: Dict[str, str] = {} # Store content of uploaded files
        self.wearable_summary: Optional[str] = None # Store wearable data summary
        self.logger = logger
        self.feedback_history: List[Dict[str, Any]] = [] # Not used interactively in API, but can store initial feedback

    def update_feedback(self, feedback: str):
        # In an API, feedback comes in the initial request.
        # This method can be used to *record* the feedback and update the query if needed.
        if feedback:
            self.feedback_history.append({
                "query": self.current_query,
                "feedback": feedback,
                "time": datetime.now().isoformat()
            })
            self.current_query = f"{self.original_query} - Additional context: {feedback}"
            self.logger.info(f"Query updated with feedback: {self.current_query}")


    async def analyze(self, subject_analyzer: SubjectAnalyzer, current_date: str):
        self.logger.info(f"Analyzing patient query for diagnosis (as of {current_date})...")
        try:
            self.analysis = subject_analyzer.analyze(f"{self.current_query} (as of {current_date})")
            self.logger.info("Subject analysis successful.")
        except Exception as e:
            self.logger.error(f"Subject analysis failed: {e}")
            # Depending on severity, you might raise the exception or return partial results
            raise # Re-raise to be caught by the API endpoint


    async def search_and_extract(self, search_service: WebSearchService, extractor: TavilyExtractor, omit_urls: List[str], search_depth: str, search_breadth: int):
        topics = [self.analysis.get("main_subject", self.current_query)]
        topics += self.analysis.get("What_needs_to_be_researched", [])
        for topic in topics:
            if not topic:
                continue
            self.logger.info(f"Searching for medical information on: {topic}")
            # detailed_query = f"{topic} medical diagnosis (Depth: {search_depth}, Breadth: {search_breadth})" # Not needed for search call
            self.logger.info(f"Executing search for topic: {topic}")
            try:
                # Assuming search_service.search_subject returns search results with raw_content
                response = search_service.search_subject(
                    topic, "medical", search_depth=search_depth, results=search_breadth
                )
                results = response.get("results", [])
                results = [res for res in results if res.get("url") and not any(
                    omit.lower() in res.get("url").lower() for omit in omit_urls)]
                self.search_results[topic] = results

                # Extraction part: Use the raw_content already fetched by Tavily search
                extracted_items = []
                for res in results:
                     url = res.get("url", "No URL")
                     title = res.get("title", "No Title")
                     # Tavily search results often contain 'content' or 'raw_content'
                     content = res.get("content") or res.get("raw_content", "")

                     if content:
                        # Optionally truncate content here if it's too large
                        # if len(content) > 1000:
                        #    content = content[:1000] + "..." # Example truncation
                         extracted_items.append({
                             "url": url,
                             "title": title,
                             "text": content, # Use the content directly
                             "raw_content": content # Use the content directly
                         })
                     else:
                         self.logger.warning(f"No content in search result for {url}. Attempting separate extraction?")
                         # Optionally attempt separate extraction here if search didn't provide content
                         # extraction_response = extractor.extract(urls=[url], ...)
                         pass # For now, rely on search result content


                self.extracted_content[topic] = extracted_items

                failed_urls = [res.get("url") for res in results if not (res.get("content") or res.get("raw_content"))]
                if failed_urls:
                    self.logger.warning(f"Warning: Could not get content for {len(failed_urls)} URLs from search results.")


            except Exception as e:
                self.logger.error(f"Search and Extraction failed for {topic}: {e}")
                self.search_results[topic] = []
                self.extracted_content[topic] = []

    # Adapted to handle user-provided URLs directly for extraction
    async def extract_from_urls(self, extractor: TavilyExtractor, urls: List[str], omit_urls: List[str]):
         if not urls:
             self.logger.info("No user-provided URLs to extract.")
             return

         filtered_urls = [url for url in urls if not any(omit.lower() in url.lower() for omit in omit_urls)]
         if not filtered_urls:
              self.logger.warning("All user-provided URLs were omitted.")
              return

         self.logger.info(f"Extracting content from user provided URLs: {filtered_urls}")
         self.search_results["User Provided"] = [{"title": "User Provided", "url": url} for url in filtered_urls] # Record in search_results for report

         try:
             extraction_response = extractor.extract(
                 urls=filtered_urls,
                 extract_depth="advanced", # Or a parameter
                 include_images=False
             )
             extracted_items = extraction_response.get("results", [])

             # Clean up extracted content - keep a central portion if needed (adjust indices)
             for item in extracted_items:
                if "text" in item and item["text"]:
                    text = item["text"]
                    if len(text) > 1000: # Example truncation
                         start_index = max(0, len(text) // 2 - 500)
                         end_index = min(len(text), len(text) // 2 + 500)
                         item["text"] = text[start_index:end_index]
                # Assuming 'raw_content' might also be returned and needs similar handling
                if "raw_content" in item and item["raw_content"]:
                     raw_content = item["raw_content"]
                     if len(raw_content) > 1000: # Example truncation
                         start_index = max(0, len(raw_content) // 2 - 500)
                         end_index = min(len(raw_content), len(raw_content) // 2 + 500)
                         item["raw_content"] = raw_content[start_index:end_index]


             self.extracted_content["User Provided"] = extracted_items
             failed = [res for res in extracted_items if res.get("error")]
             if failed:
                 self.logger.warning(f"Warning: Failed to extract {len(failed)} user-provided URLs")

         except Exception as e:
             self.logger.error(f"Extraction failed for user provided URLs: {e}")
             self.extracted_content["User Provided"] = []


    # Modified analyze_full_content_rag to use the gemini_client for embeddings and chat
    async def analyze_full_content_rag(self, gemini_client: GeminiClient, supabase_client):
        self.logger.info("Aggregating extracted content for comprehensive diagnostic analysis...")

        full_content = ""
        citations = set() # Use a set to avoid duplicate citations
        for topic, items in self.extracted_content.items():
            for item in items:
                url = item.get("url", "No URL")
                title = item.get("title", "No Title")
                # Prefer 'text' if available, fallback to 'raw_content'
                content = item.get("text") or item.get("raw_content", "")

                if content: # Only add content if it's not empty
                    full_content += f"\n\n=== Content from {url} ===\n{content}\n"
                    citations.add(f"{title}: {url}") # Add to set

        # Include content from uploaded files if available
        if self.additional_files_content:
             self.logger.info("Including content from additional uploaded files in RAG.")
             for filename, content in self.additional_files_content.items():
                  if content: # Only add if content is not empty
                      full_content += f"\n\n=== Content from Uploaded File: {filename} ===\n{content}\n"
                      citations.add(f"Uploaded File: {filename}") # Add file as a citation source


        # Include wearable data summary if available
        if self.wearable_summary:
             self.logger.info("Including wearable data summary in RAG.")
             full_content += f"\n\n=== Wearable Data Summary ===\n{self.wearable_summary}\n"
             citations.add("Wearable Data") # Add wearable data as a citation source


        def chunk_text(text, chunk_size=765):
            # Simple chunking - can be improved
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        chunks = chunk_text(full_content, chunk_size=765)
        self.logger.info(f"Total chunks generated: {len(chunks)}")

        if not chunks:
             self.logger.warning("No content chunks generated for RAG.")
             return "No relevant content found or extracted.", list(citations)

        # === Embedding Section using Gemini (Google Generative AI) ===
        # We need to call the async embed_content method
        async def get_embedding_async(text):
            try:
                # Ensure GOOGLE_EMBEDDING_MODEL is set in environment variables
                GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "embedding-001") # Common Google embedding model
                # Use the awaitable embed_content method
                response = await gemini_client.embed_content(
                    model=GOOGLE_EMBEDDING_MODEL,
                    contents=[{"parts": [{"text": text}]}] # Format required by some Google embedding APIs
                    # Or simply: contents=[text] depending on the exact client version/model
                )
                embedding_vector = response["embedding"] # Adapt access based on your client response
                return embedding_vector
            except Exception as e:
                self.logger.error(f"Embedding error for chunk: {e}")
                # Return None or raise exception for failed embeddings
                return None # Or raise e

        # Get embeddings for all chunks concurrently
        # Filter out None results if embedding fails for some chunks
        chunk_embeddings = [
            emb for emb in await asyncio.gather(*[get_embedding_async(chunk) for chunk in chunks]) if emb is not None
        ]

        # Ensure the number of embeddings matches the number of successfully embedded chunks
        embedded_chunks = [chunk for chunk, emb in zip(chunks, chunk_embeddings) if emb is not None]

        self.logger.info(f"Successfully generated {len(chunk_embeddings)} embeddings.")

        if not chunk_embeddings:
             self.logger.warning("No embeddings generated. Cannot perform RAG.")
             return "Could not generate embeddings for RAG.", list(citations)


        # === Supabase and Optional FAISS ===
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")

        if not SUPABASE_URL or not SUPABASE_KEY:
             self.logger.error("Supabase URL or Key is not set. Cannot use Supabase for RAG.")
             # Fallback or raise error
             return "Supabase is not configured, RAG failed.", list(citations)

        # Assuming supabase_client is already initialized and passed
        # Use upsert with on_conflict for robustness
        try:
             # Clear previous aggregated content embeddings if needed
             # Careful with clearing - only clear those specific to this task/session if possible
             # For simplicity, let's assume we might clear based on a source tag
             # await supabase_client.table("embeddings").delete().eq("source", "Aggregated content").execute()
             # logger.info("Cleared previous 'Aggregated content' embeddings.")
             pass # Decide on a strategy for clearing/managing embeddings

        except Exception as e:
            self.logger.warning(f"Warning: Could not clear previous embeddings in Supabase: {e}")


        data_to_insert = []
        for i, chunk in enumerate(embedded_chunks):
            # Ensure embedding_vector is a list
            embedding_vector = chunk_embeddings[i]
            if hasattr(embedding_vector, 'tolist'):
                 embedding_vector = embedding_vector.tolist()
            elif not isinstance(embedding_vector, list):
                 self.logger.error(f"Embedding vector for chunk {i} is not a list or convertible: {type(embedding_vector)}")
                 continue # Skip this chunk

            data_to_insert.append({
                "chunk": chunk,
                "embedding": embedding_vector,
                "source": "Aggregated content", # Tag to identify these embeddings
                "query_id": "some-unique-id" # Optional: Add a unique ID per query for better management
            })

        batch_size = 200
        # Use upsert with on_conflict on a unique key if you have one, otherwise insert
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i+batch_size]
            try:
                # Assuming your Supabase table can handle this structure and has an 'embedding' column of type vector
                # and a 'chunk' column for the text.
                # If you have a unique ID column, use .upsert(batch, on_conflict='unique_id_column')
                response = await supabase_client.table("embeddings").insert(batch).execute()
                if hasattr(response, 'error') and response.error:
                     self.logger.error(f"Error inserting batch {i // batch_size + 1} into Supabase: {response.error}")
                else:
                    self.logger.info(f"Inserted batch {i // batch_size + 1} into Supabase.")
            except Exception as e:
                 self.logger.error(f"Error inserting batch {i // batch_size + 1} into Supabase: {e}")


        # Obtain embedding for the query using the async function
        try:
            query_embedding = await get_embedding_async(self.current_query)
            if query_embedding is None:
                 raise ValueError("Failed to generate embedding for query.")
             # Ensure query_embedding is a list for Supabase RPC
            if hasattr(query_embedding, 'tolist'):
                 query_embedding = query_embedding.tolist()
            elif not isinstance(query_embedding, list):
                 raise TypeError(f"Query embedding is not a list or convertible: {type(query_embedding)}")

        except Exception as e:
            self.logger.error(f"Failed to get embedding for RAG query: {e}")
            # Decide how to proceed if query embedding fails - likely cannot perform RAG
            # Attempt to clear inserted embeddings if possible
            # await supabase_client.table("embeddings").delete().eq("source", "Aggregated content").execute() # Clean up
            return "Error: Could not generate query embedding for RAG.", list(citations)


        USE_FAISS = os.getenv("USE_FAISS", "False").lower() == "true"
        matched_chunks_content = []

        if USE_FAISS and chunk_embeddings: # Only use FAISS if embeddings were generated
            self.logger.info("Using FAISS for vector search.")
            try:
                # build_faiss_index needs numpy
                faiss_index = build_faiss_index(chunk_embeddings) # chunk_embeddings are already lists here
                if faiss_index is not None:
                    k = min(155, len(embedded_chunks)) # Match against successfully embedded chunks
                    import numpy as np
                    query_vec = np.array(query_embedding).reshape(1, -1).astype('float32') # query_embedding is a list
                    distances, indices = faiss_index.search(query_vec, k)
                    # Retrieve chunks corresponding to the indices from the embedded_chunks list
                    matched_chunks_content = [embedded_chunks[i] for i in indices[0] if i < len(embedded_chunks)]
                    self.logger.info(f"FAISS search returned {len(matched_chunks_content)} matched chunks.")
                else:
                    self.logger.warning("FAISS index could not be built; falling back to Supabase.")
                    # Fallback to Supabase RPC
                    match_response = await supabase_client.rpc("match_chunks", {"query_embedding": query_embedding, "match_count": 200}).execute()
                    matched_chunks_content = [row["chunk"] for row in match_response.data] if match_response.data else []
                    self.logger.info(f"Supabase (fallback) search returned {len(matched_chunks_content)} matched chunks.")

            except Exception as e:
                 self.logger.error(f"Error during FAISS search or fallback: {e}")
                 # Attempt Supabase RPC as a final fallback if FAISS failed completely
                 try:
                     match_response = await supabase_client.rpc("match_chunks", {"query_embedding": query_embedding, "match_count": 200}).execute()
                     matched_chunks_content = [row["chunk"] for row in match_response.data] if match_response.data else []
                     self.logger.info(f"Supabase (final fallback) search returned {len(matched_chunks_content)} matched chunks.")
                 except Exception as final_e:
                     self.logger.error(f"Final Supabase RPC match failed: {final_e}")
                     matched_chunks_content = [] # No chunks retrieved


        else:
            self.logger.info("Using Supabase RPC for vector search.")
            try:
                # Use Supabase RPC for matching
                match_response = await supabase_client.rpc("match_chunks", {"query_embedding": query_embedding, "match_count": 200}).execute()
                matched_chunks_content = [row["chunk"] for row in match_response.data] if match_response.data else []
                self.logger.info(f"Supabase search returned {len(matched_chunks_content)} matched chunks.")
            except Exception as e:
                self.logger.error(f"Supabase RPC match failed: {e}")
                matched_chunks_content = [] # No chunks retrieved


        if not matched_chunks_content:
             self.logger.warning("No relevant content chunks retrieved for RAG.")
             # Attempt to clear inserted embeddings
             # await supabase_client.table("embeddings").delete().eq("source", "Aggregated content").execute() # Clean up
             return "No relevant information found to generate a detailed report.", list(citations)


        aggregated_relevant = "\n\n".join(matched_chunks_content)

        # Revised prompt: instruct the LLM to answer the patient's main query directly.
        prompt = f"""You are an expert diagnostic report generator. Based on the following aggregated content, generate a comprehensive diagnostic report that directly addresses the patient's query:
"{self.current_query}"
Provide detailed medical analysis, actionable recommendations, and include citations for each source.
Ensure the recommendations are clear steps the patient should take (e.g., "Consult a doctor", "Get a blood test", "Rest and hydrate").

Aggregated Relevant Content:
{aggregated_relevant}

Citations:
{chr(10).join(list(citations))}

Respond with a detailed Markdown-formatted report.
"""
        messages = [
            {"role": "system", "content": "You are an expert diagnostic report generator."},
            {"role": "user", "content": prompt}
        ]

        self.logger.info("Performing secondary analysis via RAG (LLM call)...")
        try:
            # Use the gemini_client (LLM client) for the final report generation
            # Assuming gemini_client.chat is awaitable
            response = await gemini_client.chat(messages)
            comprehensive_report = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.logger.info("Comprehensive report generated.")

            # Decide if you want to clear embeddings after every query or manage them differently
            try:
                # await supabase_client.table("embeddings").delete().eq("source", "Aggregated content").execute()
                # self.logger.info("[green]Cleared Supabase embeddings after report generation.[/green]")
                pass # Handle embedding cleanup based on your strategy
            except Exception as e:
                self.logger.error(f"Error cleaning up Supabase embeddings: {e}")

            return comprehensive_report, list(citations) # Return citations as list

        except Exception as e:
            self.logger.error(f"Secondary analysis via RAG failed: {e}")
            # Attempt to clear inserted embeddings
            # await supabase_client.table("embeddings").delete().eq("source", "Aggregated content").execute() # Clean up
            return "Secondary analysis failed to generate a comprehensive report.", list(citations)


    # Adapted to return content instead of saving file
    def generate_report_content(self, comprehensive_report: str, citations: List[str]) -> str:
        """Generates the content of the full diagnostic report."""
        report = f"# Medical Diagnostic Report for: {self.original_query} (Generated on {datetime.today().strftime('%Y-%m-%d')})\n\n"
        report += f"**Refined Query:** {self.current_query}\n\n"
        report += "## Agent's Understanding\n"
        if self.analysis:
            for key, value in self.analysis.items():
                report += f"- **{key.capitalize()}**: {str(value) if not isinstance(value, (list, dict)) else value}\n"
        else:
             report += "Analysis not available.\n"


        report += "\n## Search Results\n"
        if self.search_results:
            for topic, results in self.search_results.items():
                report += f"### Search: {topic}\n"
                if results:
                    for res in results:
                        title = res.get("title", "No Title")
                        url = res.get("url", "No URL")
                        # Relevance score might not be available from all sources/extractions
                        # relevance = res.get("score", "N/A")
                        report += f"- **Title:** {title}\n  - **URL:** {url}\n\n" # Removed relevance score for simplicity

                else:
                    report += "No search results found.\n\n"
        else:
             report += "No search results available.\n"


        report += "## Extracted Full Content\n"
        if self.extracted_content:
            for topic, items in self.extracted_content.items():
                report += f"### Extraction for: {topic}\n"
                if items:
                    for item in items:
                        ext_url = item.get("url", "No URL")
                        # Prioritize 'text' which might be a summary or cleaned content
                        text = item.get("text") or item.get("raw_content", "")
                        # Truncate content for the report overview to keep it manageable
                        display_text = text[:500] + "..." if len(text) > 500 else text
                        report += f"- **URL:** {ext_url}\n  - **Content Snippet:**\n```\n{display_text}\n```\n\n"
                else:
                    report += "No content extracted.\n\n"
        else:
             report += "No extracted content available.\n"


        if self.additional_files_content:
            report += "## Additional Reference Files Content\n"
            for path, content in self.additional_files_content.items():
                 display_content = content[:500] + "..." if len(content) > 500 else content
                 report += f"- **{path}**:\n```\n{display_content}\n```\n\n"


        if self.wearable_summary:
            report += "## Wearable Data Summary\n"
            report += self.wearable_summary + "\n"


        report += "\n---\n\n"
        report += "## Comprehensive Diagnostic Report (via Secondary Analysis)\n"
        report += comprehensive_report + "\n\n"
        report += "## Citations\n"
        if citations:
            for citation in citations:
                report += f"- {citation}\n"
        else:
             report += "No citations available.\n"

        return report

    # Adapted to return content instead of saving file
    def generate_summary_report_content(self, comprehensive_report: str, citations: List[str]) -> str:
        """Generates the content of the summary diagnostic report."""
        summary = f"# Summary Diagnostic Report for: {self.original_query} (Generated on {datetime.today().strftime('%Y-%m-%d')})\n\n"
        summary += f"**Refined Query:** {self.current_query}\n\n"
        summary += "## Comprehensive Diagnostic Report Summary\n"
        # You might use an LLM call here to truly summarize the comprehensive report
        # For now, let's include the comprehensive report as the "summary" for simplicity
        # as your original function also included the full report content.
        summary += comprehensive_report + "\n\n"
        summary += "## Citations\n"
        if citations:
            for citation in citations:
                summary += f"- {citation}\n"
        else:
            summary += "No citations available.\n"

        return summary

    # Adapted to use gemini_client and return content instead of saving file
    async def generate_patient_summary_report_content(self, gemini_client: GeminiClient, comprehensive_report: str) -> str:
        """Generates the content of the patient-friendly summary report."""
        self.logger.info("Generating patient-friendly summary report with clear action steps...")
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
            # Use the gemini_client (LLM client) for patient summary generation
            # Assuming gemini_client.chat is awaitable
            response = await gemini_client.chat(messages)
            simplified_report = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.logger.info("Patient-friendly summary generated.")
            return simplified_report
        except Exception as e:
            self.logger.error(f"Failed to generate patient-friendly summary: {e}")
            return "Patient-friendly summary generation failed."

# --- End Copied/Adapted Functions/Classes ---

# --- API Setup ---

app = FastAPI(
    title="Medical Agent API",
    description="API for the Medical Health Help System",
    version="1.0.0",
)

# --- Global Resources Initialized on Startup ---

vectorizer = None
clf_model = None
gemini_client: Optional[GeminiClient] = None
search_client: Optional[TavilyClient] = None
extractor: Optional[TavilyExtractor] = None
search_service: Optional[WebSearchService] = None
subject_analyzer: Optional[SubjectAnalyzer] = None
supabase_client: Optional[Any] = None # Supabase client

@app.on_event("startup")
async def startup_event():
    logger.info("API startup event triggered.")
    load_dotenv()

    # Initialize Global Clients
    global gemini_client, search_client, extractor, search_service, subject_analyzer, supabase_client

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-exp-04-17") # Or your preferred model

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set.")
        # Depending on your needs, you might raise an exception here to prevent startup
    if not TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY environment variable not set.")
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase URL or Key not set. RAG and query history might be unavailable.")

    # Initialize clients if keys are available
    if GEMINI_API_KEY:
        try:
            analysis_config = AnalysisConfig(model_name=GEMINI_MODEL_NAME, temperature=0.3)
            gemini_client = GeminiClient(api_key=GEMINI_API_KEY, config=analysis_config)
            subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config)
            logger.info("Gemini client and Subject Analyzer initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client or Subject Analyzer: {e}")

    if TAVILY_API_KEY:
        try:
            search_client = TavilyClient(api_key=TAVILY_API_KEY)
            extractor = TavilyExtractor(api_key=TAVILY_API_KEY)
            search_config = SearchConfig() # Assuming SearchConfig is just a placeholder class
            search_service = WebSearchService(search_client, search_config)
            logger.info("Tavily clients and Search Service initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily clients: {e}")

    if SUPABASE_URL and SUPABASE_KEY:
         try:
             from supabase import create_client, Client
             supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
             logger.info("Supabase client initialized.")
         except Exception as e:
             logger.error(f"Failed to initialize Supabase client: {e}")

    # Train Symptom Classifier on startup
    global vectorizer, clf_model
    survey_csv_path = "Survey report_Sheet1.csv" # Ensure this file is in your deployment package
    if os.path.exists(survey_csv_path):
        try:
            vectorizer, clf_model = train_symptom_classifier(survey_csv_path)
            if vectorizer and clf_model:
                 logger.info("Symptom classifier trained/loaded on startup.")
            else:
                 logger.warning("Symptom classifier training failed on startup.")
        except Exception as e:
             logger.error(f"Exception during classifier training on startup: {e}")
    else:
        logger.warning(f"Survey report CSV not found at {survey_csv_path}. Symptom classifier will use dummy data or be unavailable.")
        # Attempt training with dummy data even if file not found
        try:
            vectorizer, clf_model = train_symptom_classifier("nonexistent_file.csv") # This path will trigger the fallback
            if vectorizer and clf_model:
                logger.info("Symptom classifier trained successfully with dummy data.")
            else:
                logger.error("Symptom classifier training failed even with dummy data fallback.")
        except Exception as e:
             logger.error(f"Exception during dummy data classifier training fallback: {e}")


# Pydantic models for API request and response
class DiagnosisRequest(BaseModel):
    query: str = Field(..., description="The patient's medical symptoms or issue.")
    feedback: Optional[str] = Field(None, description="Optional feedback to refine the agent's understanding.")
    include_urls: List[str] = Field([], description="List of URLs to include in the search/extraction.")
    omit_urls: List[str] = Field([], description="List of URLs to omit from search results.")
    search_depth: str = Field("advanced", description="Depth of web search ('basic' or 'advanced').")
    search_breadth: int = Field(10, description="Number of search results to retrieve per query.")
    # File uploads will be handled separately using Form and File in the endpoint


class VisualizationData(BaseModel):
    diagnosis_frequency: Dict[str, int] = Field(..., description="Frequency of predicted diagnoses.")
    sentiment_over_time: List[Dict[str, Any]] = Field(..., description="Sentiment compound score over time.")
    entities_distribution: Dict[str, int] = Field(..., description="Distribution of extracted medical entities.")


class DiagnosisResponse(BaseModel):
    original_query: str = Field(..., description="The original patient query.")
    refined_query: str = Field(..., description="The query used after incorporating feedback.")
    sentiment_analysis: Dict[str, float] = Field(..., description="Sentiment analysis scores.")
    extracted_entities: List[str] = Field(..., description="Medical entities extracted from the query.")
    predicted_diagnosis: str = Field(..., description="Predicted diagnosis.")
    diagnosis_probabilities: Dict[str, float] = Field(..., description="Probabilities for predicted diagnoses.")
    diagnosis_explanation: str = Field(..., description="Explanation for the predicted diagnosis.")
    agent_analysis: Dict[str, Any] = Field(..., description="Agent's detailed understanding of the query.")
    search_results_summary: Dict[str, List[Dict[str, str]]] = Field(..., description="Summary of search results (title and URL).")
    # Note: Returning full extracted content can be very large. Returning a summary or snippet is better.
    # extracted_content_summary: Dict[str, List[Dict[str, Any]]] = Field(..., description="Summary of extracted content.")
    wearable_data_summary: Optional[str] = Field(None, description="Summary of provided wearable data.")
    comprehensive_report: str = Field(..., description="Comprehensive diagnostic report.")
    patient_summary_report: str = Field(..., description="Patient-friendly summary report.")
    citations: List[str] = Field(..., description="List of sources cited in the reports.")
    feedback_history: List[Dict[str, Any]] = Field(..., description="Record of feedback applied.")
    # Visualization data is included directly
    query_trends_data: Optional[VisualizationData] = Field(None, description="Data for query trend visualizations (if available).")


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_endpoint(
    query: str = Form(..., description="The patient's medical symptoms or issue."),
    feedback: Optional[str] = Form(None, description="Optional feedback to refine the agent's understanding."),
    include_urls: Optional[str] = Form(None, description="Comma-separated list of URLs to include."),
    omit_urls: Optional[str] = Form(None, description="Comma-separated list of URLs to omit."),
    search_depth: str = Form("advanced", description="Depth of web search ('basic' or 'advanced')."),
    search_breadth: int = Form(10, description="Number of search results per query."),
    additional_files: List[UploadFile] = File([], description="Additional reference files (PDF, DOCX, CSV, Excel, TXT)."),
    wearable_data_file: Optional[UploadFile] = File(None, description="Wearable data CSV file."),
):
    logger.info(f"Received diagnosis request for query: {query}")

    # --- Initial Analysis (Synchronous) ---
    sentiment_score = analyze_sentiment(query)
    entities = extract_medical_entities(query)

    predicted_diag = "Classifier not available"
    diag_proba: Dict[str, float] = {}
    explanation = "Explanation not available"

    if vectorizer and clf_model:
        try:
            predicted_diag, diag_proba = predict_diagnosis(query, vectorizer, clf_model)
            explanation = explain_diagnosis(query, vectorizer, clf_model)
        except Exception as e:
             logger.error(f"Error during diagnosis prediction or explanation: {e}")
             predicted_diag = "Prediction Error"
             explanation = "Explanation Error"
    else:
        logger.warning("Symptom classifier not available. Skipping prediction and explanation.")


    # --- Save Query History (Asynchronous, depends on Supabase) ---
    if supabase_client:
        # Run saving in the background so the response isn't delayed by DB write
        asyncio.create_task(save_query_history_to_db(supabase_client, query, predicted_diag, sentiment_score, entities))
    else:
        logger.warning("Supabase client not initialized. Skipping query history save.")


    # --- Process Inputs and Initialize Task ---
    include_urls_list = [url.strip() for url in include_urls.split(',') if url.strip()] if include_urls else []
    omit_urls_list = [url.strip() for url in omit_urls.split(',') if url.strip()] if omit_urls else []

    task = MedicalTask(query, logger)

    # Apply feedback if provided
    if feedback:
        task.update_feedback(feedback)

    # Process uploaded additional files
    if additional_files:
        logger.info(f"Processing {len(additional_files)} additional uploaded files.")
        for file in additional_files:
            content = extract_text_from_file_obj(file, logger)
            if content:
                task.additional_files_content[file.filename] = content
                logger.info(f"Extracted content from {file.filename}")
            else:
                 logger.warning(f"Failed to extract content from {file.filename}")

    # Process uploaded wearable data file
    if wearable_data_file:
         logger.info("Processing uploaded wearable data file.")
         task.wearable_summary = read_wearable_data_from_file_obj(wearable_data_file, logger)
         if task.wearable_summary:
             logger.info("Wearable data processed.")
         else:
             logger.warning("Wearable data processing failed.")


    # --- Core Agent Workflow (Asynchronous) ---

    if subject_analyzer is None:
        raise HTTPException(status_code=500, detail="Subject analyzer not initialized.")
    if search_service is None or extractor is None:
         raise HTTPException(status_code=500, detail="Search service or extractor not initialized.")
    if gemini_client is None:
         raise HTTPException(status_code=500, detail="Gemini client not initialized.")
    if supabase_client is None:
         # RAG requires Supabase for vector search. If not initialized, RAG cannot proceed.
         # You could add a fallback here, but for full RAG, Supabase is needed.
         raise HTTPException(status_code=500, detail="Supabase client not initialized. RAG requires Supabase.")


    # Step 1: Analyze Subject
    try:
        await task.analyze(subject_analyzer, datetime.today().strftime("%Y-%m-%d"))
    except Exception as e:
        logger.error(f"Task analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


    # Step 2: Search and Extract (either web search or user-provided URLs)
    if not include_urls_list:
        try:
            await task.search_and_extract(search_service, extractor, omit_urls_list, search_depth, search_breadth)
        except Exception as e:
            logger.error(f"Task search and extract failed: {e}")
            # Decide if search/extract failure is critical or if RAG can proceed with just uploaded files
            # For now, let's allow it to proceed if there are uploaded files
            if not task.additional_files_content:
                 raise HTTPException(status_code=500, detail=f"Search and extraction failed: {e}")
            else:
                 logger.warning("Search and extraction failed, but proceeding with RAG using uploaded files.")

    else:
        try:
            await task.extract_from_urls(extractor, include_urls_list, omit_urls_list)
        except Exception as e:
             logger.error(f"Task extraction from provided URLs failed: {e}")
             if not task.additional_files_content:
                  raise HTTPException(status_code=500, detail=f"Extraction from provided URLs failed: {e}")
             else:
                  logger.warning("Extraction from provided URLs failed, but proceeding with RAG using uploaded files.")


    # Check if there's *any* content for RAG
    if not task.extracted_content and not task.additional_files_content and not task.wearable_summary:
         raise HTTPException(status_code=500, detail="No content extracted or provided for analysis.")


    # Step 3: Analyze Full Content with RAG
    try:
        comprehensive_report, citations = await task.analyze_full_content_rag(gemini_client, supabase_client)
        if "Secondary analysis failed" in comprehensive_report or "Error: Could not generate query embedding" in comprehensive_report:
             # Handle cases where RAG itself returned an error message
             raise Exception(comprehensive_report)

    except Exception as e:
        logger.error(f"Task RAG analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis (RAG) failed: {e}")


    # Step 4: Generate Patient Summary
    try:
        patient_report = await task.generate_patient_summary_report_content(gemini_client, comprehensive_report)
        if "Patient-friendly summary generation failed" in patient_report:
             raise Exception(patient_report)
    except Exception as e:
        logger.error(f"Task patient summary generation failed: {e}")
        # Patient summary failure might not be critical, return comprehensive report and a failure message
        patient_report = f"Could not generate patient-friendly summary: {e}"


    # --- Generate Report Contents (Synchronous) ---
    # These just format the strings and don't need await
    full_report_content = task.generate_report_content(comprehensive_report, citations)
    summary_report_content = task.generate_summary_report_content(comprehensive_report, citations)

    # --- Get Visualization Data (Asynchronous, depends on Supabase) ---
    query_trends_data = None
    if supabase_client:
         try:
             query_trends_data_raw = await get_query_history_data(supabase_client)
             if query_trends_data_raw:
                  query_trends_data = VisualizationData(**query_trends_data_raw) # Validate with Pydantic model
                  logger.info("Retrieved query trends data.")
             else:
                 logger.warning("No query trends data available from Supabase.")
         except Exception as e:
              logger.error(f"Failed to retrieve query trends data from Supabase: {e}")
              query_trends_data = None # Ensure it's None on failure
    else:
         logger.warning("Supabase client not initialized. Cannot retrieve query trends data.")


    # --- Return Response ---
    response_data = DiagnosisResponse(
        original_query=task.original_query,
        refined_query=task.current_query,
        sentiment_analysis=sentiment_score,
        extracted_entities=entities,
        predicted_diagnosis=predicted_diag,
        diagnosis_probabilities=diag_proba,
        diagnosis_explanation=explanation,
        agent_analysis=task.analysis,
        search_results_summary={topic: [{"title": r.get("title"), "url": r.get("url")} for r in results]
                                for topic, results in task.search_results.items()},
        wearable_data_summary=task.wearable_summary,
        comprehensive_report=comprehensive_report,
        patient_summary_report=patient_report,
        citations=citations,
        feedback_history=task.feedback_history,
        query_trends_data=query_trends_data, # Include visualization data
    )

    return JSONResponse(content=response_data.model_dump(), status_code=200) # Use model_dump() for Pydantic v2+


# Optional: Add a root endpoint
@app.get("/")
async def read_root():
    return {"message": "Medical Agent API is running. Send a POST request to /diagnose to get a diagnosis."}

# Optional: Add an endpoint just to get visualization data
@app.get("/query_trends", response_model=Optional[VisualizationData])
async def get_trends_data_endpoint():
     if supabase_client is None:
          raise HTTPException(status_code=500, detail="Supabase client not initialized. Query trends data unavailable.")
     try:
          query_trends_data_raw = await get_query_history_data(supabase_client)
          if query_trends_data_raw:
               return VisualizationData(**query_trends_data_raw)
          else:
               return None # Return 200 with None if no data
     except Exception as e:
          logger.error(f"Failed to retrieve query trends data for endpoint: {e}")
          raise HTTPException(status_code=500, detail=f"Failed to retrieve query trends data: {e}")
