import streamlit as st
import google.generativeai as genai
import os
import asyncio
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from dotenv import load_dotenv
from web_agent.src.services.web_search import WebSearchService
from web_agent.src.models.search_models import SearchConfig
from subject_analyzer.src.services.tavily_client import TavilyClient
from subject_analyzer.src.services.tavily_extractor import TavilyExtractor
from subject_analyzer.src.services.subject_analyzer import SubjectAnalyzer
from subject_analyzer.src.services.gemini_client import GeminiClient
from subject_analyzer.src.models.analysis_models import AnalysisConfig
from supabase import create_client, Client
import numpy as np
import faiss
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load environment variables (as a fallback, but primarily using Streamlit secrets)
load_dotenv()

# Fetch secrets from Streamlit
TAVILY_API_KEY = st.secrets["api_keys"]["TAVILY_API_KEY"]
GEMINI_API_KEY = st.secrets["api_keys"]["GEMINI_API_KEY"]
SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]
SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
MODEL_NAME = st.secrets["MODEL_NAME"]
TEMPERATURE = st.secrets["TEMPERATURE"]
MAX_SOURCES = st.secrets["MAX_SOURCES"]
MAX_ITERATIONS = st.secrets["MAX_ITERATIONS"]

# Configure Gemini API
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set it in Streamlit secrets.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Gemini NLP Functions
def analyze_sentiment(query: str) -> dict:
    prompt = f"Analyze the sentiment of the following medical query and return a sentiment score (positive, negative, neutral):\n\n{query}"
    try:
        response = model.generate_content(prompt)
        sentiment = response.text.strip()
        return {"sentiment": sentiment, "compound": 0.5 if "positive" in sentiment.lower() else -0.5 if "negative" in sentiment.lower() else 0.0}
    except Exception as e:
        return {"error": str(e), "compound": 0.0}

def extract_medical_entities(query: str) -> list:
    prompt = f"Extract medical entities (e.g., diseases, symptoms, medications) from the following text:\n\n{query}"
    try:
        response = model.generate_content(prompt)
        entities = response.text.strip().split(", ")
        return [e for e in entities if e]
    except Exception as e:
        return [f"Error extracting entities: {str(e)}"]

def predict_diagnosis(query: str) -> str:
    prompt = f"Based on the following symptoms or medical query, predict a possible diagnosis:\n\n{query}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error predicting diagnosis: {str(e)}"

def generate_patient_summary(comprehensive_report: str) -> str:
    prompt = f"Summarize the following comprehensive diagnostic report in simple, clear language for a patient:\n\n{comprehensive_report}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Visualization Function
def visualize_query_trends():
    filename = "query_history.csv"
    if not os.path.exists(filename):
        st.warning("No query history available for visualization.")
        return
    df = pd.read_csv(filename)
    if df.empty:
        st.warning("Query history is empty.")
        return

    # Bar plot for predicted diagnosis frequency
    diag_counts = df['predicted_diagnosis'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=diag_counts.index, y=diag_counts.values, palette="viridis", ax=ax)
    ax.set_title("Frequency of Predicted Diagnoses")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Line plot for sentiment compound over time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="timestamp", y="sentiment_compound", marker="o", ax=ax)
    ax.set_title("Sentiment Compound Score Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Sentiment Compound Score")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Pie chart for common medical entities
    entity_list = []
    for entities_str in df['entities']:
        if pd.isna(entities_str) or entities_str.strip() == "":
            continue
        entities = [e.strip() for e in entities_str.split(",") if e.strip()]
        entity_list.extend(entities)
    if entity_list:
        entity_counts = Counter(entity_list)
        labels = list(entity_counts.keys())
        sizes = list(entity_counts.values())
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.set_title("Distribution of Extracted Medical Entities")
        st.pyplot(fig)

# Helper Functions
def save_query_history(query, diagnosis, sentiment, entities):
    filename = "query_history.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "query", "predicted_diagnosis", "sentiment_compound", "entities"])
        writer.writerow([datetime.now().isoformat(), query, diagnosis, sentiment.get("compound", 0), ", ".join(entities)])

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "".join(page.extract_text() or "" for page in reader.pages)
        elif ext == '.docx':
            import docx
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        elif ext in ['.csv', '.xls', '.xlsx']:
            df = pd.read_csv(file_path) if ext == '.csv' else pd.read_excel(file_path)
            return df.to_csv(index=False)
        else:
            with open(file_path, 'r', encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        st.warning(f"Could not extract file {file_path}: {e}")
        return ""

# Streamlit App
def main():
    st.set_page_config(page_title="Medical Agent", layout="wide")
    st.title("🏥 Medical Agent")
    st.markdown("An interactive tool to analyze medical queries and generate diagnostic reports.")

    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        search_depth = st.selectbox("Search Depth", ["basic", "advanced"], index=1)
        search_breadth = st.number_input("Search Breadth (results)", min_value=1, max_value=MAX_SOURCES, value=10)
        include_urls = st.text_input("Include URLs (comma-separated)", "")
        omit_urls = st.text_input("Omit URLs (comma-separated)", "")
        own_files = st.text_input("Local Files (comma-separated)", "")
        st.markdown("---")
        st.info("API keys and configurations are managed via Streamlit secrets.")

    # Main Interface
    query = st.text_area("Enter your medical query:", height=150, placeholder="Describe your symptoms or medical issue...")
    col1, col2 = st.columns([1, 1])
    with col1:
        submit_button = st.button("Analyze Query")
    with col2:
        visualize_button = st.button("Visualize Trends")

    if submit_button and query:
        with st.spinner("Processing your query..."):
            # Initialize Services
            analysis_config = AnalysisConfig(model_name=MODEL_NAME, temperature=TEMPERATURE)
            gemini_client = GeminiClient(api_key=GEMINI_API_KEY, config=analysis_config)
            search_client = TavilyClient(api_key=TAVILY_API_KEY)
            extractor = TavilyExtractor(api_key=TAVILY_API_KEY)
            search_service = WebSearchService(search_client, SearchConfig())
            subject_analyzer = SubjectAnalyzer(llm_client=gemini_client, config=analysis_config)
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

            # Process Additional Files
            additional_files = {}
            if own_files:
                for file_path in [f.strip() for f in own_files.split(",")]:
                    if os.path.exists(file_path):
                        additional_files[file_path] = extract_text_from_file(file_path)

            # Medical Task Processing
            class MedicalTask:
                def __init__(self, query):
                    self.original_query = query
                    self.current_query = query
                    self.analysis = {}
                    self.search_results = {}
                    self.extracted_content = {}

                async def analyze(self):
                    self.analysis = subject_analyzer.analyze(f"{self.current_query} (as of {datetime.today().strftime('%Y-%m-%d')})")

                async def search_and_extract(self):
                    topics = [self.analysis.get("main_subject", self.current_query)] + self.analysis.get("What_needs_to_be_researched", [])
                    omit_list = [o.strip() for o in omit_urls.split(",")] if omit_urls else []
                    for topic in topics[:MAX_ITERATIONS]:  # Limit iterations
                        if not topic:
                            continue
                        results = search_service.search_subject(topic, "medical", search_depth=search_depth, results=search_breadth).get("results", [])
                        self.search_results[topic] = [res for res in results[:MAX_SOURCES] if not any(o.lower() in res.get("url", "").lower() for o in omit_list)]
                        urls = [res.get("url") for res in self.search_results[topic]]
                        extraction = extractor.extract(urls=urls, extract_depth="advanced", include_images=False)
                        self.extracted_content[topic] = extraction.get("results", [])

                async def analyze_full_content_rag(self):
                    full_content = "".join(f"\n\n=== Content from {item.get('url', 'No URL')} ===\n{item.get('text') or item.get('raw_content', '')}\n" 
                                          for topic, items in self.extracted_content.items() for item in items)
                    chunks = [full_content[i:i+765] for i in range(0, len(full_content), 765)]
                    embeddings = [model.embed_content(chunk).embeddings[0].values for chunk in chunks]
                    faiss_index = faiss.IndexFlatL2(len(embeddings[0]))
                    faiss_index.add(np.array(embeddings).astype('float32'))
                    query_embedding = model.embed_content(self.current_query).embeddings[0].values
                    distances, indices = faiss_index.search(np.array([query_embedding]).astype('float32'), min(155, len(chunks)))
                    matched_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
                    prompt = f"Generate a comprehensive diagnostic report for: {self.current_query}\nBased on:\n{'\n\n'.join(matched_chunks)}"
                    response = gemini_client.chat([{"role": "user", "content": prompt}])
                    return response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Execute Task
            task = MedicalTask(query)
            asyncio.run(task.analyze())
            if not include_urls:
                asyncio.run(task.search_and_extract())
            else:
                task.search_results = {"User Provided": [{"title": "User Provided", "url": url} for url in include_urls.split(",")]}
                extraction = extractor.extract(urls=[url.strip() for url in include_urls.split(",")], extract_depth="advanced", include_images=False)
                task.extracted_content["User Provided"] = extraction.get("results", [])

            # Display Results
            st.subheader("Analysis")
            sentiment = analyze_sentiment(query)
            entities = extract_medical_entities(query)
            diagnosis = predict_diagnosis(query)
            st.write(f"**Sentiment:** {sentiment.get('sentiment', 'Error')}")
            st.write(f"**Entities:** {', '.join(entities)}")
            st.write(f"**Predicted Diagnosis:** {diagnosis}")
            save_query_history(query, diagnosis, sentiment, entities)

            comprehensive_report = asyncio.run(task.analyze_full_content_rag())
            st.subheader("Comprehensive Report")
            st.markdown(comprehensive_report)

            patient_summary = generate_patient_summary(comprehensive_report)
            st.subheader("Patient-Friendly Summary")
            st.markdown(patient_summary)

            # Download Buttons
            base_filename = ''.join(c if c.isalnum() or c.isspace() else '_' for c in query).strip().replace(' ', '_') or "report"
            st.download_button("Download Full Report", comprehensive_report, f"{base_filename}_full.md")
            st.download_button("Download Patient Summary", patient_summary, f"{base_filename}_patient.md")

    if visualize_button:
        visualize_query_trends()

if __name__ == "__main__":
    main()
