"""Web utilities for research"""

from typing import Dict
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from .text_utils import clean_text

def extract_domain(url: str) -> str:
    """Extract domain name from URL"""
    parsed = urlparse(url)
    return parsed.netloc

def validate_source(url: str) -> Dict:
    """Validate and score a research source"""
    domain = extract_domain(url)
    score = 0
    academic_domains = {'.edu', '.gov', '.org', '.ac.uk'}
    
    # Score based on domain
    if any(domain.endswith(ac_dom) for ac_dom in academic_domains):
        score += 2
    
    # Score based on HTTPS
    if url.startswith('https'):
        score += 1
    
    return {
        'url': url,
        'domain': domain,
        'score': score,
        'is_academic': score >= 2
    }

def extract_webpage_content(url: str) -> str:
    """Extract main content from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Extract text content
        text = soup.get_text(separator=' ')
        return clean_text(text)
    
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return ""

def create_citation(url: str, title: str, authors: list, date: str) -> str:
    """Create academic citation in APA format"""
    if not authors:
        authors = ["N.A."]
    
    authors_str = ", ".join(authors[:-1])
    if len(authors) > 1:
        authors_str += f", & {authors[-1]}"
    else:
        authors_str = authors[0]
    
    return f"{authors_str}. ({date}). {title}. Retrieved from {url}" 