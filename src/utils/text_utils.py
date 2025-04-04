"""Text processing utilities"""

import re
from typing import List

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s.,;?!-]', '', text)
    return text.strip()

def format_markdown_section(title: str, content: List[str]) -> str:
    """Format a section in markdown"""
    return f"## {title}\n\n" + "\n".join([f"- {item}" for item in content]) + "\n\n"

def create_markdown_document(sections: List[dict]) -> str:
    """Create a markdown document from sections"""
    doc = []
    for section in sections:
        title = section.get("title", "")
        content = section.get("content", [])
        if isinstance(content, list):
            doc.append(format_markdown_section(title, content))
        else:
            doc.append(f"## {title}\n\n{content}\n\n")
    return "".join(doc)

def extract_keywords(text: str, stopwords: set = None) -> List[str]:
    """Extract keywords from text"""
    if stopwords is None:
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
    
    # Tokenize and clean
    words = re.findall(r'\w+', text.lower())
    # Remove stopwords
    keywords = [word for word in words if word not in stopwords]
    
    # Count frequency
    from collections import Counter
    keyword_freq = Counter(keywords)
    
    # Return top keywords
    return [word for word, _ in keyword_freq.most_common(10)] 