"""
Utility functions for the control mapping application
"""

import re
import string
import json
from typing import List, Dict, Any

def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing punctuation, extra whitespace, and lowercasing.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        print(f"WARNING: Input to preprocess_text is not a string: {type(text)}")
        if text is None:
            return ""
        text = str(text)
    
    # Print original text for debugging (first 50 chars)
    original_sample = text[:50] + "..." if len(text) > 50 else text
    print(f"Original text sample: '{original_sample}'")
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (but keep some meaningful ones like hyphens)
    # Modified to be less aggressive - only replace certain punctuation with spaces
    translator = str.maketrans({p: ' ' for p in string.punctuation if p not in ['-', '_', '.', '/']})
    text = text.translate(translator)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Print processed text for debugging
    processed_sample = text[:50] + "..." if len(text) > 50 else text
    print(f"Processed text sample: '{processed_sample}'")
    
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks

def format_llm_prompt(
    doc_name: str,
    doc_text: str,
    analyst_note: str,
    candidate_controls: List[Dict[str, Any]]
) -> str:
    """
    Format prompt for LLM to evaluate controls.
    
    Args:
        doc_name: Name of cloud service
        doc_text: Security documentation text
        analyst_note: Analyst's security note
        candidate_controls: List of candidate controls
        
    Returns:
        Formatted prompt
    """
    controls_text = "\n\n".join([
        f"Control {i+1}: {control['control_text']}"
        for i, control in enumerate(candidate_controls)
    ])
    
    # Convert candidate controls to JSON format for easy reference by ID
    controls_json = json.dumps([
        {
            "id": control['control_id'],
            "text": control['control_text']
        }
        for control in candidate_controls
    ], indent=2)
    
    prompt = f"""You are a cybersecurity expert evaluating which security controls apply to a cloud service.

CLOUD SERVICE: {doc_name}

SECURITY DOCUMENTATION EXCERPT:
{doc_text[:4000]}...

ANALYST SECURITY NOTE:
{analyst_note}

CANDIDATE CONTROLS:
{controls_text}

TASK:
Analyze the cloud service documentation and analyst note to determine which of the candidate controls should apply to this service.

For each control you determine is applicable:
1. Assign a confidence score (LOW, MEDIUM, or HIGH)
2. Provide a 1-sentence justification of why the control applies

Return your analysis as a JSON object with the following structure:
{{
  "service_name": "{doc_name}",
  "controls": [
    {{
      "control_id": "ID of the control",
      "control_text": "Full text of the control",
      "confidence": "HIGH|MEDIUM|LOW",
      "justification": "One sentence justification"
    }}
  ]
}}

CONTROLS JSON FOR REFERENCE:
{controls_json}

Your response should be ONLY the JSON object, properly formatted without any additional text.
"""
    return prompt 