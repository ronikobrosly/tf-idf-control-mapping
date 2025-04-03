#!/usr/bin/env python3
"""
Control Mapping Application for Cloud Services

This script analyzes cloud service documentation and maps it to relevant security controls
using BM25 retrieval and LLM inference.
"""

import pdb
import argparse
import json
import os
import pandas as pd
import PyPDF2
from typing import Dict, List, Any

from llm_client import LlamaClient
from retrieval import BM25Retriever
from utils import preprocess_text, chunk_text, format_llm_prompt

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Map cloud services to security controls')
    parser.add_argument('--controls', required=True, help='Path to CSV file containing controls')
    parser.add_argument('--name', required=True, help='Name of the cloud service')
    parser.add_argument('--security_start_page', required=True, type=int, help='Start page of security section')
    parser.add_argument('--security_end_page', required=True, type=int, help='End page of security section')
    parser.add_argument('--analyst_note', required=True, help='Analyst note about security concerns')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument('--pdf_path', required=True, help='Path to PDF documentation')
    
    return parser.parse_args()

def read_controls(file_path: str) -> pd.DataFrame:
    """Read control policies from CSV file."""
    try:
        controls = pd.read_csv(file_path, sep = "|")
        print(f"Successfully loaded {len(controls)} controls")
        return controls
    except Exception as e:
        print(f"Error reading controls file: {e}")
        raise

def extract_pdf_text(pdf_path: str, start_page: int, end_page: int) -> str:
    """Extract text from PDF documentation between specified pages."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Adjust for 0-based indexing
            start_idx = max(0, start_page - 1)
            end_idx = min(len(reader.pages), end_page)
            
            for page_num in range(start_idx, end_idx):
                text += reader.pages[page_num].extract_text() + "\n"
        
        print(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        raise

def map_controls(
    controls_df: pd.DataFrame, 
    doc_name: str,
    doc_text: str, 
    analyst_note: str,
    llm_client: LlamaClient
) -> Dict[str, Any]:
    """Map cloud service to controls using BM25 retrieval and LLM."""
    # Preprocess documentation text
    processed_doc = preprocess_text(doc_text)
    
    # Create chunks for processing
    doc_chunks = chunk_text(processed_doc, chunk_size=1000, overlap=200)
    

    # Initialize BM25 retriever
    retriever = BM25Retriever(controls_df)

    # Get top relevant controls for each chunk
    all_relevant_controls = []
    for chunk in doc_chunks:
        relevant_controls = retriever.retrieve(chunk, top_k=10)
        all_relevant_controls.extend(relevant_controls)
    
    # Get top relevant controls based on analyst note
    note_relevant_controls = retriever.retrieve(analyst_note, top_k=10)
    all_relevant_controls.extend(note_relevant_controls)
    
    # Count occurrences of each control to find the most relevant ones
    control_counts = {}
    for control in all_relevant_controls:
        control_id = control['control_id']
        if control_id in control_counts:
            control_counts[control_id]['count'] += 1
            control_counts[control_id]['score'] += control['score']
        else:
            control_counts[control_id] = {
                'control': control,
                'count': 1,
                'score': control['score']
            }
    
    # Sort by count and score
    sorted_controls = sorted(
        control_counts.values(), 
        key=lambda x: (x['count'], x['score']), 
        reverse=True
    )
    
    # Take top 10 controls for LLM evaluation
    candidate_controls = [item['control'] for item in sorted_controls[:10]]
    
    # Format prompt for LLM
    prompt = format_llm_prompt(
        doc_name=doc_name,
        doc_text=processed_doc[:4000],  # First 4000 chars for context
        analyst_note=analyst_note,
        candidate_controls=candidate_controls
    )
    
    # Get LLM response
    llm_response = llm_client.generate(prompt)
    
    # Parse LLM response into structured format
    try:
        result = json.loads(llm_response)
    except json.JSONDecodeError:
        # Fallback parsing if LLM doesn't return valid JSON
        result = {
            "service_name": doc_name,
            "controls": [],
            "error": "Failed to parse LLM response"
        }
        print("Error parsing LLM response. Using fallback output.")
    
    return result

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize LLM client
    llm_client = LlamaClient()
    
    # Read controls
    controls_df = read_controls(args.controls)
    
    # Extract text from PDF
    doc_text = extract_pdf_text(
        args.pdf_path, 
        args.security_start_page, 
        args.security_end_page
    )
    
    # Map controls
    result = map_controls(
        controls_df,
        args.name,
        doc_text,
        args.analyst_note,
        llm_client
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 