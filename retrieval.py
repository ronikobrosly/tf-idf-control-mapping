"""
TF-IDF retrieval module for finding relevant controls
"""

import pdb

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from utils import preprocess_text

class BM25Retriever:
    """TF-IDF retriever as a replacement for BM25."""
    
    def __init__(self, controls_df: pd.DataFrame):
        """Initialize the TF-IDF retriever with controls data."""
        self.controls_df = controls_df
        self.control_texts = []
        self.processed_texts = []
        
        print(f"Processing {len(controls_df)} controls...")
        # Print DataFrame info for debugging
        print("DataFrame columns:", controls_df.columns.tolist())
        print("First few rows of DataFrame:")
        print(controls_df.head())
        
        # Handle the case where all control text is in a single 'Controls' column
        if 'Controls' in controls_df.columns and len(controls_df.columns) == 1:
            print("Detected single 'Controls' column format")
            
            for idx, row in controls_df.iterrows():
                control_id = str(idx)
                control_text = row['Controls']
                
                # Make sure the control text is valid
                if isinstance(control_text, str) and control_text.strip():
                    combined_text = control_text.strip()
                    processed_text = preprocess_text(combined_text)
                    
                    if processed_text:
                        self.control_texts.append({
                            'control_id': control_id,
                            'text': combined_text,
                            'processed': processed_text
                        })
                        
                        self.processed_texts.append(processed_text)
        else:
            # Original approach for multi-column data
            for idx, row in controls_df.iterrows():
                control_id = row.get('control_id', str(idx))
                
                # Try different possible column names
                possible_text_columns = ['control_text', 'control', 'text', 'policy', 'statement']
                control_text = ""
                for col in possible_text_columns:
                    if col in row and isinstance(row[col], str) and row[col].strip():
                        control_text = row[col].strip()
                        break
                
                possible_desc_columns = ['description', 'desc', 'details', 'rationale']
                control_description = ""
                for col in possible_desc_columns:
                    if col in row and isinstance(row[col], str) and row[col].strip():
                        control_description = row[col].strip()
                        break
                
                # Debug info
                if idx < 5:  # Print details for first 5 rows
                    print(f"Row {idx} - ID: {control_id}")
                    print(f"  Text columns found: {control_text[:50]}...")
                    print(f"  Description columns found: {control_description[:50]}...")
                
                combined_text = f"{control_text} {control_description}".strip()
                
                if not combined_text:
                    if idx < 5:
                        print(f"  WARNING: No text found for control {control_id}")
                    continue
                
                processed_text = preprocess_text(combined_text)
                
                if idx < 5:
                    print(f"  After preprocessing: {processed_text[:50]}...")
                
                # Skip empty texts
                if not processed_text:
                    if idx < 5:
                        print(f"  WARNING: Empty text after preprocessing for control {control_id}")
                    continue
                    
                self.control_texts.append({
                    'control_id': control_id,
                    'text': combined_text,
                    'processed': processed_text
                })
                
                self.processed_texts.append(processed_text)

        print(f"Processed texts count: {len(self.processed_texts)}")
        if not self.processed_texts:
            # Print sample of raw data to help diagnose
            print("\nRAW DATA SAMPLE:")
            for idx, row in controls_df.head(3).iterrows():
                print(f"Row {idx}:")
                for col, val in row.items():
                    print(f"  {col}: {val}")
            
            raise ValueError("No valid control texts found after preprocessing. Check column names and data formatting.")
        
        print(f"Building TF-IDF model with {len(self.processed_texts)} valid controls...")
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_texts)
        print("TF-IDF model built successfully")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant controls for a query using TF-IDF.
        
        Args:
            query: The query text
            top_k: Number of results to return
            
        Returns:
            List of relevant controls with scores
        """
        try:
            processed_query = preprocess_text(query)
            
            # Handle empty query
            if not processed_query:
                return []
                
            # Transform query to TF-IDF space
            query_vec = self.vectorizer.transform([processed_query])
            
            # Calculate cosine similarity
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(cosine_similarities)[-min(top_k, len(cosine_similarities)):][::-1]
            
            # Return relevant controls with scores
            results = []
            for idx in top_indices:
                if cosine_similarities[idx] > 0:  # Only include positive scores
                    control = self.control_texts[idx]
                    results.append({
                        'control_id': control['control_id'],
                        'control_text': control['text'],
                        'score': float(cosine_similarities[idx])
                    })
            
            return results
        except Exception as e:
            print(f"Error in TF-IDF retrieval: {e}")
            # Return empty list on error instead of crashing
            return [] 