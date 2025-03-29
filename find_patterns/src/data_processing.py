"""Functions for loading and preprocessing BTC-DOGE data."""

import pandas as pd
import numpy as np
import os
from src.feature_engineering import preprocess_data, preprocess_chunk

def load_and_preprocess_data(btc_path, doge_path):
    """Load and preprocess BTC and DOGE data from CSV files."""
    print(f"Loading data from {btc_path} and {doge_path}...")
    
    # Load data
    btc_data = pd.read_csv(btc_path)
    doge_data = pd.read_csv(doge_path)
    
    # Use the preprocess_data function from feature_engineering.py
    combined_data = preprocess_data(btc_data, doge_data)
    
    print(f"Data loaded and preprocessed. Shape: {combined_data.shape}")
    return combined_data

def load_data_in_chunks(file_path, chunk_size=100000):
    """Load and process large data files in chunks."""
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    processed_chunks = []
    
    for chunk in chunks:
        processed_chunk = preprocess_chunk(chunk)
        processed_chunks.append(processed_chunk)
    
    return pd.concat(processed_chunks)