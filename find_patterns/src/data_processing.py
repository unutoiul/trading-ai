"""Functions for loading and preprocessing BTC-DOGE data."""

import pandas as pd
import numpy as np
import os
from src.feature_engineering import preprocess_data, preprocess_chunk

def load_and_preprocess_data(btc_path, alt_path):
    """Load and preprocess BTC and altcoin data from CSV files."""
    print(f"Loading data from {btc_path} and {alt_path}...")
    
    # Extract altcoin name from the file path
    alt_filename = os.path.basename(alt_path)
    alt_prefix = None
    
    # Try different extraction patterns
    # Pattern 1: <symbol>_USDT_*.csv (like ETH_USDT_1m.csv)
    if '_USDT_' in alt_filename:
        alt_prefix = alt_filename.split('_USDT_')[0].lower()
    # Pattern 2: *_<symbol>.csv (like data_ETH.csv)
    elif alt_filename.count('_') > 0:
        alt_prefix = alt_filename.split('_')[-1].split('.')[0].lower()
    # Pattern 3: <symbol>.csv (like ETH.csv)
    else:
        alt_prefix = alt_filename.split('.')[0].lower()
    
    print(f"Detected altcoin: {alt_prefix.upper() if alt_prefix else 'Unknown'}")
    
    # Load data
    btc_data = pd.read_csv(btc_path)
    alt_data = pd.read_csv(alt_path)
    
    # Use the preprocess_data function from feature_engineering.py with the detected alt_prefix
    combined_data = preprocess_data(btc_data, alt_data, alt_prefix=alt_prefix)
    
    # Store the altcoin name in a column for easier access later
    if 'altcoin_name' not in combined_data.columns:
        combined_data['altcoin_name'] = alt_prefix
    
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