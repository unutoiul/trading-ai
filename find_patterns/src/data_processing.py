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
    
    # Check compatibility before processing
    compatibility_report = check_data_compatibility(btc_data, alt_data)
    
    # Continue with preprocessing if compatibility is acceptable
    if compatibility_report["match_percent"] < 50:
        print("ERROR: Datasets have less than 50% matching timestamps!")
        print("Analysis results may be severely compromised.")
        # You could raise an exception here or continue with a warning
    
    # Use the preprocess_data function from feature_engineering.py
    combined_data = preprocess_data(btc_data, alt_data, alt_prefix=alt_prefix)
    
    # Run another check on the combined data to verify merge quality
    final_report = check_data_compatibility(btc_data, alt_data, combined_data)
    
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

def check_data_compatibility(btc_data, alt_data, combined_data=None):
    """
    Verify compatibility between BTC and altcoin datasets.
    
    Args:
        btc_data: DataFrame with BTC data
        alt_data: DataFrame with altcoin data
        combined_data: Optional already merged DataFrame
        
    Returns:
        Dictionary with compatibility metrics
    """
    # Ensure timestamps are datetime objects
    if 'timestamp' in btc_data.columns and not pd.api.types.is_datetime64_any_dtype(btc_data['timestamp']):
        btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
    
    if 'timestamp' in alt_data.columns and not pd.api.types.is_datetime64_any_dtype(alt_data['timestamp']):
        alt_data['timestamp'] = pd.to_datetime(alt_data['timestamp'])
    
    # Count original timestamps
    btc_timestamps = btc_data['timestamp'].nunique() if 'timestamp' in btc_data.columns else btc_data.index.nunique()
    alt_timestamps = alt_data['timestamp'].nunique() if 'timestamp' in alt_data.columns else alt_data.index.nunique()
    
    # Get common timestamps
    if 'timestamp' in btc_data.columns and 'timestamp' in alt_data.columns:
        btc_ts_set = set(btc_data['timestamp'])
        alt_ts_set = set(alt_data['timestamp'])
    else:
        btc_ts_set = set(btc_data.index)
        alt_ts_set = set(alt_data.index)
    
    common_timestamps = len(btc_ts_set.intersection(alt_ts_set))
    
    # Calculate matching percentage
    match_percent = (common_timestamps / min(btc_timestamps, alt_timestamps)) * 100
    
    # Check combined data if provided
    combined_timestamps = 0
    if combined_data is not None:
        combined_timestamps = len(combined_data)
    
    # Calculate time intervals
    def calc_interval(df):
        if 'timestamp' in df.columns:
            ts_series = df['timestamp'].sort_values()
        else:
            ts_series = pd.Series(df.index).sort_values()
        
        diffs = ts_series.diff().dropna()
        return diffs.median().total_seconds() if not diffs.empty else None
    
    btc_interval = calc_interval(btc_data)
    alt_interval = calc_interval(alt_data)
    
    # Check timezone consistency
    def get_timezone(df):
        if 'timestamp' in df.columns and len(df) > 0:
            if pd.api.types.is_datetime64_dtype(df['timestamp']):
                return df['timestamp'].dt.tz
        elif hasattr(df.index, 'tz'):
            return df.index.tz
        return None
    
    btc_tz = get_timezone(btc_data)
    alt_tz = get_timezone(alt_data)
    
    # Print report
    print("\n=== Data Compatibility Report ===")
    print(f"BTC timestamps: {btc_timestamps}")
    print(f"Altcoin timestamps: {alt_timestamps}")
    print(f"Common timestamps: {common_timestamps} ({match_percent:.2f}%)")
    print(f"Combined dataset size: {combined_timestamps}")
    print(f"BTC time interval: {btc_interval} seconds")
    print(f"Altcoin time interval: {alt_interval} seconds")
    print(f"BTC timezone: {btc_tz}")
    print(f"Altcoin timezone: {alt_tz}")
    
    # Warnings
    if match_percent < 90:
        print("\n⚠️ WARNING: Less than 90% timestamp match between datasets!")
    
    if btc_interval and alt_interval and abs(btc_interval - alt_interval) > 5:
        print(f"\n⚠️ WARNING: Time interval mismatch between datasets!")
    
    if btc_tz != alt_tz:
        print(f"\n⚠️ WARNING: Timezone mismatch between datasets!")
    
    return {
        "btc_timestamps": btc_timestamps,
        "alt_timestamps": alt_timestamps,
        "common_timestamps": common_timestamps,
        "match_percent": match_percent,
        "combined_size": combined_timestamps,
        "btc_interval": btc_interval,
        "alt_interval": alt_interval,
        "btc_timezone": str(btc_tz),
        "alt_timezone": str(alt_tz)
    }