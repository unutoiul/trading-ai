"""Technical indicator and feature generation for crypto analysis."""

import pandas as pd
import numpy as np
import ta

def generate_technical_indicators(df, suffix):
    """
    Add technical indicators to the dataframe.
    
    Args:
        df: DataFrame with price data
        suffix: String suffix for column names (e.g., '_btc')
        
    Returns:
        DataFrame with added technical indicators
    """
    print(f"Generating technical indicators for {suffix}...")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # RSI
    df[f'rsi{suffix}'] = ta.momentum.RSIIndicator(
        close=df[f'close{suffix}'], window=14
    ).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df[f'close{suffix}'])
    df[f'macd{suffix}'] = macd.macd()
    df[f'macd_signal{suffix}'] = macd.macd_signal()
    df[f'macd_diff{suffix}'] = macd.macd_diff()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=df[f'high{suffix}'],
        low=df[f'low{suffix}'],
        close=df[f'close{suffix}']
    )
    df[f'stoch_k{suffix}'] = stoch.stoch()
    
    # Rate of Change
    df[f'roc{suffix}'] = ta.momentum.ROCIndicator(
        close=df[f'close{suffix}'], window=9
    ).roc()
    
    return df

def create_momentum_features(df, asset_suffix):
    """
    Create momentum and volatility features at different timeframes.
    
    Args:
        df: DataFrame with returns columns
        asset_suffix: String suffix for the asset (e.g., 'btc')
        
    Returns:
        DataFrame with momentum features
    """
    df = df.copy()
    
    # Create momentum features at different timeframes
    for window in [5, 15, 30, 60]:
        df[f'{asset_suffix}_momentum_{window}m'] = df[f'{asset_suffix}_returns'].rolling(window).sum()
        df[f'{asset_suffix}_volatility_{window}m'] = df[f'{asset_suffix}_returns'].rolling(window).std()
    
    return df

def preprocess_data(btc_data, doge_data):
    """
    Preprocess BTC and DOGE data and merge into a single dataframe.
    
    Args:
        btc_data: DataFrame with BTC data
        doge_data: DataFrame with DOGE data
        
    Returns:
        DataFrame with processed and merged data
    """
    print("Preprocessing data...")
    
    # Convert timestamp to datetime
    btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
    doge_data['timestamp'] = pd.to_datetime(doge_data['timestamp'])
    
    # Set timestamp as index
    btc_data.set_index('timestamp', inplace=True)
    doge_data.set_index('timestamp', inplace=True)
    
    # Merge the data on the index
    print("Merging datasets...")
    combined_data = pd.merge(
        btc_data, 
        doge_data, 
        left_index=True, 
        right_index=True, 
        suffixes=('_btc', '_doge')
    )
    
    # Calculate returns
    combined_data['btc_returns'] = combined_data['close_btc'].pct_change()
    combined_data['doge_returns'] = combined_data['close_doge'].pct_change()
    
    # Drop NaN values created by pct_change
    combined_data.dropna(inplace=True)
    
    # Add technical indicators for both assets
    combined_data = generate_technical_indicators(combined_data, '_btc')
    combined_data = generate_technical_indicators(combined_data, '_doge')
    
    # Create momentum features
    combined_data = create_momentum_features(combined_data, 'btc')
    combined_data = create_momentum_features(combined_data, 'doge')
    
    # Drop NaN values created by rolling windows
    combined_data.dropna(inplace=True)
    
    print(f"Data preprocessed. Final shape: {combined_data.shape}")
    return combined_data

def preprocess_chunk(chunk):
    """
    Preprocess a single chunk of data (for use with large datasets).
    
    Args:
        chunk: DataFrame chunk to process
        
    Returns:
        Processed DataFrame chunk
    """
    # If timestamp is a column, convert it to datetime and set as index
    if 'timestamp' in chunk.columns:
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        chunk.set_index('timestamp', inplace=True)
    
    # Identify asset type (BTC or DOGE) based on column names
    suffix = '_btc' if 'close_btc' in chunk.columns else '_doge'
    
    # Calculate returns
    return_col = f"{'btc' if suffix == '_btc' else 'doge'}_returns"
    chunk[return_col] = chunk[f'close{suffix}'].pct_change()
    
    # Add technical indicators
    chunk = generate_technical_indicators(chunk, suffix)
    
    # Create momentum features
    asset_prefix = 'btc' if suffix == '_btc' else 'doge'
    chunk = create_momentum_features(chunk, asset_prefix)
    
    # Drop NaN values
    chunk.dropna(inplace=True)
    
    return chunk