"""Enhanced multi-timeframe volatility breakout analysis for crypto trading."""

import pandas as pd
import numpy as np
from .volatility_breakout_detector import VolatilityBreakoutDetector

def add_multi_timeframe_features(df, prefix='btc', timeframes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """Add multi-timeframe volatility breakout features using the new detector for 1-10 minute analysis."""
    # Track original columns to return what was added
    orig_cols = df.columns.tolist()
    
    try:
        # Initialize the volatility breakout detector
        detector = VolatilityBreakoutDetector(df, prefix)
        
        # Generate all breakout conditions
        breakout_df = detector.generate_all_conditions(timeframes=timeframes)
        
        # Add all new breakout features to the dataframe
        for col in breakout_df.columns:
            if col not in df.columns:  # Only add new columns
                df[col] = breakout_df[col]
        
        print(f"Added {len([col for col in breakout_df.columns if col not in orig_cols])} multi-timeframe breakout features for {prefix.upper()}")
        
    except Exception as e:
        print(f"Warning: Could not add multi-timeframe features for {prefix}: {e}")
        # Fallback to legacy features
        add_legacy_momentum_features(df, prefix)
    
    # Return names of newly added columns
    new_cols = [col for col in df.columns if col not in orig_cols]
    return new_cols

def add_legacy_momentum_features(df, prefix='btc'):
    """Fallback: Add basic momentum features if breakout detector fails."""
    # Find price column
    close_col = f'{prefix}_close'
    if close_col not in df.columns:
        close_col = f'close_{prefix}'
        if close_col not in df.columns:
            print(f"Warning: Could not find close price column for {prefix}")
            return []
    
    # Enhanced momentum over multiple timeframes
    for period in [3, 5, 10, 15, 30]:
        # Price momentum (returns over period)
        df[f'{prefix}_momentum_{period}'] = df[close_col].pct_change(period)
        
        # Acceleration (change in momentum)
        if period > 5:
            df[f'{prefix}_accel_{period}'] = df[f'{prefix}_momentum_{period}'].diff(3)
    
    # Simple volatility
    df[f'{prefix}_volatility_15'] = df[f'{prefix}_returns'].rolling(15).std() if f'{prefix}_returns' in df.columns else None
    
    return [f'{prefix}_momentum_{p}' for p in [3, 5, 10, 15, 30]]

def add_enhanced_momentum_features(df, prefix='btc'):
    """Add enhanced momentum features with volume weighting."""
    # Track original columns to return what was added
    orig_cols = df.columns.tolist()
    
    # Find price column
    close_col = f'{prefix}_close'
    if close_col not in df.columns:
        close_col = f'close_{prefix}'
        if close_col not in df.columns:
            print(f"Warning: Could not find close price column for {prefix}")
            return []
    
    # Volume column
    volume_col = f'{prefix}_volume'
    if volume_col not in df.columns:
        volume_col = f'volume_{prefix}'
        if volume_col not in df.columns:
            print(f"Warning: Could not find volume column for {prefix}")
            volume_col = None
    
    # Enhanced momentum over multiple timeframes focused on breakouts
    for period in [3, 5, 10, 15, 30, 60]:
        # Price momentum (returns over period)
        df[f'{prefix}_momentum_{period}'] = df[close_col].pct_change(period)
        
        # Acceleration (change in momentum)
        if period > 5:
            df[f'{prefix}_accel_{period}'] = df[f'{prefix}_momentum_{period}'].diff(3)
        
        # Volume-weighted momentum
        if volume_col:
            # Calculate volume change
            vol_change = df[volume_col].pct_change(period)
            # Volume-weighted momentum
            df[f'{prefix}_vol_weighted_mom_{period}'] = df[f'{prefix}_momentum_{period}'] * vol_change.abs()
    
    # Relative strength (ratio of upward vs downward movement)
    for period in [14, 30]:
        # Get price changes
        changes = df[close_col].diff()
        # Calculate up and down moves
        up_moves = changes.copy()
        up_moves[up_moves < 0] = 0
        down_moves = changes.copy()
        down_moves[down_moves > 0] = 0
        down_moves = down_moves.abs()
        
        # Calculate rolling averages
        avg_up = up_moves.rolling(period).mean()
        avg_down = down_moves.rolling(period).mean()
        
        # Calculate relative strength
        rs = avg_up / avg_down
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
        df[f'{prefix}_rel_strength_{period}'] = rs
    
    # Return names of newly added columns
    new_cols = [col for col in df.columns if col not in orig_cols]
    return new_cols

def add_composite_features(df, btc_prefix='btc', alt_prefix='doge'):
    """Create composite features based on price action relationships."""
    # Track original columns to return what was added
    orig_cols = df.columns.tolist()
    
    # Ensure we have the basic columns
    btc_returns = f'{btc_prefix}_returns'
    alt_returns = f'{alt_prefix}_returns'
    
    if btc_returns not in df.columns or alt_returns not in df.columns:
        print(f"Warning: Missing basic return columns for composite features")
        return []
    
    # Divergence/convergence (correlation-based)
    # Calculate whether BTC and altcoin are moving in the same direction
    df['btc_alt_aligned'] = (df[btc_returns] * df[alt_returns]) > 0
    
    # Simple relative strength: altcoin returns divided by BTC returns
    # Avoid division by zero
    df['alt_btc_rel_strength'] = df[alt_returns] / df[btc_returns].replace(0, np.nan)
    df['alt_btc_rel_strength'] = df['alt_btc_rel_strength'].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0
    df['alt_btc_rel_strength'] = df['alt_btc_rel_strength'].fillna(0)
    
    # Relative momentum over different timeframes
    for period in [5, 15, 30]:
        # Get momentum columns, create if they don't exist
        btc_mom_col = f'{btc_prefix}_momentum_{period}'
        alt_mom_col = f'{alt_prefix}_momentum_{period}'
        
        if btc_mom_col not in df.columns:
            df[btc_mom_col] = df[f'{btc_prefix}_close'].pct_change(period)
            
        if alt_mom_col not in df.columns:
            df[alt_mom_col] = df[f'{alt_prefix}_close'].pct_change(period)
        
        # Calculate relative momentum
        df[f'alt_btc_rel_momentum_{period}'] = df[alt_mom_col] / df[btc_mom_col].replace(0, np.nan)
        df[f'alt_btc_rel_momentum_{period}'] = df[f'alt_btc_rel_momentum_{period}'].replace([np.inf, -np.inf], np.nan)
        df[f'alt_btc_rel_momentum_{period}'] = df[f'alt_btc_rel_momentum_{period}'].fillna(0)
        
        # Momentum alignment (same direction)
        df[f'momentum_aligned_{period}'] = (df[btc_mom_col] * df[alt_mom_col]) > 0
    
    # Create BTC directional movement features (simplified)
    df['btc_strong_up'] = df[btc_returns] > 0.001
    df['btc_medium_up'] = (df[btc_returns] > 0.0005) & (df[btc_returns] <= 0.001)
    df['btc_small_up'] = (df[btc_returns] > 0) & (df[btc_returns] <= 0.0005)
    df['btc_small_down'] = (df[btc_returns] < 0) & (df[btc_returns] >= -0.0005)
    df['btc_medium_down'] = (df[btc_returns] < -0.0005) & (df[btc_returns] >= -0.001)
    df['btc_strong_down'] = df[btc_returns] < -0.001
    
    # Return names of newly added columns
    new_cols = [col for col in df.columns if col not in orig_cols]
    return new_cols

def add_price_action_features(df, prefix='btc'):
    """Add essential price action features only."""
    # Track original columns to return what was added
    orig_cols = df.columns.tolist()
    
    # Find the appropriate columns regardless of naming convention
    open_col = next((col for col in df.columns if f'{prefix}_open' in col or f'open_{prefix}' in col), None)
    high_col = next((col for col in df.columns if f'{prefix}_high' in col or f'high_{prefix}' in col), None)
    low_col = next((col for col in df.columns if f'{prefix}_low' in col or f'low_{prefix}' in col), None)
    close_col = next((col for col in df.columns if f'{prefix}_close' in col or f'close_{prefix}' in col), None)
    volume_col = next((col for col in df.columns if f'{prefix}_volume' in col or f'volume_{prefix}' in col), None)
    
    # Check if we found all necessary columns
    if not all([open_col, high_col, low_col, close_col]):
        print(f"Warning: Missing price columns for {prefix}")
        return []
    
    # Price changes
    df[f'{prefix}_returns'] = df[close_col].pct_change()
    df[f'{prefix}_range_pct'] = (df[high_col] - df[low_col]) / df[close_col]
    
    # Basic candle patterns
    df[f'{prefix}_body_size'] = abs(df[close_col] - df[open_col])
    df[f'{prefix}_upper_wick'] = df[high_col] - df[[close_col, open_col]].max(axis=1)
    df[f'{prefix}_lower_wick'] = df[[close_col, open_col]].min(axis=1) - df[low_col]
    
    # Direction
    df[f'{prefix}_bullish'] = df[close_col] > df[open_col]
    df[f'{prefix}_strong_up'] = (df[f'{prefix}_returns'] > 0.001)
    df[f'{prefix}_medium_up'] = (df[f'{prefix}_returns'] > 0.0005) & (df[f'{prefix}_returns'] <= 0.001)
    df[f'{prefix}_small_up'] = (df[f'{prefix}_returns'] > 0) & (df[f'{prefix}_returns'] <= 0.0005)
    df[f'{prefix}_small_down'] = (df[f'{prefix}_returns'] < 0) & (df[f'{prefix}_returns'] >= -0.0005)
    df[f'{prefix}_medium_down'] = (df[f'{prefix}_returns'] < -0.0005) & (df[f'{prefix}_returns'] >= -0.001)
    df[f'{prefix}_strong_down'] = (df[f'{prefix}_returns'] < -0.001)
    
    # Calculate momentum over different timeframes
    for period in [5, 15, 30]:
        df[f'{prefix}_momentum_{period}'] = df[close_col].pct_change(period)
        if volume_col:
            df[f'{prefix}_volume_momentum_{period}'] = df[volume_col].pct_change(period)
    
    # Simple volatility
    df[f'{prefix}_volatility_15'] = df[f'{prefix}_returns'].rolling(15).std()
    
    # Return names of newly added columns
    new_cols = [col for col in df.columns if col not in orig_cols]
    return new_cols

def add_relationship_features(df, btc_prefix='btc', alt_prefix='doge'):
    """Calculate the relationship between BTC and altcoin price action."""
    # Track original columns to return what was added
    orig_cols = df.columns.tolist()
    
    # Find appropriate return columns
    btc_returns = f'{btc_prefix}_returns'
    alt_returns = f'{alt_prefix}_returns'
    
    if btc_returns not in df.columns or alt_returns not in df.columns:
        print(f"Warning: Missing return columns for {btc_prefix} or {alt_prefix}")
        # Try to calculate returns if we can
        btc_close = next((col for col in df.columns if f'{btc_prefix}_close' in col or f'close_{btc_prefix}' in col), None)
        alt_close = next((col for col in df.columns if f'{alt_prefix}_close' in col or f'close_{alt_prefix}' in col), None)
        
        if btc_close:
            df[btc_returns] = df[btc_close].pct_change()
        
        if alt_close:
            df[alt_returns] = df[alt_close].pct_change()
    
    # Only continue if we have returns
    if btc_returns in df.columns and alt_returns in df.columns:
        # Calculate correlation over different windows
        for window in [15, 30, 60]:
            df[f'btc_alt_corr_{window}'] = df[btc_returns].rolling(window).corr(df[alt_returns])
        
        # Calculate beta (how much altcoin moves relative to BTC)
        df['alt_btc_beta_30'] = (
            df[alt_returns].rolling(30).cov(df[btc_returns]) / 
            df[btc_returns].rolling(30).var()
        )
        
        # Replace inf/NaN values
        df['alt_btc_beta_30'] = df['alt_btc_beta_30'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Return names of newly added columns
    new_cols = [col for col in df.columns if col not in orig_cols]
    return new_cols

def preprocess_data(btc_data, alt_data, alt_prefix=None):
    """Preprocess BTC and altcoin data with multi-timeframe volatility breakout analysis."""
    print("Preprocessing data with multi-timeframe volatility breakout focus...")
    
    # Detect altcoin prefix if not provided
    if alt_prefix is None:
        alt_prefix = 'alt'  # Default name
        for col in alt_data.columns:
            if 'close' in col.lower() and 'btc' not in col.lower():
                parts = col.lower().split('_')
                for part in parts:
                    if part not in ['close', 'price', 'btc', 'usd', 'usdt']:
                        alt_prefix = part
                        break
    
    print(f"Processing altcoin data with prefix: {alt_prefix}")
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in btc_data.columns:
        btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
        btc_data.set_index('timestamp', inplace=True)
    
    if 'timestamp' in alt_data.columns:
        alt_data['timestamp'] = pd.to_datetime(alt_data['timestamp'])
        alt_data.set_index('timestamp', inplace=True)
    
    # Merge the data on the index
    print("Merging datasets...")
    combined_data = pd.merge(
        btc_data, 
        alt_data, 
        left_index=True, 
        right_index=True, 
        suffixes=(f'_btc', f'_{alt_prefix}')
    )
    
    # Standardize column names for the breakout detector
    # BTC columns
    if 'close_btc' in combined_data.columns:
        combined_data['btc_close'] = combined_data['close_btc']
        combined_data['btc_high'] = combined_data['high_btc']
        combined_data['btc_low'] = combined_data['low_btc']
        combined_data['btc_open'] = combined_data['open_btc']
        if 'volume_btc' in combined_data.columns:
            combined_data['btc_volume'] = combined_data['volume_btc']
    
    # Altcoin columns
    alt_close_col = f'close_{alt_prefix}'
    if alt_close_col in combined_data.columns:
        combined_data[f'{alt_prefix}_close'] = combined_data[alt_close_col]
        combined_data[f'{alt_prefix}_high'] = combined_data[f'high_{alt_prefix}']
        combined_data[f'{alt_prefix}_low'] = combined_data[f'low_{alt_prefix}']
        combined_data[f'{alt_prefix}_open'] = combined_data[f'open_{alt_prefix}']
        if f'volume_{alt_prefix}' in combined_data.columns:
            combined_data[f'{alt_prefix}_volume'] = combined_data[f'volume_{alt_prefix}']
    
    # Calculate basic returns (legacy compatibility)
    combined_data['btc_returns'] = combined_data['btc_close'].pct_change().replace([np.inf, -np.inf], np.nan)
    combined_data[f'{alt_prefix}_returns'] = combined_data[f'{alt_prefix}_close'].pct_change().replace([np.inf, -np.inf], np.nan)
    
    # Add multi-timeframe volatility breakout features
    print("Adding multi-timeframe breakout features for BTC...")
    add_multi_timeframe_features(combined_data, 'btc', timeframes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    print(f"Adding multi-timeframe breakout features for {alt_prefix.upper()}...")
    add_multi_timeframe_features(combined_data, alt_prefix, timeframes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Add basic price action features for backward compatibility
    add_price_action_features(combined_data, 'btc')
    add_price_action_features(combined_data, alt_prefix)
    
    # Add relationship features
    add_relationship_features(combined_data, 'btc', alt_prefix)
    
    # Store the altcoin name in a column for easier access later
    combined_data['altcoin_name'] = alt_prefix
    
    # Fill NaNs with appropriate values
    # For momentum and returns, fill with 0
    momentum_cols = [col for col in combined_data.columns if 'momentum' in col or 'returns' in col or 'return_' in col]
    combined_data[momentum_cols] = combined_data[momentum_cols].fillna(0)
    
    # For boolean columns (breakout conditions), fill with False
    boolean_cols = [col for col in combined_data.columns if 
                   any(pattern in col for pattern in ['_breakout_', '_strong_', '_sustained_'])]
    combined_data[boolean_cols] = combined_data[boolean_cols].fillna(False)
    
    # For other indicators, use forward fill then backward fill
    remaining_cols = [col for col in combined_data.columns if col not in momentum_cols + boolean_cols]
    combined_data[remaining_cols] = combined_data[remaining_cols].ffill().bfill()
    
    print(f"Data loaded and preprocessed with multi-timeframe volatility analysis. Shape: {combined_data.shape}")
    print(f"Breakout conditions available: {len(boolean_cols)}")
    return combined_data

def preprocess_chunk(chunk, add_features=True):
    """
    Preprocess a single chunk of data for price action analysis.
    
    Args:
        chunk: DataFrame chunk to process
        add_features: Whether to add price action features or just do basic preprocessing
        
    Returns:
        Processed DataFrame chunk
    """
    # If timestamp is a column, convert it to datetime and set as index
    if 'timestamp' in chunk.columns:
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        chunk.set_index('timestamp', inplace=True)
    
    # Identify if we have BTC and/or altcoin data
    has_btc = False
    altcoin_prefix = None
    
    # Check column names to identify which assets we have
    for col in chunk.columns:
        col_lower = col.lower()
        if 'btc' in col_lower and ('close' in col_lower or 'price' in col_lower):
            has_btc = True
        elif 'close' in col_lower and 'btc' not in col_lower:
            # Try to extract altcoin prefix
            parts = col_lower.split('_')
            for part in parts:
                if part not in ['close', 'price', 'btc', 'usd', 'usdt']:
                    altcoin_prefix = part
                    break
    
    # If we couldn't detect, use defaults
    if not has_btc:
        print("Warning: No BTC data detected in chunk")
    
    if not altcoin_prefix:
        altcoin_prefix = 'alt'
    
    # Only add features if requested
    if add_features:
        # For BTC data
        if has_btc:
            try:
                # First standardize column names
                for base_col in ['open', 'high', 'low', 'close', 'volume']:
                    if f'{base_col}_btc' in chunk.columns:
                        chunk[f'btc_{base_col}'] = chunk[f'{base_col}_btc']
                    elif f'btc_{base_col}' in chunk.columns:
                        pass  # Already has the right format
                
                # Calculate returns if not already present
                if 'btc_returns' not in chunk.columns:
                    btc_close_col = next((col for col in chunk.columns 
                                        if col == 'btc_close' or col == 'close_btc'), None)
                    if btc_close_col:
                        chunk['btc_returns'] = chunk[btc_close_col].pct_change()
                
                # Add price action features for BTC
                add_price_action_features(chunk, 'btc')
                
                # Add enhanced momentum features for BTC
                add_enhanced_momentum_features(chunk, 'btc')
            except Exception as e:
                print(f"Warning: Could not add BTC features: {e}")
        
        # For altcoin data
        if altcoin_prefix:
            try:
                # First standardize column names
                for base_col in ['open', 'high', 'low', 'close', 'volume']:
                    alt_col1 = f'{base_col}_{altcoin_prefix}'
                    alt_col2 = f'{altcoin_prefix}_{base_col}'
                    if alt_col1 in chunk.columns:
                        chunk[alt_col2] = chunk[alt_col1]
                    elif alt_col2 in chunk.columns:
                        pass  # Already has the right format
                
                # Calculate altcoin returns
                if f'{altcoin_prefix}_returns' not in chunk.columns:
                    alt_close_col = next((col for col in chunk.columns 
                                        if col == f'{altcoin_prefix}_close' or col == f'close_{altcoin_prefix}'), None)
                    if alt_close_col:
                        chunk[f'{altcoin_prefix}_returns'] = chunk[alt_close_col].pct_change()
                
                # Add price action features for altcoin
                add_price_action_features(chunk, altcoin_prefix)
                
                # If we have both BTC and altcoin, add relationship features
                if has_btc:
                    add_relationship_features(chunk, 'btc', altcoin_prefix)
                    add_composite_features(chunk, 'btc', altcoin_prefix)
            except Exception as e:
                print(f"Warning: Could not add {altcoin_prefix} features: {e}")
    
    # Fill NaN values - focus only on returns and momentum
    momentum_cols = [col for col in chunk.columns if 'momentum' in col or 'returns' in col]
    chunk[momentum_cols] = chunk[momentum_cols].fillna(0)
    
    # For other columns, use forward fill then backward fill
    chunk = chunk.ffill().bfill()
    
    return chunk