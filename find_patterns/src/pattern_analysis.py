"""Pattern detection and analysis for BTC-DOGE momentum relationships."""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.strategy_optimizer import StrategyOptimizer


def calculate_pattern_success(df, pattern_series):
    """
    Calculate the success rate of a given pattern.
    
    Args:
        df: DataFrame with indicators and momentum metrics
        pattern_series: Series with binary pattern occurrences
        
    Returns:
        float: Success rate of the pattern
    """
    # Placeholder implementation
    return np.random.rand()

def classify_momentum_patterns(df, confidence_threshold=0.8):
    """Classify different types of momentum patterns in both BTC and altcoin."""
    print("Identifying momentum patterns for both assets...")
    patterns = pd.DataFrame(index=df.index)

    # Find altcoin prefix from column names
    alt_prefix = None
    for col in df.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            alt_prefix = col.split('_')[0]
            break
    
    if not alt_prefix:
        alt_prefix = 'alt'
        print("WARNING: Could not detect altcoin prefix. Using 'alt' as default.")

    # ===== Define BTC Patterns =====
    # Pattern 1: Strong Upward Momentum
    patterns['btc_pattern_strong_up'] = (
        (df['btc_momentum_15m'] > 0) &
        (df['btc_returns'] > 0) &
        (df['rsi_btc'] > 60) &
        (df['btc_macd'] > df['btc_macd_signal'])
    ).fillna(False)

    # Pattern 2: Strong Downward Momentum
    patterns['btc_pattern_strong_down'] = (
        (df['btc_momentum_15m'] < 0) &
        (df['btc_returns'] < 0) &
        (df['rsi_btc'] < 40) &
        (df['btc_macd'] < df['btc_macd_signal'])
    ).fillna(False)

    # Enhanced Pattern 2: Strong Downward Momentum with Volume Confirmation
    patterns['btc_pattern_strong_down_enhanced'] = (
        (df['btc_momentum_15m'] < 0) &
        (df['btc_returns'] < 0) &
        (df['rsi_btc'] < 40) &
        (df['btc_macd'] < df['btc_macd_signal']) &
        (df['volume_btc'] > df['volume_btc'].rolling(20).mean())  # Add volume confirmation
    ).fillna(False)

    # Pattern 3: BTC Volume Spike with Price Increase
    patterns['btc_pattern_volume_spike_up'] = (
        (df['volume_btc'] > df['volume_btc'].rolling(20, min_periods=5).mean() * 2) &
        (df['btc_returns'] > 0)
    ).fillna(False)

    # Pattern 4: BTC Volume Spike with Price Decrease
    patterns['btc_pattern_volume_spike_down'] = (
        (df['volume_btc'] > df['volume_btc'].rolling(20, min_periods=5).mean() * 2) &
        (df['btc_returns'] < 0)
    ).fillna(False)

    # Pattern 5: BTC Volatility Breakout
    patterns['btc_pattern_volatility_breakout'] = (
        (df['btc_volatility_15m'] > df['btc_volatility_15m'].rolling(100, min_periods=20).mean() * 1.5) &
        (abs(df['btc_returns']) > abs(df['btc_returns']).rolling(30, min_periods=5).mean() * 2)
    ).fillna(False)

    # Pattern 6: BTC Bollinger Band Breakout Up
    patterns['btc_pattern_bb_breakout_up'] = (
        (df['close_btc'] > df['btc_bb_upper']) &
        (df['close_btc'].shift(1) <= df['btc_bb_upper'].shift(1))
    ).fillna(False)

    # Pattern 7: BTC Bollinger Band Breakout Down
    patterns['btc_pattern_bb_breakout_down'] = (
        (df['close_btc'] < df['btc_bb_lower']) &
        (df['close_btc'].shift(1) >= df['btc_bb_lower'].shift(1))
    ).fillna(False)

    # Pattern 8: BTC Stochastic Overbought
    patterns['btc_pattern_stoch_overbought'] = (
        (df['btc_stoch_k'] > 80) &
        (df['btc_stoch_d'] > 80) &
        (df['btc_stoch_k'].shift(1) <= df['btc_stoch_d'].shift(1)) &
        (df['btc_stoch_k'] > df['btc_stoch_d'])
    ).fillna(False)

    # Pattern 9: BTC Stochastic Oversold
    patterns['btc_pattern_stoch_oversold'] = (
        (df['btc_stoch_k'] < 20) &
        (df['btc_stoch_d'] < 20) &
        (df['btc_stoch_k'].shift(1) >= df['btc_stoch_d'].shift(1)) &
        (df['btc_stoch_k'] < df['btc_stoch_d'])
    ).fillna(False)

    # Pattern 10: Bullish MACD Cross
    patterns['btc_pattern_macd_cross_up'] = (
        (df['btc_macd'] > df['btc_macd_signal']) &
        (df['btc_macd'].shift(1) <= df['btc_macd_signal'].shift(1))
    ).fillna(False)

    # Pattern 11: Bearish MACD Cross
    patterns['btc_pattern_macd_cross_down'] = (
        (df['btc_macd'] < df['btc_macd_signal']) &
        (df['btc_macd'].shift(1) >= df['btc_macd_signal'].shift(1))
    ).fillna(False)

    # Pattern: Altcoin follows BTC up
    patterns[f'btc_pattern_{alt_prefix}_follows_btc_up'] = (
        (df['btc_returns'].shift(1) > 0) & 
        (df[f'{alt_prefix}_returns'] > 0) &
        (df[f'{alt_prefix}_returns'] > df[f'{alt_prefix}_returns'].rolling(10, min_periods=3).mean())
    ).fillna(False)

    # Pattern: Altcoin follows BTC down
    patterns[f'btc_pattern_{alt_prefix}_follows_btc_down'] = (
        (df['btc_returns'].shift(1) < 0) & 
        (df[f'{alt_prefix}_returns'] < 0) &
        (df[f'{alt_prefix}_returns'] < df[f'{alt_prefix}_returns'].rolling(10, min_periods=3).mean())
    ).fillna(False)

    # Pattern: Altcoin diverges from BTC (BTC up, alt down)
    patterns[f'btc_pattern_{alt_prefix}_diverges_btc_up'] = (
        (df['btc_returns'] > 0) & 
        (df[f'{alt_prefix}_returns'] < 0)
    ).fillna(False)

    # Pattern: Altcoin diverges from BTC (BTC down, alt up)
    patterns[f'btc_pattern_{alt_prefix}_diverges_btc_down'] = (
        (df['btc_returns'] < 0) & 
        (df[f'{alt_prefix}_returns'] > 0)
    ).fillna(False)

    # Pattern: BTC leads altcoin with lag
    for lag in [1, 3, 5, 10]:
        patterns[f'btc_pattern_leads_{alt_prefix}_lag_{lag}'] = (
            (df['btc_returns'].shift(lag) > 0) & 
            (df[f'{alt_prefix}_returns'] > 0) &
            (abs(df['btc_returns'].shift(lag)) > abs(df[f'{alt_prefix}_returns'].rolling(5, min_periods=2).mean()))
        ).fillna(False)

    # Correct implementation for combined patterns
    patterns['btc_pattern_both_up'] = patterns['btc_pattern_strong_up'] & patterns[f'btc_pattern_{alt_prefix}_follows_btc_up']

    # Print pattern instance counts
    for col in patterns.columns:
        count = patterns[col].sum()
        print(f"Pattern {col}: {count} instances ({count/len(patterns)*100:.1f}% of {len(patterns)} rows)")

    # Store the altcoin name in a column that can be accessed later
    if len(patterns.index) > 0:
        patterns.loc[patterns.index[0], 'altcoin_name'] = alt_prefix
    else:
        print("WARNING: Empty patterns DataFrame. Could not store altcoin name.")
        # Create an empty Series with the altcoin name for reference
        patterns = pd.DataFrame({'altcoin_name': [alt_prefix]})
    
    return patterns

def analyze_lag_relationships(combined_data, patterns, max_lag_seconds=600, lag_step_seconds=10):
    """
    Analyze lag relationships between BTC patterns and altcoin returns with finer granularity.
    
    Args:
        combined_data: DataFrame with combined BTC and altcoin data
        patterns: DataFrame with pattern signals
        max_lag_seconds: Maximum lag in seconds to analyze (default: 600 seconds = 10 minutes)
        lag_step_seconds: Step size in seconds between lag measurements (default: 10 seconds)
    
    Returns:
        Dictionary with pattern statistics
    """
    print(f"Analyzing lag patterns from {lag_step_seconds} to {max_lag_seconds} seconds in {lag_step_seconds}-second steps...")
    pattern_stats = {}
    
    # Check if patterns DataFrame has actual pattern columns
    pattern_cols = [col for col in patterns.columns if col != 'altcoin_name']
    if len(pattern_cols) == 0:
        print("WARNING: No patterns detected in data. Cannot perform lag analysis.")
        # Return minimal structure to prevent downstream errors
        altcoin_prefix = patterns['altcoin_name'].iloc[0] if 'altcoin_name' in patterns else 'alt'
        return {'no_patterns_detected': {
            'instances': 0,
            'optimal_lag': 0,
            'correlation': 0,
            'avg_return': 0,
            'win_rate': 0,
            'returns': {},
            'win_rates': {},
            'altcoin_name': altcoin_prefix
        }}
        
    # Get altcoin column name
    altcoin_returns_col = None
    altcoin_prefix = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            altcoin_returns_col = col
            altcoin_prefix = col.split('_')[0]
            break
    
    if not altcoin_returns_col:
        print("Warning: Could not find altcoin returns column")
        altcoin_returns_col = 'alt_returns'  # generic default fallback
        altcoin_prefix = 'alt'
    
    print(f"Using altcoin: {altcoin_prefix}")
    
    # First, check the data frequency to ensure it supports 10-second analysis
    timestamps = combined_data.index
    if len(timestamps) < 2:
        print("WARNING: Not enough data points for lag analysis")
        return pattern_stats
    
    # Calculate typical time difference between consecutive rows
    time_diffs = []
    for i in range(1, min(100, len(timestamps))):
        if isinstance(timestamps[i], pd.Timestamp) and isinstance(timestamps[i-1], pd.Timestamp):
            diff_seconds = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_diffs.append(diff_seconds)
    
    if not time_diffs:
        print("WARNING: Could not determine data frequency")
        return pattern_stats
    
    # Get median time difference as the data frequency
    data_frequency = np.median(time_diffs)
    print(f"Detected data frequency: {data_frequency} seconds")
    
    # Check if data frequency is compatible with requested lag step
    if data_frequency > lag_step_seconds:
        print(f"WARNING: Data frequency ({data_frequency} seconds) is larger than requested lag step ({lag_step_seconds} seconds)")
        print("Using data frequency as the minimum lag step")
        lag_step_seconds = max(int(data_frequency), 1)  # Ensure it's at least 1 second
    
    # Analyze each pattern
    for pattern in pattern_cols:
        # First add the pattern to combined_data if it's not already there
        if pattern not in combined_data.columns:
            print(f"Warning: Pattern '{pattern}' not found in combined_data columns, adding it")
            # Merge the pattern column from patterns to combined_data
            if pattern in patterns.columns:
                combined_data[pattern] = patterns[pattern]
            else:
                print(f"Warning: Pattern '{pattern}' not found in patterns DataFrame, skipping")
                continue
        
        # Find rows where pattern is True
        pattern_instances = combined_data[combined_data[pattern] == True]
        if len(pattern_instances) == 0:
            print(f"No instances of pattern '{pattern}' found in the data.")
            continue
            
        # Now analyze lag relationships for this pattern
        lag_correlations = {}
        lag_win_rates = {}
        lag_returns = {}
        
        # Calculate metrics for each lag in seconds
        for lag_seconds in range(lag_step_seconds, max_lag_seconds + lag_step_seconds, lag_step_seconds):
            # Convert lag from seconds to data rows
            lag_rows = max(round(lag_seconds / data_frequency), 1)
            
            # Get indices where the pattern occurs
            pattern_indices = combined_data.index[combined_data[pattern] == True].tolist()
            
            # Skip if no pattern instances
            if not pattern_indices:
                continue
                
            # Calculate returns after pattern occurs
            future_returns = []
            win_count = 0
            
            for idx in pattern_indices:
                try:
                    # Get the index position, not the timestamp
                    idx_position = combined_data.index.get_loc(idx)
                    
                    # Check if lag steps ahead is within dataframe bounds
                    if idx_position + lag_rows < len(combined_data):
                        future_idx = combined_data.index[idx_position + lag_rows]
                        future_return = combined_data.loc[future_idx, altcoin_returns_col]
                        
                        future_returns.append(future_return)
                        if future_return > 0:
                            win_count += 1
                except Exception as e:
                    pass  # Skip any index errors
            
            # Calculate lag metrics
            if future_returns:
                avg_return = sum(future_returns) / len(future_returns)
                win_rate = win_count / len(future_returns)
                
                # Calculate correlation between pattern and future returns
                pattern_series = combined_data[pattern].astype(int)
                shifted_returns = combined_data[altcoin_returns_col].shift(-lag_rows)
                
                # Only calculate correlation where we have valid data
                valid_data = ~(pattern_series.isna() | shifted_returns.isna())
                if valid_data.sum() > 10:  # Need enough data points
                    try:
                        from scipy.stats import pearsonr
                        correlation = pearsonr(
                            pattern_series[valid_data], 
                            shifted_returns[valid_data]
                        )[0]
                    except:
                        correlation = 0
                else:
                    correlation = 0
                
                lag_correlations[lag_seconds] = correlation
                lag_win_rates[lag_seconds] = win_rate
                lag_returns[lag_seconds] = avg_return
        
        # Find optimal lag based on return
        if lag_returns:
            optimal_lag_seconds = max(lag_returns.items(), key=lambda x: x[1])[0]
            
            # Store pattern statistics with both seconds and minutes
            pattern_stats[pattern] = {
                'instances': len(pattern_instances),
                'optimal_lag_seconds': optimal_lag_seconds,
                'optimal_lag': optimal_lag_seconds / 60,  # Keep the original key for compatibility
                'correlation': lag_correlations.get(optimal_lag_seconds, 0),
                'avg_return': lag_returns.get(optimal_lag_seconds, 0),
                'win_rate': lag_win_rates.get(optimal_lag_seconds, 0),
                'returns': lag_returns,
                'win_rates': lag_win_rates,
                'altcoin_name': altcoin_prefix
            }
            
            # Print some stats
            print(f"Pattern {pattern} - {len(pattern_instances)} instances, optimal lag: {optimal_lag_seconds/60:.2f} minutes ({optimal_lag_seconds} seconds)")
    
    return pattern_stats

def analyze_cross_asset_relationships(combined_data, patterns, max_lag_minutes=20, lag_step_minutes=1):
    """
    Analyze relationships between BTC patterns and altcoin patterns/returns.
    
    Args:
        combined_data: DataFrame with combined BTC and altcoin data
        patterns: DataFrame with pattern signals
        max_lag_minutes: Maximum lag in minutes to analyze (default: 20 minutes)
        lag_step_minutes: Step size in minutes between lag measurements (default: 1 minute)
    
    Returns:
        Dictionary with relationship statistics
    """
    print(f"Analyzing cross-asset pattern relationships from {lag_step_minutes} to {max_lag_minutes} minutes in {lag_step_minutes}-minute steps...")
    relationship_stats = {}
    
    # Find altcoin prefix from column names
    alt_prefix = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            alt_prefix = col.split('_')[0]
            break
    
    if not alt_prefix:
        alt_prefix = 'alt'
        
    print(f"Using altcoin: {alt_prefix}")
    
    # Get pattern columns, excluding metadata
    pattern_cols = [col for col in patterns.columns if col != 'altcoin_name']
    
    # Get BTC and altcoin pattern columns
    btc_patterns = [col for col in pattern_cols if col.startswith('btc_')]
    alt_patterns = [col for col in pattern_cols if not col.startswith('btc_') and col != 'altcoin_name']
    
    print(f"Found {len(btc_patterns)} BTC patterns and {len(alt_patterns)} altcoin patterns")
    
    # First, check the data frequency to ensure it supports minute-by-minute analysis
    timestamps = combined_data.index
    if len(timestamps) < 2:
        print("WARNING: Not enough data points for lag analysis")
        return relationship_stats
    
    # Calculate typical time difference between consecutive rows
    time_diffs = []
    for i in range(1, min(100, len(timestamps))):
        if isinstance(timestamps[i], pd.Timestamp) and isinstance(timestamps[i-1], pd.Timestamp):
            diff_seconds = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_diffs.append(diff_seconds)
    
    if not time_diffs:
        print("WARNING: Could not determine data frequency")
        return relationship_stats
    
    # Get median time difference as the data frequency
    data_frequency_seconds = np.median(time_diffs)
    data_frequency_minutes = data_frequency_seconds / 60
    print(f"Detected data frequency: {data_frequency_seconds:.1f} seconds ({data_frequency_minutes:.2f} minutes)")
    
    # Check if data frequency is compatible with requested lag step
    min_lag_step = max(lag_step_minutes, data_frequency_minutes)
    if data_frequency_minutes > lag_step_minutes:
        print(f"WARNING: Data frequency ({data_frequency_minutes:.2f} min) is larger than requested lag step ({lag_step_minutes} min)")
        print(f"Using data frequency as the minimum lag step")
        
    # Analyze relationships between BTC patterns and altcoin returns
    for btc_pattern in btc_patterns:
        # Add the pattern to combined_data if needed
        if btc_pattern not in combined_data.columns and btc_pattern in patterns.columns:
            combined_data[btc_pattern] = patterns[btc_pattern]
            
        # Skip pattern if not in combined_data
        if btc_pattern not in combined_data.columns:
            continue
            
        # Get pattern instances
        pattern_instances = combined_data[combined_data[btc_pattern] == True]
        if len(pattern_instances) == 0:
            continue
            
        print(f"Analyzing {btc_pattern}: {len(pattern_instances)} instances")
        relationship_stats[btc_pattern] = {}
        
        # Analyze impact on altcoin patterns and returns after this BTC pattern
        for lag_minutes in range(int(min_lag_step), max_lag_minutes + 1, int(min_lag_step)):
            # Calculate rows for this lag
            lag_rows = int(round(lag_minutes / data_frequency_minutes))
            
            triggered_patterns = {alt_pattern: 0 for alt_pattern in alt_patterns}
            alt_returns = []
            
            # For each BTC pattern instance, check what happens lag steps later
            for idx in pattern_instances.index:
                try:
                    # Get the index position
                    idx_position = combined_data.index.get_loc(idx)
                    
                    # Check if lag steps ahead is within dataframe bounds
                    if idx_position + lag_rows < len(combined_data):
                        future_idx = combined_data.index[idx_position + lag_rows]
                        
                        # Check which altcoin patterns occur at this future idx
                        for alt_pattern in alt_patterns:
                            if alt_pattern in combined_data.columns:
                                # Safe way to check if pattern is True at future_idx
                                pattern_val = combined_data.loc[future_idx, alt_pattern]
                                if isinstance(pattern_val, pd.Series) and not pattern_val.empty:
                                    if pattern_val.iloc[0]:
                                        triggered_patterns[alt_pattern] += 1
                                elif pattern_val:  # If scalar value
                                    triggered_patterns[alt_pattern] += 1
                        
                        # Calculate altcoin return at this lag
                        alt_return_col = f"{alt_prefix}_returns"
                        if alt_return_col in combined_data.columns:
                            alt_returns.append(combined_data.loc[future_idx, alt_return_col])
                except Exception as e:
                    pass  # Skip any index errors
            
            # Calculate statistics
            total_instances = len(pattern_instances)
            if total_instances > 0 and alt_returns:
                avg_return = sum(alt_returns) / len(alt_returns)
                win_rate = sum(1 for r in alt_returns if r > 0) / len(alt_returns)
                
                # Calculate probability of each altcoin pattern
                pattern_probs = {
                    alt_pattern: triggered_patterns[alt_pattern] / total_instances
                    for alt_pattern in triggered_patterns
                }
                
                # Find top triggered patterns
                top_patterns = sorted(pattern_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                
                relationship_stats[btc_pattern][lag_minutes] = {
                    'avg_return': avg_return,
                    'win_rate': win_rate,
                    'pattern_probabilities': pattern_probs,
                    'top_patterns': top_patterns,
                    'sample_size': total_instances
                }
                
                # Print some stats for important lags
                if lag_minutes in [1, 5, 10, 20]:
                    print(f"  Lag {lag_minutes} min: Avg Return={avg_return*100:.4f}%, Win Rate={win_rate*100:.1f}%, Sample={len(alt_returns)}")
                    if top_patterns:
                        print(f"  Top triggered patterns: {', '.join([f'{p[0]}({p[1]*100:.1f}%)' for p in top_patterns])}")
    
    return relationship_stats

def analyze_patterns_parallel(combined_data, patterns, max_lag=20):
    """Analyze patterns using parallel processing."""
    pattern_stats = {}
    
    with ProcessPoolExecutor() as executor:
        future_to_pattern = {
            executor.submit(analyze_lag_relationships, combined_data, patterns[[pattern]], max_lag): pattern
            for pattern in patterns.columns
        }
        
        for future in as_completed(future_to_pattern):
            pattern = future_to_pattern[future]
            try:
                pattern_stats[pattern] = future.result()
            except Exception as e:
                print(f"Error processing {pattern}: {e}")
    
    return pattern_stats

def optimize_strategy_parameters(data, patterns, max_lag=20, altcoin_name=None, results_dirs=None):
    """Optimize trading strategy parameters."""
    try:
        # Detect altcoin name if not provided
        if not altcoin_name:
            # Try to detect from column names
            for col in data.columns:
                if col.endswith('_returns') and not col.startswith('btc'):
                    altcoin_name = col.split('_')[0]
                    print(f"Auto-detected altcoin: {altcoin_name}")
                    break
                    
            # If still not found, use first pattern stat
            # Fix: Properly check if patterns has altcoin_name column
            if not altcoin_name and isinstance(patterns, pd.DataFrame) and 'altcoin_name' in patterns.columns:
                # Fix: Get the first value to avoid Series in boolean context
                altcoin_name = patterns['altcoin_name'].iloc[0]
                
        if not altcoin_name:
            altcoin_name = "alt"  # Generic fallback
            
        # Create strategy optimizer
        optimizer = StrategyOptimizer(data, patterns)
        optimizer.altcoin_name = altcoin_name
        
        # Run optimization with proper backtesting for 24/7 markets
        print(f"Running strategy optimization for {altcoin_name}...")
        results = optimizer.ml_optimization()
        
        # Get path to results directory
        if results_dirs is not None:
            # Use the provided results directories
            html_dir = results_dirs['html']
            charts_dir = results_dirs['charts']
            reports_dir = results_dirs['reports']
        else:
            # Fallback to inferring from data
            if hasattr(data, 'name'):
                results_dir = os.path.dirname(os.path.dirname(data.name))
            else:
                results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                
            html_dir = os.path.join(results_dir, 'html')
            charts_dir = os.path.join(results_dir, 'charts')
            reports_dir = os.path.join(results_dir, 'reports')
        
        # Make sure directories exist
        os.makedirs(html_dir, exist_ok=True)
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate reports
        html_output = os.path.join(html_dir, 'strategy_optimization_results.html')
        optimizer.generate_html_report(html_output, charts_dir)
        
        # Save strategy params to JSON file
        if optimizer.best_params and optimizer.performance_metrics:
            # Save best parameters to JSON file - CONVERT NUMPY TYPES FIRST
            strategy_params = {
                'best_params': optimizer._convert_numpy_types(optimizer.best_params),
                'performance_metrics': optimizer._convert_numpy_types(optimizer.performance_metrics),
                'summary': optimizer.get_optimal_strategy_summary()
            }
            
            with open(os.path.join(reports_dir, 'strategy_params.json'), 'w') as f:
                import json
                json.dump(strategy_params, f, indent=4)
        
        return {
            'best_params': optimizer.best_params,
            'performance_metrics': optimizer.performance_metrics,
            'strategy_summary': optimizer.get_optimal_strategy_summary() if hasattr(optimizer, 'get_optimal_strategy_summary') else None,
            'optimizer': optimizer  # Return the optimizer for further use
        }
        
    except Exception as e:
        print(f"Error in strategy optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_btc_momentum_impact(combined_data, altcoin_name=None):
    """
    Analyze how BTC momentum affects the altcoin's price.
    
    Args:
        combined_data: DataFrame with BTC and altcoin data
        altcoin_name: Name of altcoin to analyze
        
    Returns:
        dict: Analysis results
    """
    print("Analyzing BTC momentum impact on altcoin...")
    
    # Detect altcoin prefix if not provided
    if not altcoin_name:
        for col in combined_data.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                altcoin_name = col.split('_')[0]
                print(f"Auto-detected altcoin: {altcoin_name}")
                break
    
    if not altcoin_name:
        print("ERROR: Could not detect altcoin name")
        return {}
    
    # Calculate momentum indicators if not present
    if 'btc_momentum_15m' not in combined_data.columns:
        combined_data['btc_momentum_15m'] = combined_data['btc_returns'].rolling(15).mean() * 100
        combined_data['btc_momentum_30m'] = combined_data['btc_returns'].rolling(30).mean() * 100
        combined_data['btc_momentum_1h'] = combined_data['btc_returns'].rolling(60).mean() * 100
    
    # Calculate BTC volatility
    combined_data['btc_volatility_15m'] = combined_data['btc_returns'].rolling(15).std() * 100
    
    # Segment BTC by momentum strength
    strong_up = combined_data['btc_momentum_15m'] > combined_data['btc_momentum_15m'].quantile(0.75)
    strong_down = combined_data['btc_momentum_15m'] < combined_data['btc_momentum_15m'].quantile(0.25)
    neutral = (~strong_up) & (~strong_down)
    
    # Calculate altcoin returns following different BTC momentum conditions
    alt_returns_col = f"{altcoin_name.lower()}_returns"
    
    # For different lags (1, 5, 15, 30 minutes)
    results = {}
    for lag in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45]:
        # Get altcoin returns 'lag' minutes after each BTC momentum condition
        returns_after_strong_up = combined_data.loc[strong_up, alt_returns_col].shift(-lag).dropna()
        returns_after_strong_down = combined_data.loc[strong_down, alt_returns_col].shift(-lag).dropna()
        returns_after_neutral = combined_data.loc[neutral, alt_returns_col].shift(-lag).dropna()
        
        # Calculate metrics
        results[f"lag_{lag}"] = {
            'strong_up': {
                'mean_return': returns_after_strong_up.mean(),
                'win_rate': (returns_after_strong_up > 0).mean(),
                'count': len(returns_after_strong_up)
            },
            'strong_down': {
                'mean_return': returns_after_strong_down.mean(),
                'win_rate': (returns_after_strong_down > 0).mean(),
                'count': len(returns_after_strong_down)
            },
            'neutral': {
                'mean_return': returns_after_neutral.mean(),
                'win_rate': (returns_after_neutral > 0).mean(),
                'count': len(returns_after_neutral)
            }
        }
    
    # Print summary of findings
    print("\nBTC Momentum Impact Analysis:")
    for lag, lag_results in results.items():
        print(f"\nLag: {lag} minutes")
        for condition, metrics in lag_results.items():
            print(f"  BTC {condition}:")
            print(f"    Mean {altcoin_name} return: {metrics['mean_return']*100:.4f}%")
            print(f"    Win rate: {metrics['win_rate']*100:.1f}%")
            print(f"    Sample size: {metrics['count']}")
            
    return {
        'momentum_analysis': results,
        'altcoin_name': altcoin_name
    }

def run_momentum_backtests(combined_data, patterns, altcoin_name=None, results_dirs=None):
    """
    Run multiple backtests based on BTC momentum conditions.
    
    Args:
        combined_data: DataFrame with BTC and altcoin data
        patterns: DataFrame with pattern signals
        altcoin_name: Name of altcoin to analyze
        results_dirs: Dictionary of result directories
        
    Returns:
        dict: Backtest results
    """
    print("\nRunning momentum-based backtests...")
    
    # Detect altcoin if not provided
    if not altcoin_name:
        for col in combined_data.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                altcoin_name = col.split('_')[0]
                break
    
    # Create strategy optimizer
    from src.strategy_optimizer import StrategyOptimizer
    
    # Segment patterns by BTC momentum
    momentum_conditions = [
        ('strong_up', 'btc_pattern_strong_up'),
        ('strong_down', 'btc_pattern_strong_down'),
        ('volume_spike_up', 'btc_pattern_volume_spike_up'),
        ('volume_spike_down', 'btc_pattern_volume_spike_down'),
        ('volatility_breakout', 'btc_pattern_volatility_breakout')
    ]
    
    results = {}
    best_strategy = None
    best_sharpe = -float('inf')
    
    # Run a backtest for each momentum condition
    for condition_name, pattern_col in momentum_conditions:
        if pattern_col in patterns.columns:
            print(f"\nBacktesting {condition_name} momentum condition...")
            
            # Create optimizer for this condition
            optimizer = StrategyOptimizer(combined_data, patterns)
            optimizer.altcoin_name = altcoin_name
            
            # Set pattern to test
            params = {
                'entry_threshold': 0,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 3.0,
                'max_holding_time': 120,
                'pattern_lag': 1,
                'use_pattern': pattern_col,
                'position_size_pct': 100
            }
            
            # Run backtest
            metrics = optimizer.backtest_strategy(params)
            
            # Store results
            results[condition_name] = {
                'params': params,
                'metrics': metrics
            }
            
            # Track best strategy
            if metrics['sharpe_ratio'] > best_sharpe and metrics['total_trades'] > 10:
                best_sharpe = metrics['sharpe_ratio']
                best_strategy = {
                    'condition': condition_name,
                    'params': params,
                    'metrics': metrics
                }
    
    # Print summary of momentum backtests
    print("\nMomentum Backtests Summary:")
    for condition, result in results.items():
        metrics = result['metrics']
        print(f"\nCondition: {condition}")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    if best_strategy:
        print("\nBest Strategy:")
        print(f"  Condition: {best_strategy['condition']}")
        print(f"  Win Rate: {best_strategy['metrics']['win_rate']*100:.1f}%")
        print(f"  Return: {best_strategy['metrics']['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {best_strategy['metrics']['sharpe_ratio']:.2f}")
    
    return {
        'momentum_backtest_results': results,
        'best_strategy': best_strategy,
        'altcoin_name': altcoin_name
    }

def analyze_btc_directional_impact(combined_data, lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30], min_price_move=0.001):
    """
    Analyze what happens to altcoin when BTC moves up or down at various time lags.
    
    Args:
        combined_data: DataFrame with BTC and altcoin data
        lags: List of time lags in minutes to analyze
        min_price_move: Minimum BTC price move to consider significant (0.1% default)
        
    Returns:
        Dictionary with directional impact statistics
    """
    print("\nAnalyzing BTC directional impact on altcoin...")
    
    # First detect which columns we're working with
    alt_prefix = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            alt_prefix = col.split('_')[0]
            break
    
    if not alt_prefix:
        print("WARNING: Could not detect altcoin returns column")
        return {}
    
    alt_returns_col = f'{alt_prefix}_returns'
    print(f"Using BTC returns to predict {alt_returns_col}")
    
    # Define the different BTC move scenarios
    scenarios = {
        'btc_up_strong': (combined_data['btc_returns'] > min_price_move),
        'btc_up_medium': (combined_data['btc_returns'] > min_price_move/2) & (combined_data['btc_returns'] <= min_price_move),
        'btc_up_small': (combined_data['btc_returns'] > 0) & (combined_data['btc_returns'] <= min_price_move/2),
        'btc_down_small': (combined_data['btc_returns'] < 0) & (combined_data['btc_returns'] >= -min_price_move/2),
        'btc_down_medium': (combined_data['btc_returns'] < -min_price_move/2) & (combined_data['btc_returns'] >= -min_price_move),
        'btc_down_strong': (combined_data['btc_returns'] < -min_price_move)
    }
    
    # Store results
    results = {}
    
    # Calculate average altcoin returns and win rates at each lag for each scenario
    for scenario_name, condition in scenarios.items():
        results[scenario_name] = {'instances': condition.sum(), 'lags': {}}
        scenario_indices = combined_data.index[condition]
        
        # For each lag, calculate what happens to the altcoin
        for lag in lags:
            lag_returns = []
            for idx in scenario_indices:
                try:
                    # Get the index position
                    idx_pos = combined_data.index.get_loc(idx)
                    # Get the future index at lag distance if within bounds
                    if idx_pos + lag < len(combined_data):
                        future_idx = combined_data.index[idx_pos + lag]
                        # Calculate cumulative altcoin return from idx to future_idx
                        lag_return = sum(combined_data.loc[
                            combined_data.index[idx_pos:idx_pos+lag+1], 
                            alt_returns_col
                        ])
                        lag_returns.append(lag_return)
                except Exception as e:
                    pass
            
            # Calculate statistics if we have lag returns
            if lag_returns:
                avg_return = sum(lag_returns) / len(lag_returns)
                win_rate = sum(1 for r in lag_returns if r > 0) / len(lag_returns)
                
                results[scenario_name]['lags'][lag] = {
                    'avg_return': avg_return,
                    'win_rate': win_rate,
                    'sample_size': len(lag_returns)
                }
                
                # Print results for important lags
                if lag in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45]:
                    print(f"{scenario_name} → {lag}min lag: Avg Return={avg_return*100:.4f}%, "
                          f"Win Rate={win_rate*100:.1f}%, Sample={len(lag_returns)}")
    
    return results
