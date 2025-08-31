"""Price action analysis focusing on BTC-altcoin momentum relationships."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    """Classify different types of momentum patterns based purely on price action."""
    print("Identifying price action patterns for both assets...")
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
    
    print(f"Detected altcoin: {alt_prefix.upper()}")

    # ===== Define BTC Price Action Patterns =====
    
    # Check if we have the momentum columns
    has_momentum_15 = 'btc_momentum_15' in df.columns
    has_volatility = 'btc_volatility_15' in df.columns
    
    # Pattern 1: Strong Upward Momentum
    patterns['btc_strong_up'] = df['btc_returns'] > 0.001
    
    # More sophisticated version if we have momentum data
    if has_momentum_15:
        patterns['btc_strong_up_momentum'] = (
            (df['btc_momentum_15'] > 0) &
            (df['btc_returns'] > 0.0008)
        )
    
    # Pattern 2: Strong Downward Momentum
    patterns['btc_strong_down'] = df['btc_returns'] < -0.001
    
    # More sophisticated version
    if has_momentum_15:
        patterns['btc_strong_down_momentum'] = (
            (df['btc_momentum_15'] < 0) &
            (df['btc_returns'] < -0.0008)
        )

    # Pattern 3: BTC Volume Spike with Price Increase
    if 'btc_volume' in df.columns:
        patterns['btc_volume_spike_up'] = (
            (df['btc_volume'] > df['btc_volume'].rolling(20, min_periods=5).mean() * 1.5) &
            (df['btc_returns'] > 0)
        ).fillna(False)

        # Pattern 4: BTC Volume Spike with Price Decrease
        patterns['btc_volume_spike_down'] = (
            (df['btc_volume'] > df['btc_volume'].rolling(20, min_periods=5).mean() * 1.5) &
            (df['btc_returns'] < 0)
        ).fillna(False)

    # Pattern 5: BTC Volatility Breakout
    if has_volatility:
        patterns['btc_volatility_breakout'] = (
            (df['btc_volatility_15'] > df['btc_volatility_15'].rolling(100, min_periods=20).mean() * 1.5) &
            (abs(df['btc_returns']) > abs(df['btc_returns']).rolling(30, min_periods=5).mean() * 2)
        ).fillna(False)

    # Pattern: Medium movements
    patterns['btc_medium_up'] = (df['btc_returns'] > 0.0005) & (df['btc_returns'] <= 0.001)
    patterns['btc_medium_down'] = (df['btc_returns'] < -0.0005) & (df['btc_returns'] >= -0.001)
    
    # Pattern: Small movements
    patterns['btc_small_up'] = (df['btc_returns'] > 0) & (df['btc_returns'] <= 0.0005)
    patterns['btc_small_down'] = (df['btc_returns'] < 0) & (df['btc_returns'] >= -0.0005)

    # Pattern: Altcoin follows BTC up
    patterns[f'btc_pattern_{alt_prefix}_follows_btc_up'] = (
        (df['btc_returns'].shift(1) > 0) & 
        (df[f'{alt_prefix}_returns'] > 0)
    ).fillna(False)

    # Pattern: Altcoin follows BTC down
    patterns[f'btc_pattern_{alt_prefix}_follows_btc_down'] = (
        (df['btc_returns'].shift(1) < 0) & 
        (df[f'{alt_prefix}_returns'] < 0)
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
            (df[f'{alt_prefix}_returns'] > 0)
        ).fillna(False)

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

def analyze_btc_directional_impact(combined_data, lags=[1, 2, 3, 4, 5, 10, 15, 30], min_price_move=0.001):
    """Analyze what happens to altcoin when BTC moves up or down at various time lags."""
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
    print(f"Using BTC price action to predict {alt_returns_col}")
    
    # Define the different BTC move scenarios based on price action
    scenarios = {
        'btc_strong_up': (combined_data['btc_returns'] > min_price_move),
        'btc_medium_up': (combined_data['btc_returns'] > min_price_move/2) & (combined_data['btc_returns'] <= min_price_move),
        'btc_small_up': (combined_data['btc_returns'] > 0) & (combined_data['btc_returns'] <= min_price_move/2),
        'btc_small_down': (combined_data['btc_returns'] < 0) & (combined_data['btc_returns'] >= -min_price_move/2),
        'btc_medium_down': (combined_data['btc_returns'] < -min_price_move/2) & (combined_data['btc_returns'] >= -min_price_move),
        'btc_strong_down': (combined_data['btc_returns'] < -min_price_move)
    }
    
    # Add all scenario columns to the dataframe
    for name, condition in scenarios.items():
        combined_data[name] = condition
    
    # Store results
    results = {}
    
    # Calculate average altcoin returns and win rates at each lag for each scenario
    for scenario_name, condition in scenarios.items():
        results[scenario_name] = {'instances': condition.sum(), 'lags': {}}
        
        scenario_indices = combined_data.index[condition]
        
        # For each lag, calculate what happens to the altcoin
        for lag in lags:
            # Calculate forward returns at this lag
            if not scenario_indices.empty:
                # Get future altcoin returns at this lag
                future_returns = combined_data.loc[scenario_indices, alt_returns_col].shift(-lag)
                
                # Calculate average return and win rate
                valid_returns = future_returns.dropna()
                if len(valid_returns) > 0:
                    avg_return = valid_returns.mean() * 100  # as percent
                    win_rate = (valid_returns > 0).mean()
                    
                    results[scenario_name]['lags'][lag] = {
                        'mean_return': avg_return,
                        'win_rate': win_rate,
                        'count': len(valid_returns)
                    }
    
    # Print summary of findings
    print("\nBTC Price Movement Impact Summary:")
    for scenario, scenario_data in results.items():
        instances = scenario_data['instances']
        print(f"\n{scenario.upper()} ({instances} instances):")
        
        # Find best lag for this scenario
        best_lag = None
        best_return = -float('inf')
        
        for lag, lag_data in scenario_data.get('lags', {}).items():
            mean_return = lag_data.get('mean_return', 0)
            win_rate = lag_data.get('win_rate', 0) * 100
            count = lag_data.get('count', 0)
            
            print(f"  Lag {lag}: Mean Return = {mean_return:.4f}%, Win Rate = {win_rate:.1f}%, Count = {count}")
            
            if mean_return > best_return and count >= 5:
                best_return = mean_return
                best_lag = lag
        
        if best_lag:
            print(f"  Best lag: {best_lag} with {best_return:.4f}% return")
    
    return results

def analyze_cross_asset_relationships(combined_data, patterns, max_lag_minutes=20, lag_step_minutes=1):
    """
    Analyze how BTC price movements affect altcoin prices over different time lags.
    Focuses only on price action relationships.
    
    Args:
        combined_data: DataFrame with BTC and altcoin data
        patterns: Dictionary of patterns (can be empty, only used for compatibility)
        max_lag_minutes: Maximum lag to test in minutes
        lag_step_minutes: Step size for lag testing in minutes
        
    Returns:
        Dictionary with cross-asset relationship data
    """
    print("\nAnalyzing cross-asset price action relationships...")
    
    # Determine which columns to use
    btc_returns_col = 'btc_returns'
    
    # Find altcoin column 
    alt_prefix = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            alt_prefix = col.split('_')[0]
            break
    
    if not alt_prefix:
        print("Could not identify altcoin return column")
        return {}
        
    alt_returns_col = f'{alt_prefix}_returns'
    print(f"Analyzing relationship between {btc_returns_col} and {alt_returns_col}")
    
    # Define BTC price movement thresholds
    small_threshold = 0.0005  # 0.05%
    medium_threshold = 0.001  # 0.1%
    large_threshold = 0.002   # 0.2%
    
    # Define the price action scenarios
    scenarios = {
        'btc_strong_up': combined_data[btc_returns_col] > large_threshold,
        'btc_medium_up': (combined_data[btc_returns_col] > medium_threshold) & (combined_data[btc_returns_col] <= large_threshold),
        'btc_small_up': (combined_data[btc_returns_col] > small_threshold) & (combined_data[btc_returns_col] <= medium_threshold),
        'btc_flat': (combined_data[btc_returns_col] >= -small_threshold) & (combined_data[btc_returns_col] <= small_threshold),
        'btc_small_down': (combined_data[btc_returns_col] < -small_threshold) & (combined_data[btc_returns_col] >= -medium_threshold),
        'btc_medium_down': (combined_data[btc_returns_col] < -medium_threshold) & (combined_data[btc_returns_col] >= -large_threshold),
        'btc_strong_down': combined_data[btc_returns_col] < -large_threshold
    }
    
    # Calculate data frequency in minutes
    if isinstance(combined_data.index, pd.DatetimeIndex) and len(combined_data) > 1:
        time_diff = (combined_data.index[1] - combined_data.index[0]).total_seconds() / 60
        data_frequency_minutes = max(1, int(round(time_diff)))
    else:
        data_frequency_minutes = 1  # Default to 1 minute
        
    print(f"Detected data frequency: {data_frequency_minutes} minutes")
    
    # Minimum lag step should be at least the data frequency
    min_lag_step = max(lag_step_minutes, data_frequency_minutes)
    
    # Results structure
    results = {}
    
    # For each scenario, calculate effect on altcoin at different lags
    for scenario_name, condition in scenarios.items():
        # Get all timestamps where this condition is met
        scenario_instances = combined_data.index[condition]
        instance_count = len(scenario_instances)
        
        if instance_count < 5:  # Skip scenarios with too few instances
            print(f"  Skipping {scenario_name}: only {instance_count} instances")
            continue
            
        print(f"  Analyzing {scenario_name}: {instance_count} instances")
        
        # Store scenario results
        scenario_results = {
            'name': scenario_name,
            'instances': instance_count,
            'lags': {}
        }
        
        # Test different lag periods
        for lag_minutes in range(int(min_lag_step), max_lag_minutes + 1, int(min_lag_step)):
            lag_rows = int(round(lag_minutes / data_frequency_minutes))
            
            alt_returns = []
            
            # For each instance, check what happens lag steps later
            for idx in scenario_instances:
                # Find the position of this timestamp in the DataFrame
                try:
                    position = combined_data.index.get_loc(idx)
                    
                    # Calculate future position
                    future_pos = position + lag_rows
                    
                    # Check if future position is valid
                    if future_pos < len(combined_data):
                        # Get altcoin return from current to future point
                        if alt_prefix in combined_data.columns:
                            current_price = combined_data[f'{alt_prefix}_close'].iloc[position]
                            future_price = combined_data[f'{alt_prefix}_close'].iloc[future_pos]
                        else:
                            # Try alternate column naming
                            current_price = combined_data[f'close_{alt_prefix}'].iloc[position]
                            future_price = combined_data[f'close_{alt_prefix}'].iloc[future_pos]
                            
                        if current_price > 0:  # Avoid division by zero
                            return_pct = (future_price / current_price - 1) * 100
                            alt_returns.append(return_pct)
                except:
                    continue
            
            # Calculate statistics
            if alt_returns:
                avg_return = sum(alt_returns) / len(alt_returns)
                win_rate = sum(1 for r in alt_returns if r > 0) / len(alt_returns)
                
                scenario_results['lags'][lag_minutes] = {
                    'avg_return': avg_return,
                    'win_rate': win_rate,
                    'sample_size': len(alt_returns)
                }
        
        # Find optimal lag with highest return * win_rate
        best_score = -float('inf')
        optimal_lag = None
        
        for lag, stats in scenario_results['lags'].items():
            score = stats['avg_return'] * stats['win_rate']
            if score > best_score and stats['sample_size'] >= 5:
                best_score = score
                optimal_lag = lag
                
        if optimal_lag:
            scenario_results['optimal_lag'] = optimal_lag
            scenario_results['optimal_return'] = scenario_results['lags'][optimal_lag]['avg_return']
            scenario_results['optimal_win_rate'] = scenario_results['lags'][optimal_lag]['win_rate']
            
            print(f"    Optimal lag: {optimal_lag} min, Return: {scenario_results['optimal_return']:.4f}%, "
                  f"Win rate: {scenario_results['optimal_win_rate']*100:.1f}%")
        
        # Store this scenario's results
        results[scenario_name] = scenario_results
    
    print(f"Cross-asset analysis complete. Analyzed {len(results)} scenarios.")
    return results