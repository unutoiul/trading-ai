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
    
    # Merge patterns into combined_data, avoiding duplicates
    for pattern in pattern_cols:
        if pattern not in combined_data.columns:
            if pattern in patterns.columns:
                combined_data[pattern] = patterns[pattern]
            else:
                print(f"Warning: Pattern '{pattern}' not found in patterns DataFrame, skipping")
                continue
        else:
            # Pattern already exists in combined_data, verify it matches
            if pattern in patterns.columns:
                # Use patterns version if it exists (it might be more up-to-date)
                combined_data[pattern] = patterns[pattern]
    
    # Analyze each pattern
    for pattern in pattern_cols:
        
        # Find rows where pattern is True - fix the indexing issue
        if pattern in patterns.columns:
            pattern_indices = patterns[patterns[pattern] == True].index
            pattern_instances = combined_data.loc[pattern_indices]
        else:
            print(f"Pattern '{pattern}' not found in patterns DataFrame.")
            continue
            
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
            
            # Get indices where the pattern occurs - fix the indexing issue
            if pattern in patterns.columns:
                pattern_indices = patterns.index[patterns[pattern] == True].tolist()
            else:
                continue
            
            # Skip if no pattern instances
            if not pattern_indices:
                continue
                
            # Calculate returns after pattern occurs
            future_returns = []
            positive_returns = []
            negative_returns = []
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
                            positive_returns.append(future_return)
                        else:
                            negative_returns.append(future_return)
                except Exception as e:
                    pass  # Skip any index errors
            
            # Calculate lag metrics
            if future_returns:
                avg_return = sum(future_returns) / len(future_returns)
                win_rate = win_count / len(future_returns)
                positive_count = len(positive_returns)
                negative_count = len(negative_returns)
                avg_positive_return = sum(positive_returns) / len(positive_returns) if positive_returns else 0
                avg_negative_return = sum(negative_returns) / len(negative_returns) if negative_returns else 0
                
                # Calculate correlation between pattern and future returns
                if pattern in patterns.columns:
                    pattern_series = patterns[pattern].astype(int)
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
                else:
                    correlation = 0
                
                lag_correlations[lag_seconds] = correlation
                lag_win_rates[lag_seconds] = win_rate
                lag_returns[lag_seconds] = avg_return
                
                # Store detailed metrics for display
                if not hasattr(lag_returns, 'detailed_metrics'):
                    lag_returns.detailed_metrics = {}
                lag_returns.detailed_metrics[lag_seconds] = {
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'avg_positive_return': avg_positive_return,
                    'avg_negative_return': avg_negative_return,
                    'total_count': len(future_returns)
                }
        
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
                'altcoin_name': altcoin_prefix,
                'detailed_metrics': getattr(lag_returns, 'detailed_metrics', {})
            }
            
            # Print some stats
            print(f"Pattern {pattern} - {len(pattern_instances)} instances, optimal lag: {optimal_lag_seconds/60:.2f} minutes ({optimal_lag_seconds} seconds)")
    
    return pattern_stats

def analyze_synchronous_movements(combined_data, min_price_move=0.001):
    """
    Analyze synchronous BTC-altcoin movements without lag.
    Check if altcoin moves in same direction as BTC at the same timestamp.
    
    Args:
        combined_data: DataFrame with BTC and altcoin data
        min_price_move: Minimum price move threshold to consider as a signal
        
    Returns:
        Dictionary with synchronous movement analysis results
    """
    print("Analyzing synchronous BTC-altcoin movements...")
    
    # Find altcoin prefix from column names
    alt_prefix = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            alt_prefix = col.split('_')[0]
            break
    
    if not alt_prefix:
        print("Warning: Could not detect altcoin prefix")
        return {}
    
    print(f"Analyzing BTC and {alt_prefix.upper()} synchronous movements")
    
    # Define BTC movement categories based on returns
    scenarios = {
        'btc_strong_up': combined_data['btc_returns'] > 0.0015,
        'btc_medium_up': (combined_data['btc_returns'] > 0.00075) & (combined_data['btc_returns'] <= 0.0015),
        'btc_small_up': (combined_data['btc_returns'] > 0) & (combined_data['btc_returns'] <= 0.00075),
        'btc_small_down': (combined_data['btc_returns'] < 0) & (combined_data['btc_returns'] >= -0.00075),
        'btc_medium_down': (combined_data['btc_returns'] < -0.00075) & (combined_data['btc_returns'] >= -0.0015),
        'btc_strong_down': combined_data['btc_returns'] < -0.0015
    }
    
    results = {}
    alt_returns_col = f'{alt_prefix}_returns'
    
    if alt_returns_col not in combined_data.columns:
        print(f"Warning: {alt_returns_col} not found in data")
        return {}
    
    for scenario_name, btc_condition in scenarios.items():
        print(f"\nAnalyzing {scenario_name.replace('btc_', '').replace('_', ' ').title()}...")
        
        # Get BTC signals
        btc_signals = btc_condition
        signal_count = btc_signals.sum()
        
        if signal_count < 5:
            print(f"  Insufficient signals: {signal_count} (need at least 5)")
            results[scenario_name] = {
                'total_signals': signal_count,
                'synchronous_same_direction': 0,
                'synchronous_opposite_direction': 0,
                'mean_alt_return_when_same': 0,
                'mean_alt_return_when_opposite': 0,
                'win_rate_same_direction': 0,
                'win_rate_opposite_direction': 0,
                'same_direction_percentage': 0
            }
            continue
        
        # Get altcoin returns when BTC signals occur (same timestamp)
        alt_returns_at_signal = combined_data.loc[btc_signals, alt_returns_col]
        
        # Determine if altcoin moved in same direction as BTC
        btc_direction = 'up' if 'up' in scenario_name else 'down'
        
        if btc_direction == 'up':
            same_direction = alt_returns_at_signal > 0
            opposite_direction = alt_returns_at_signal <= 0
        else:
            same_direction = alt_returns_at_signal < 0
            opposite_direction = alt_returns_at_signal >= 0
        
        # Calculate statistics
        same_direction_count = same_direction.sum()
        opposite_direction_count = opposite_direction.sum()
        same_direction_percentage = (same_direction_count / signal_count) * 100
        
        # Calculate mean returns
        mean_alt_return_same = alt_returns_at_signal[same_direction].mean() if same_direction_count > 0 else 0
        mean_alt_return_opposite = alt_returns_at_signal[opposite_direction].mean() if opposite_direction_count > 0 else 0
        
        # Calculate win rates (positive returns for altcoin)
        win_rate_same = (alt_returns_at_signal[same_direction] > 0).mean() if same_direction_count > 0 else 0
        win_rate_opposite = (alt_returns_at_signal[opposite_direction] > 0).mean() if opposite_direction_count > 0 else 0
        
        results[scenario_name] = {
            'total_signals': signal_count,
            'synchronous_same_direction': same_direction_count,
            'synchronous_opposite_direction': opposite_direction_count,
            'mean_alt_return_when_same': mean_alt_return_same,
            'mean_alt_return_when_opposite': mean_alt_return_opposite,
            'win_rate_same_direction': win_rate_same,
            'win_rate_opposite_direction': win_rate_opposite,
            'same_direction_percentage': same_direction_percentage
        }
        
        # Enhanced logging
        print(f"    Total BTC signals: {signal_count:>4}")
        print(f"    Same direction:    {same_direction_count:>4} ({same_direction_percentage:>5.1f}%)")
        print(f"    Opposite direction: {opposite_direction_count:>4} ({100-same_direction_percentage:>5.1f}%)")
        print(f"    Mean return (same):     {mean_alt_return_same*100:>7.4f}%")
        print(f"    Mean return (opposite): {mean_alt_return_opposite*100:>7.4f}%")
        print(f"    Win rate (same):        {win_rate_same*100:>7.1f}%")
        print(f"    Win rate (opposite):    {win_rate_opposite*100:>7.1f}%")
    
    return results


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
                    
                    # Calculate positive and negative breakdowns
                    positive_returns = valid_returns[valid_returns > 0]
                    negative_returns = valid_returns[valid_returns <= 0]
                    
                    positive_count = len(positive_returns)
                    negative_count = len(negative_returns)
                    avg_positive_return = positive_returns.mean() if len(positive_returns) > 0 else 0
                    avg_negative_return = negative_returns.mean() if len(negative_returns) > 0 else 0
                    
                    results[scenario_name]['lags'][lag] = {
                        'mean_return': avg_return,
                        'win_rate': win_rate,
                        'count': len(valid_returns),
                        'positive_count': positive_count,
                        'negative_count': negative_count,
                        'avg_positive_return': avg_positive_return,
                        'avg_negative_return': avg_negative_return
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
            positive_count = lag_data.get('positive_count', 0)
            negative_count = lag_data.get('negative_count', 0)
            avg_positive_return = lag_data.get('avg_positive_return', 0) * 100
            avg_negative_return = lag_data.get('avg_negative_return', 0) * 100
            
            # Enhanced logging with altcoin response breakdown
            if count > 0:
                print(f"    Lag {lag:>2}min: {mean_return:>7.4f}% altcoin response | Pos Rate: {win_rate:>5.1f}% | "
                      f"Signals: {count:>3} | Pos: {positive_count:>2}({avg_positive_return:>6.2f}%) | "
                      f"Neg: {negative_count:>2}({avg_negative_return:>6.2f}%)")
            else:
                print(f"    Lag {lag:>2}min: No valid BTC signals")
            
            if mean_return > best_return and count >= 5:
                best_return = mean_return
                best_lag = lag
        
        if best_lag:
            best_data = scenario_data['lags'][best_lag]
            print(f"  🏆 Best correlation lag: {best_lag}min with {best_return:.4f}% altcoin response")
            print(f"     Positive rate: {best_data.get('win_rate', 0)*100:.1f}%, Total BTC signals: {best_data.get('count', 0)}")
        else:
            print(f"  ❌ No profitable lags found with sufficient trades (>=5)")
    
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
    
    # Define multi-timeframe BTC price movement thresholds
    # Base thresholds for 1-minute data
    base_thresholds = {
        'small': 0.0005,   # 0.05%
        'medium': 0.001,   # 0.1%
        'large': 0.002     # 0.2%
    }
    
    # Timeframe-specific threshold adjustments (1-10 minutes)
    timeframe_multipliers = {
        1: 1.0,      # Base timeframe
        2: 1.4,      # √2 scaling for 2-minute moves
        3: 1.7,      # √3 scaling for 3-minute moves
        4: 2.0,      # √4 scaling for 4-minute moves
        5: 2.2,      # √5 scaling for 5-minute moves
        6: 2.4,      # √6 scaling for 6-minute moves
        7: 2.6,      # √7 scaling for 7-minute moves
        8: 2.8,      # √8 scaling for 8-minute moves
        9: 3.0,      # √9 scaling for 9-minute moves
        10: 3.2      # √10 scaling for 10-minute moves
    }
    
    # Create multi-timeframe scenarios
    scenarios = {}
    
    # Add 1-minute base scenarios
    small_threshold = base_thresholds['small']
    medium_threshold = base_thresholds['medium'] 
    large_threshold = base_thresholds['large']
    
    scenarios.update({
        'btc_strong_up': combined_data[btc_returns_col] > large_threshold,
        'btc_medium_up': (combined_data[btc_returns_col] > medium_threshold) & (combined_data[btc_returns_col] <= large_threshold),
        'btc_small_up': (combined_data[btc_returns_col] > small_threshold) & (combined_data[btc_returns_col] <= medium_threshold),
        'btc_flat': (combined_data[btc_returns_col] >= -small_threshold) & (combined_data[btc_returns_col] <= small_threshold),
        'btc_small_down': (combined_data[btc_returns_col] < -small_threshold) & (combined_data[btc_returns_col] >= -medium_threshold),
        'btc_medium_down': (combined_data[btc_returns_col] < -medium_threshold) & (combined_data[btc_returns_col] >= -large_threshold),
        'btc_strong_down': combined_data[btc_returns_col] < -large_threshold
    })
    
    # Add multi-timeframe scenarios (2-10 minutes)
    for tf in range(2, 11):
        tf_col = f'btc_return_{tf}m'
        
        if tf_col in combined_data.columns:
            multiplier = timeframe_multipliers[tf]
            tf_small = base_thresholds['small'] * multiplier
            tf_medium = base_thresholds['medium'] * multiplier  
            tf_large = base_thresholds['large'] * multiplier
            
            scenarios.update({
                f'btc_strong_up_{tf}m': combined_data[tf_col] > tf_large,
                f'btc_medium_up_{tf}m': (combined_data[tf_col] > tf_medium) & (combined_data[tf_col] <= tf_large),
                f'btc_small_up_{tf}m': (combined_data[tf_col] > tf_small) & (combined_data[tf_col] <= tf_medium),
                f'btc_flat_{tf}m': (combined_data[tf_col] >= -tf_small) & (combined_data[tf_col] <= tf_small),
                f'btc_small_down_{tf}m': (combined_data[tf_col] < -tf_small) & (combined_data[tf_col] >= -tf_medium),
                f'btc_medium_down_{tf}m': (combined_data[tf_col] < -tf_medium) & (combined_data[tf_col] >= -tf_large),
                f'btc_strong_down_{tf}m': combined_data[tf_col] < -tf_large
            })
            
            print(f"Added {tf}-minute timeframe thresholds: Small={tf_small:.5f}, Medium={tf_medium:.5f}, Large={tf_large:.5f}")
    
    print(f"Total BTC scenarios created: {len([k for k in scenarios.keys() if k.startswith('btc')])}")
    
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