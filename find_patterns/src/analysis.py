"""Analyze lag relationships between BTC patterns and DOGE returns."""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def analyze_pattern_lags(combined_data, btc_patterns, max_lag=20):
    """
    Analyze the lag relationship between each BTC pattern and DOGE returns.
    
    Args:
        combined_data: DataFrame with BTC and DOGE data
        btc_patterns: DataFrame with pattern detection results
        max_lag: Maximum lag to check (minutes)
        
    Returns:
        Dictionary with pattern statistics
    """
    print("Analyzing lag patterns...")
    pattern_stats = {}
    
    for pattern in btc_patterns.columns:
        pattern_instances = combined_data[combined_data[pattern]]
        if len(pattern_instances) == 0:
            print(f"No instances of pattern '{pattern}' found in the data.")
            continue
            
        # For each time BTC shows this pattern, check DOGE returns at different lags
        lag_returns = {}
        lag_win_rates = {}
        lag_corrs = {}
        
        for lag in range(1, max_lag + 1):
            # Get DOGE returns 'lag' minutes after each pattern occurrence
            doge_lagged_returns = []
            
            for idx in pattern_instances.index:
                try:
                    # Get DOGE return 'lag' minutes after the pattern
                    future_idx = combined_data.index.get_indexer([idx])[0] + lag
                    if future_idx < len(combined_data):
                        future_return = combined_data['doge_returns'].iloc[future_idx]
                        doge_lagged_returns.append(future_return)
                except Exception as e:
                    pass
            
            if doge_lagged_returns:
                avg_return = np.mean(doge_lagged_returns)
                win_rate = np.mean([1 if r > 0 else 0 for r in doge_lagged_returns])
                lag_returns[lag] = avg_return
                lag_win_rates[lag] = win_rate
                
                # Calculate correlation between pattern and lagged DOGE returns
                pattern_series = combined_data[pattern].astype(int)
                lagged_returns_series = combined_data['doge_returns'].shift(-lag)
                valid_indices = ~lagged_returns_series.isna()
                
                if valid_indices.any():
                    corr = pearsonr(
                        pattern_series[valid_indices], 
                        lagged_returns_series[valid_indices]
                    )[0]
                    lag_corrs[lag] = corr
        
        # Find optimal lag based on highest correlation
        if lag_corrs:
            optimal_lag = max(lag_corrs.items(), key=lambda x: abs(x[1]))
            optimal_return = lag_returns.get(optimal_lag[0], 0)
            optimal_win_rate = lag_win_rates.get(optimal_lag[0], 0)
            
            pattern_stats[pattern] = {
                'instances': len(pattern_instances),
                'optimal_lag': optimal_lag[0],
                'correlation': optimal_lag[1],
                'avg_return': optimal_return,
                'win_rate': optimal_win_rate,
                'returns': lag_returns,
                'win_rates': lag_win_rates
            }
    
    # Print summary statistics
    print("\nPattern Analysis Results:")
    for pattern, stats in pattern_stats.items():
        print(f"\nPattern: {pattern}")
        print(f"  Instances: {stats['instances']}")
        print(f"  Optimal lag: {stats['optimal_lag']} minutes")
        print(f"  Correlation: {stats['correlation']:.4f}")
        print(f"  Avg return at optimal lag: {stats['avg_return']:.6f}")
        print(f"  Win rate at optimal lag: {stats['win_rate']:.2%}")
    
    return pattern_stats