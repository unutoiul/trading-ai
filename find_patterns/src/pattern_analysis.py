"""Pattern detection and analysis for BTC-DOGE momentum relationships."""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor, as_completed

def classify_momentum_patterns(df):
    """
    Classify different types of momentum patterns in Bitcoin.
    
    Args:
        df: DataFrame with indicators and momentum metrics
        
    Returns:
        DataFrame with binary pattern columns
    """
    print("Identifying momentum patterns...")
    patterns = pd.DataFrame(index=df.index)
    
    # Pattern 1: Strong Upward Momentum
    patterns['strong_up'] = (
        (df['btc_momentum_15m'] > df['btc_momentum_15m'].rolling(100).mean() + df['btc_momentum_15m'].rolling(100).std()) &
        (df['btc_returns'] > 0) &
        (df['rsi_btc'] > 60)
    )
    
    # Pattern 2: Strong Downward Momentum
    patterns['strong_down'] = (
        (df['btc_momentum_15m'] < df['btc_momentum_15m'].rolling(100).mean() - df['btc_momentum_15m'].rolling(100).std()) &
        (df['btc_returns'] < 0) &
        (df['rsi_btc'] < 40)
    )
    
    # Pattern 3: Volatility Breakout
    patterns['volatility_breakout'] = (
        (df['btc_volatility_15m'] > df['btc_volatility_15m'].rolling(100).mean() + df['btc_volatility_15m'].rolling(100).std() * 1.5) &
        (abs(df['btc_returns']) > abs(df['btc_returns']).rolling(30).mean() * 2)
    )
    
    # Pattern 4: Steady Climb (Low Volatility Uptrend)
    patterns['steady_climb'] = (
        (df['btc_momentum_30m'] > 0) &
        (df['btc_volatility_30m'] < df['btc_volatility_30m'].rolling(100).mean()) &
        (df['btc_returns'] > 0) &
        (df['rsi_btc'] > 50) & (df['rsi_btc'] < 70)
    )
    
    # Pattern 5: Steady Decline (Low Volatility Downtrend)
    patterns['steady_decline'] = (
        (df['btc_momentum_30m'] < 0) &
        (df['btc_volatility_30m'] < df['btc_volatility_30m'].rolling(100).mean()) &
        (df['btc_returns'] < 0) &
        (df['rsi_btc'] < 50) & (df['rsi_btc'] > 30)
    )
    
    # Pattern 6: RSI Divergence (Price up, RSI down)
    patterns['rsi_divergence'] = (
        (df['btc_returns'] > 0) &
        (df['rsi_btc'] < df['rsi_btc'].shift(1)) &
        (df['btc_momentum_15m'] > 0)
    )
    
    # Pattern 7: MACD Crossover
    patterns['macd_crossover'] = (
        (df['macd_btc'] > df['macd_signal_btc']) &
        (df['macd_btc'].shift(1) <= df['macd_signal_btc'].shift(1))
    )
    
    # Pattern 8: Stochastic Oversold Bounce
    patterns['stoch_oversold_bounce'] = (
        (df['stoch_k_btc'] < 20) &
        (df['stoch_k_btc'] > df['stoch_k_btc'].shift(1)) &
        (df['btc_returns'] > 0)
    )
    
    # Pattern 9: RSI Overbought
    patterns['rsi_overbought'] = (df['rsi_btc'] > 70)
    
    # Pattern 10: RSI Oversold
    patterns['rsi_oversold'] = (df['rsi_btc'] < 30)
    
    # Print pattern instance counts
    for col in patterns.columns:
        count = patterns[col].sum()
        print(f"Pattern '{col}': {count} instances")
        
    return patterns

def analyze_lag_relationships(combined_data, patterns, max_lag=20):
    """
    Analyze lag relationships between BTC patterns and DOGE returns.
    
    Args:
        combined_data: DataFrame with both BTC and DOGE data
        patterns: DataFrame with pattern columns
        max_lag: Maximum lag to analyze in minutes
        
    Returns:
        Dictionary of pattern statistics
    """
    print(f"Analyzing lag patterns up to {max_lag} minutes...")
    pattern_stats = {}
    
    for pattern in patterns.columns:
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
    
    # Print summary of findings
    print("\nPattern Analysis Results:")
    for pattern, stats in pattern_stats.items():
        print(f"\nPattern: {pattern}")
        print(f"  Instances: {stats['instances']}")
        print(f"  Optimal lag: {stats['optimal_lag']} minutes")
        print(f"  Correlation: {stats['correlation']:.4f}")
        print(f"  Avg return at optimal lag: {stats['avg_return']:.6f}")
        print(f"  Win rate at optimal lag: {stats['win_rate']:.2%}")
            
    return pattern_stats

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

def optimize_strategy_parameters(combined_data, patterns, max_lag=20):
    """
    Optimize trading strategy parameters using ML.
    
    Args:
        combined_data: DataFrame with BTC and altcoin data
        patterns: DataFrame with pattern columns
        max_lag: Maximum lag to analyze
        
    Returns:
        dict: Optimized strategy parameters and performance
    """
    from src.strategy_optimizer import StrategyOptimizer
    import numpy as np
    
    print("Initializing strategy optimizer...")
    
    # Get pattern stats first
    pattern_stats = analyze_lag_relationships(combined_data, patterns, max_lag)
    
    # Initialize optimizer with the pattern statistics
    optimizer = StrategyOptimizer(combined_data, pattern_stats)
    
    # Run ML optimization
    print("\nRunning machine learning optimization...")
    optimization_results = optimizer.ml_optimization()
    
    # Generate visualizations
    print("\nGenerating performance visualizations...")
    optimizer.visualize_results()
    
    # Get strategy summary
    strategy_summary = optimizer.get_optimal_strategy_summary()
    print("\n" + strategy_summary)
    
    # Ensure we have proper metric values for serialization
    performance_metrics = {}
    if optimization_results.get('best_metrics'):
        for k, v in optimization_results['best_metrics'].items():
            if isinstance(v, (int, float, np.number)):
                performance_metrics[k] = float(v) 
            else:
                performance_metrics[k] = v
    
    return {
        'best_params': optimization_results.get('best_params', {}),
        'performance_metrics': performance_metrics,
        'strategy_summary': strategy_summary,
        'optimizer': optimizer  # Important: Return the optimizer object
    }