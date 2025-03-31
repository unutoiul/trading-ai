import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ta
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
print("Loading data...")
btc_data = pd.read_csv('exports/BTC_USDT_24-February-2025_to_26-March-2025.csv')
doge_data = pd.read_csv('exports/DOGE_USDT_24-February-2025_to_26-March-2025.csv')

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

print("Generating technical indicators...")
# Feature Engineering - Add technical indicators for both assets
for suffix in ['_btc', '_doge']:
    # RSI
    combined_data[f'rsi{suffix}'] = ta.momentum.RSIIndicator(
        close=combined_data[f'close{suffix}'], window=14
    ).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=combined_data[f'close{suffix}'])
    combined_data[f'macd{suffix}'] = macd.macd()
    combined_data[f'macd_signal{suffix}'] = macd.macd_signal()
    combined_data[f'macd_diff{suffix}'] = macd.macd_diff()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=combined_data[f'high{suffix}'],
        low=combined_data[f'low{suffix}'],
        close=combined_data[f'close{suffix}']
    )
    combined_data[f'stoch_k{suffix}'] = stoch.stoch()

# Create momentum features at different timeframes
for window in [5, 15, 30, 60]:
    combined_data[f'btc_momentum_{window}m'] = combined_data['btc_returns'].rolling(window).sum()
    combined_data[f'doge_momentum_{window}m'] = combined_data['doge_returns'].rolling(window).sum()
    combined_data[f'btc_volatility_{window}m'] = combined_data['btc_returns'].rolling(window).std()
    combined_data[f'doge_volatility_{window}m'] = combined_data['doge_returns'].rolling(window).std()

# Drop NaN values created by rolling windows
combined_data.dropna(inplace=True)

# =======================================
# PATTERN IDENTIFICATION & CLASSIFICATION
# =======================================

print("Identifying momentum patterns...")

# 1. Define pattern classification functions
def classify_momentum_patterns(df):
    """Classify different types of momentum patterns in Bitcoin"""
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
    
    return patterns

# 2. Identify the patterns in our data
btc_patterns = classify_momentum_patterns(combined_data)
combined_data = pd.concat([combined_data, btc_patterns], axis=1)

# 3. Analyze the lag relationship for each pattern
print("Analyzing lag patterns...")
pattern_stats = {}
max_lag = 20  # Maximum lag to check (minutes)

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

print("\nPattern Analysis Results:")
for pattern, stats in pattern_stats.items():
    print(f"\nPattern: {pattern}")
    print(f"  Instances: {stats['instances']}")
    print(f"  Optimal lag: {stats['optimal_lag']} minutes")
    print(f"  Correlation: {stats['correlation']:.4f}")
    print(f"  Avg return at optimal lag: {stats['avg_return']:.6f}")
    print(f"  Win rate at optimal lag: {stats['win_rate']:.2%}")

# =======================================
# ADVANCED VISUALIZATION
# =======================================

print("\nCreating visualizations...")

# 1. Lag response curves for each pattern
plt.figure(figsize=(14, 10))
for i, (pattern, stats) in enumerate(pattern_stats.items()):
    plt.subplot(3, 4, i+1)
    lags = list(stats['returns'].keys())
    returns = list(stats['returns'].values())
    plt.plot(lags, returns, marker='o', linestyle='-')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=stats['optimal_lag'], color='g', linestyle='--', alpha=0.3)
    plt.title(pattern.replace('_', ' ').title())
    plt.xlabel('Lag (minutes)')
    plt.ylabel('Avg DOGE Return')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pattern_lag_responses.png')

# 2. Create a detailed interactive visualization using Plotly
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        'BTC-DOGE Prices', 
        'BTC Momentum', 
        'Pattern Occurrences', 
        'Pattern-DOGE Response'
    ),
    specs=[
        [{"secondary_y": True}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": True}]
    ]
)

# Plot 1: BTC and DOGE prices
fig.add_trace(
    go.Scatter(x=combined_data.index, y=combined_data['close_btc'], name='BTC Price'),
    row=1, col=1, secondary_y=False
)
fig.add_trace(
    go.Scatter(x=combined_data.index, y=combined_data['close_doge'], name='DOGE Price'),
    row=1, col=1, secondary_y=True
)

# Plot 2: BTC Momentum indicators
fig.add_trace(
    go.Scatter(x=combined_data.index, y=combined_data['btc_momentum_15m'], name='BTC 15m Momentum'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=combined_data.index, y=combined_data['rsi_btc'], name='BTC RSI'),
    row=2, col=1
)

# Plot 3: Pattern occurrences (heatmap style)
for i, pattern in enumerate(combined_data.columns[combined_data.columns.str.contains('strong_up|strong_down|volatility_breakout|steady_climb|steady_decline')]):
    # Only plot the first 5 patterns to avoid clutter
    if i < 5:
        pattern_data = combined_data[pattern].astype(int) * (i+1)  # Multiply by i+1 to stack them
        fig.add_trace(
            go.Scatter(
                x=combined_data.index, 
                y=pattern_data, 
                mode='markers',
                marker=dict(size=10),
                name=pattern.replace('_', ' ').title()
            ),
            row=3, col=1
        )

# Plot 4: DOGE response after BTC patterns
# For the most significant pattern, show DOGE returns after the pattern
best_pattern = max(pattern_stats.items(), key=lambda x: abs(x[1]['correlation']))[0]
best_lag = pattern_stats[best_pattern]['optimal_lag']

# Create a time-shifted series for visualization
shifted_returns = []
pattern_times = []

for idx in combined_data[combined_data[best_pattern]].index:
    idx_pos = combined_data.index.get_loc(idx)
    if idx_pos + best_lag < len(combined_data):
        lagged_return = combined_data['doge_returns'].iloc[idx_pos + best_lag]
        shifted_returns.append(lagged_return)
        pattern_times.append(combined_data.index[idx_pos + best_lag])

if pattern_times:
    fig.add_trace(
        go.Scatter(
            x=pattern_times, 
            y=shifted_returns, 
            mode='markers',
            marker=dict(
                size=12, 
                color=shifted_returns,
                colorscale='RdYlGn',
                cmin=-0.01,
                cmax=0.01
            ),
            name=f'DOGE Return {best_lag}min After {best_pattern}'
        ),
        row=4, col=1, secondary_y=False
    )

# Add DOGE momentum for comparison
fig.add_trace(
    go.Scatter(x=combined_data.index, y=combined_data['doge_momentum_15m'], 
              name='DOGE 15m Momentum', line=dict(color='blue', width=1)),
    row=4, col=1, secondary_y=True
)

# Update layout
fig.update_layout(
    height=1200, 
    title_text="BTC-DOGE Momentum Pattern Analysis",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
)

# Save the interactive plot
fig.write_html('btc_doge_pattern_analysis.html')

# 3. Create a pattern-specific visualization for the top 3 most predictive patterns
top_patterns = sorted(pattern_stats.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)[:3]

for pattern, stats in top_patterns:
    # Create figure with secondary y-axis
    pattern_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              vertical_spacing=0.1,
                              subplot_titles=(
                                  f'{pattern.replace("_", " ").title()} Pattern', 
                                  'BTC-DOGE Price Action',
                                  f'DOGE Returns {stats["optimal_lag"]} Minutes After Pattern'
                              ))
    
    # Add pattern occurrences
    pattern_times = combined_data[combined_data[pattern]].index
    
    # Plot 1: Pattern indicators
    pattern_fig.add_trace(
        go.Scatter(x=combined_data.index, y=combined_data['rsi_btc'], name='BTC RSI'),
        row=1, col=1
    )
    pattern_fig.add_trace(
        go.Scatter(x=combined_data.index, y=combined_data['btc_momentum_15m'], name='BTC 15m Momentum'),
        row=1, col=1
    )
    
    # Highlight pattern occurrences
    for time in pattern_times:
        pattern_fig.add_vline(x=time, line_width=1, line_dash="dash", line_color="red",
                           row=1, col=1)
    
    # Plot 2: BTC-DOGE Prices
    pattern_fig.add_trace(
        go.Scatter(x=combined_data.index, y=combined_data['close_btc'], name='BTC'),
        row=2, col=1
    )
    pattern_fig.add_trace(
        go.Scatter(x=combined_data.index, y=combined_data['close_doge'], name='DOGE'),
        row=2, col=1
    )
    
    # Plot 3: DOGE returns after pattern
    lagged_returns = []
    optimal_lag = stats['optimal_lag']
    
    for idx in pattern_times:
        idx_pos = combined_data.index.get_loc(idx)
        if idx_pos + optimal_lag < len(combined_data):
            future_time = combined_data.index[idx_pos + optimal_lag]
            future_return = combined_data.loc[future_time, 'doge_returns']
            lagged_returns.append((future_time, future_return))
    
    if lagged_returns:
        x_vals, y_vals = zip(*lagged_returns)
        pattern_fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=y_vals, 
                mode='markers',
                marker=dict(
                    size=12, 
                    color=y_vals,
                    colorscale='RdYlGn',
                    cmin=-0.01,
                    cmax=0.01
                ),
                name=f'DOGE Return after {optimal_lag}min'
            ),
            row=3, col=1
        )
    
    # Update layout
    pattern_fig.update_layout(
        height=900, 
        title_text=f"Analysis of {pattern.replace('_', ' ').title()} Pattern (Corr: {stats['correlation']:.4f})"
    )
    
    # Save the pattern-specific analysis
    pattern_fig.write_html(f'{pattern}_analysis.html')

# Generate pattern summary report
print("Generating detailed pattern report...")
with open('btc_doge_pattern_report.txt', 'w') as f:
    f.write("BTC-DOGE MOMENTUM PATTERN ANALYSIS\n")
    f.write("==================================\n\n")
    
    f.write("SUMMARY OF IDENTIFIED PATTERNS\n")
    for pattern, stats in sorted(pattern_stats.items(), key=lambda x: abs(x[1]['correlation']), reverse=True):
        f.write(f"\n{pattern.replace('_', ' ').upper()}\n")
        f.write(f"  Instances: {stats['instances']}\n")
        f.write(f"  Optimal lag: {stats['optimal_lag']} minutes\n")
        f.write(f"  Correlation with DOGE returns: {stats['correlation']:.4f}\n")
        f.write(f"  Average DOGE return at optimal lag: {stats['avg_return']:.6f}\n")
        f.write(f"  Win rate at optimal lag: {stats['win_rate']:.2%}\n")
        
        # Include lag-by-lag analysis
        f.write("\n  Lag-by-lag analysis:\n")
        for lag, ret in stats['returns'].items():
            f.write(f"    Lag {lag} min: Return = {ret:.6f}, Win Rate = {stats['win_rates'].get(lag, 0):.2%}\n")
        
    f.write("\n\nPATTERN DEFINITIONS\n")
    f.write("- strong_up: BTC shows strong upward momentum with RSI > 60\n")
    f.write("- strong_down: BTC shows strong downward momentum with RSI < 40\n")
    f.write("- volatility_breakout: Sudden increase in BTC volatility with large price movement\n")
    f.write("- steady_climb: Sustained upward movement with lower volatility\n")
    f.write("- steady_decline: Sustained downward movement with lower volatility\n")
    f.write("- rsi_divergence: Price increases while RSI decreases\n")
    f.write("- macd_crossover: MACD line crosses above the signal line\n")
    f.write("- stoch_oversold_bounce: Stochastic oscillator bounces from oversold territory\n")
    f.write("- rsi_overbought: RSI exceeds 70, indicating overbought conditions\n")
    f.write("- rsi_oversold: RSI falls below 30, indicating oversold conditions\n")
    
    f.write("\n\nKEY FINDINGS\n")
    # Sort patterns by absolute correlation
    sorted_patterns = sorted(pattern_stats.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
    
    f.write(f"1. Most predictive pattern: {sorted_patterns[0][0].replace('_', ' ').upper()}\n")
    f.write(f"   - Correlation: {sorted_patterns[0][1]['correlation']:.4f}\n")
    f.write(f"   - Optimal lag: {sorted_patterns[0][1]['optimal_lag']} minutes\n")
    
    # Find pattern with highest average return
    best_return_pattern = max(pattern_stats.items(), key=lambda x: x[1]['avg_return'])
    f.write(f"\n2. Pattern with highest average return: {best_return_pattern[0].replace('_', ' ').upper()}\n")
    f.write(f"   - Average return: {best_return_pattern[1]['avg_return']:.6f}\n")
    f.write(f"   - Win rate: {best_return_pattern[1]['win_rate']:.2%}\n")
    
    # Find pattern with highest win rate
    best_winrate_pattern = max(pattern_stats.items(), key=lambda x: x[1]['win_rate'])
    f.write(f"\n3. Pattern with highest win rate: {best_winrate_pattern[0].replace('_', ' ').upper()}\n")
    f.write(f"   - Win rate: {best_winrate_pattern[1]['win_rate']:.2%}\n")
    f.write(f"   - Average return: {best_winrate_pattern[1]['avg_return']:.6f}\n")

print("Analysis complete!")
print("Generated files:")
print("- pattern_lag_responses.png: Visual comparison of lag effects for each pattern")
print("- btc_doge_pattern_analysis.html: Interactive visualization of all patterns")
print("- [pattern]_analysis.html: Detailed analysis for top 3 patterns")
print("- btc_doge_pattern_report.txt: Comprehensive text report of all findings")