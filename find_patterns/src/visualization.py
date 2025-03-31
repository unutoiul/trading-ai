"""Visualization functions for BTC-DOGE pattern analysis."""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that's thread-safe

import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

def create_results_directory(base_dir):
    """
    Create a timestamped directory structure for analysis results.
    
    Args:
        base_dir: Base directory for results
        
    Returns:
        Dictionary of directory paths
    """
    # Create main results directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create timestamped subfolder with day.month.year-hh.mm.ss format
    timestamp = datetime.now().strftime('%d.%m.%Y-%H.%M.%S')
    result_dir = os.path.join(base_dir, timestamp)
    
    # Create subdirectories
    charts_dir = os.path.join(result_dir, 'charts')
    html_dir = os.path.join(result_dir, 'html')
    reports_dir = os.path.join(result_dir, 'reports')
    
    for directory in [charts_dir, html_dir, reports_dir]:
        os.makedirs(directory)
    
    return {
        'base': result_dir,
        'charts': charts_dir,
        'html': html_dir,
        'reports': reports_dir
    }

def plot_lag_responses(pattern_stats, output_dir):
    """
    Create lag response plots for each pattern.
    
    Args:
        pattern_stats: Dictionary of pattern statistics
        output_dir: Directory to save the plots
    """
    print("Creating lag response plots...")
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
    output_file = os.path.join(output_dir, 'pattern_lag_responses.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Saved lag response plots to {output_file}")

def create_interactive_overview(combined_data, pattern_stats, output_dir):
    """
    Create a comprehensive interactive visualization of patterns and responses.
    
    Args:
        combined_data: DataFrame with combined BTC and DOGE data
        pattern_stats: Dictionary of pattern statistics
        output_dir: Directory to save the HTML file
    """
    print("Creating interactive overview visualization...")
    
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
    pattern_cols = [col for col in combined_data.columns if col in pattern_stats]
    for i, pattern in enumerate(pattern_cols[:5]):  # Only plot first 5 to avoid clutter
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
    output_file = os.path.join(output_dir, 'btc_doge_pattern_analysis.html')
    fig.write_html(output_file)
    print(f"Saved interactive overview to {output_file}")

def create_pattern_specific_plots(combined_data, pattern_stats, output_dir):
    """
    Create detailed visualizations for the top 3 most predictive patterns.
    
    Args:
        combined_data: DataFrame with combined BTC and DOGE data
        pattern_stats: Dictionary of pattern statistics
        output_dir: Directory to save the HTML files
    """
    print("Creating pattern-specific visualizations...")
    
    # Sort patterns by correlation strength
    top_patterns = sorted(pattern_stats.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)[:3]
    
    for pattern, stats in top_patterns:
        print(f"  Processing pattern: {pattern}...")
        
        # Create figure with secondary y-axis
        pattern_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                  vertical_spacing=0.1,
                                  subplot_titles=(
                                      f'{pattern.replace("_", " ").title()} Pattern', 
                                      'BTC-DOGE Price Action',
                                      f'DOGE Returns {stats["optimal_lag"]} Minutes After Pattern'
                                  ))
        
        # Add pattern occurrences (limit to 100 most recent to avoid performance issues)
        pattern_occurrences = combined_data[combined_data[pattern] == True]
        pattern_times = pattern_occurrences.index[-100:] if len(pattern_occurrences) > 100 else pattern_occurrences.index
        
        print(f"  Found {len(pattern_times)} pattern occurrences to visualize")
        
        # Plot 1: Pattern indicators
        pattern_fig.add_trace(
            go.Scatter(x=combined_data.index, y=combined_data['rsi_btc'], name='BTC RSI'),
            row=1, col=1
        )
        pattern_fig.add_trace(
            go.Scatter(x=combined_data.index, y=combined_data['btc_momentum_15m'], name='BTC 15m Momentum'),
            row=1, col=1
        )
        
        # Highlight pattern occurrences (max 50 to prevent slowdown)
        for time in pattern_times[-50:]:
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
        
        # Plot 3: DOGE returns after pattern - use vectorized operations where possible
        print(f"  Computing lagged returns for optimal lag: {stats['optimal_lag']}")
        lagged_returns = []
        optimal_lag = stats['optimal_lag']
        
        # More efficient approach using a counter
        count = 0
        max_patterns = 100  # Limit to prevent performance issues
        
        for idx in pattern_times:
            try:
                idx_pos = combined_data.index.get_loc(idx)
                if idx_pos + optimal_lag < len(combined_data):
                    future_time = combined_data.index[idx_pos + optimal_lag]
                    future_return = combined_data.loc[future_time, 'doge_returns']
                    lagged_returns.append((future_time, future_return))
                
                # Increment counter and check limit
                count += 1
                if count >= max_patterns:
                    print(f"  Reached maximum pattern limit ({max_patterns})")
                    break
            except Exception as e:
                print(f"  Error processing pattern at {idx}: {str(e)}")
                continue
        
        print(f"  Processed {count} patterns, found {len(lagged_returns)} valid lagged returns")
        
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
        output_file = os.path.join(output_dir, f'{pattern}_analysis.html')
        print(f"  Saving visualization to {output_file}")
        pattern_fig.write_html(output_file)
    
    print(f"Saved pattern-specific visualizations to {output_dir}")

def generate_report(pattern_stats, output_dir):
    """
    Generate a comprehensive text report of pattern analysis findings.
    
    Args:
        pattern_stats: Dictionary of pattern statistics
        output_dir: Directory to save the report
    """
    print("Generating detailed pattern report...")
    
    output_file = os.path.join(output_dir, 'btc_doge_pattern_report.txt')
    
    with open(output_file, 'w') as f:
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
        
        if sorted_patterns:
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
    
    print(f"Saved comprehensive report to {output_file}")

# Modify the generate_index_html function to include Claude results
def generate_index_html(result_dirs, base_dir, claude_results=None):
    """Generate an index.html file listing all analysis results."""
    output_file = os.path.join(base_dir, 'index.html')
    
    print("Generating index.html...")
    
    # Sort directories by name (newest first)
    result_dirs.sort(reverse=True)
    
    html_content = """<!DOCTYPE html>
        <html>
        <head>
            <title>BTC-DOGE Pattern Analysis Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .analysis-card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }
                .analysis-card h2 { margin-top: 0; }
                .analysis-links { display: flex; flex-wrap: wrap; }
                .analysis-links a { margin-right: 15px; margin-bottom: 10px; padding: 8px 12px; 
                                   background-color: #f5f5f5; text-decoration: none; color: #333;
                                   border-radius: 4px; display: inline-block; }
                .analysis-links a:hover { background-color: #e0e0e0; }
                .timestamp { color: #666; font-size: 0.9em; margin-bottom: 10px; }
                .strategy-section { margin-top: 15px; padding-top: 10px; border-top: 1px solid #eee; }
                .download-btn { background-color: #4CAF50; color: white !important; }
                .download-btn:hover { background-color: #45a049; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px; margin: 10px 0; }
                .metric-box { border: 1px solid #ddd; border-radius: 4px; padding: 8px; text-align: center; }
                .metric-label { font-size: 0.8em; color: #666; }
                .metric-value { font-weight: bold; font-size: 1.1em; }
                .code-block { background-color: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; max-height: 300px; overflow-y: auto; }
            </style>
        </head>
        <body>
            <h1>BTC-DOGE Pattern Analysis Results</h1>
        """
    
    for dir_name in result_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
            
        html_content += f"""
            <div class="analysis-card">
                <h2>Analysis: {dir_name}</h2>
                <div class="timestamp">Timestamp: {dir_name}</div>
                <div class="analysis-links">
        """
        
        # Add links to reports
        reports_dir = os.path.join(dir_path, 'reports')
        if os.path.exists(reports_dir):
            for file in os.listdir(reports_dir):
                file_path = os.path.join(dir_name, 'reports', file)
                
                # Special handling for strategy_params.json
                if file == 'strategy_params.json':
                    # Add a link to the dedicated strategy optimization page instead of embedding
                    html_content += f'<a href="{dir_name}/html/strategy_optimization_results.html" class="download-btn">Strategy Optimization Results</a>'
                else:
                    # Regular file link
                    html_content += f'<a href="{file_path}">Report: {file}</a>'
        
        # Add links to charts
        charts_dir = os.path.join(dir_path, 'charts')
        if os.path.exists(charts_dir):
            for file in os.listdir(charts_dir):
                if file.endswith('.png'):
                    file_path = os.path.join(dir_name, 'charts', file)
                    title = file.replace('.png', '').replace('_', ' ')
                    html_content += f'<a href="{file_path}" target="_blank">Chart: {title}</a>'
        
        # Add links to HTML visualizations
        html_dir = os.path.join(dir_path, 'html')
        if os.path.exists(html_dir):
            for file in os.listdir(html_dir):
                # Skip strategy_optimization_results.html - we handle it separately
                if file.endswith('.html') and file != 'strategy_optimization_results.html':
                    file_path = os.path.join(dir_name, 'html', file)
                    title = file.replace('.html', '').replace('_', ' ')
                    html_content += f'<a href="{file_path}">Pattern: {title}</a>'
        
        html_content += """
                </div>
            </div>
        """
    
    html_content += """
        </body>
        </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Generated index.html at {output_file}")

def plot_ml_predictions(y_true, y_pred, timestamps, output_dir):
    """
    Plot ML model predictions against actual values.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        timestamps: Timestamps for the test data
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Plot actual vs predicted
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, y_true, label='Actual DOGE Returns', color='blue')
    plt.plot(timestamps, y_pred, label='XGBoost Predictions', color='red', alpha=0.7)
    plt.title('DOGE Returns: Actual vs XGBoost Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot prediction errors
    plt.subplot(2, 1, 2)
    errors = y_true - y_pred
    plt.bar(timestamps, errors, alpha=0.7, color='green')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Prediction Errors')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'xgboost_predictions.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Saved prediction plot to {output_file}")

# Modify your plot_feature_importance function to ensure proper cleanup
def plot_feature_importance(model, feature_names, output_dir):
    # Calculate feature importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(feature_names)), importance[indices], align='center')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_file)
    plt.close('all')  # Close all figures to prevent memory leaks
    
    # Return sorted feature importance for reporting
    return [(feature_names[i], importance[i]) for i in indices]