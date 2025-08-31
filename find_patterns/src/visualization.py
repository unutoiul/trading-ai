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
import glob

def get_return_class(stats, lag=1):
    # """Get CSS class based on return value."""
    return 'up-value' if stats['lags'].get(lag, {}).get('avg_return', 0) > 0 else 'down-value'

def format_return_value(stats, lag=1):
    """Format return value with win rate."""
    avg_return = stats['lags'].get(lag, {}).get('avg_return', 0) * 100
    win_rate = stats['lags'].get(lag, {}).get('win_rate', 0) * 100
    return f"{avg_return:.4f}% ({win_rate:.1f}%)"

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
    
    # Create subdirectories - charts directly in results dir, not inside reports
    reports_dir = os.path.join(result_dir, 'reports')
    charts_dir = os.path.join(result_dir, 'charts')  
    html_dir = os.path.join(result_dir, 'html')
    
    # Create all directories
    os.makedirs(reports_dir)
    os.makedirs(charts_dir)
    os.makedirs(html_dir)
    
    return {
        'base': result_dir,
        'charts': charts_dir,
        'html': html_dir,
        'reports': reports_dir
    }

def plot_lag_responses(pattern_stats, output_dir):
    """Create lag response plots for patterns."""
    print("Creating lag response plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through patterns
    for pattern, stats in pattern_stats.items():
        if pattern == 'no_patterns_detected':
            continue
            
        try:
            if 'returns' not in stats or not stats['returns']:
                print(f"No lag return data available for pattern {pattern}")
                continue
                
            # Create a figure for this pattern
            plt.figure(figsize=(10, 8))
            
            # Convert the lag returns from dict to lists for plotting
            lags = list(stats['returns'].keys())
            returns = list(stats['returns'].values())
            win_rates = [stats['win_rates'].get(lag, 0) * 100 for lag in lags]
            
            # Create stacked subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Plot returns
            ax1.plot(lags, returns, 'b-o', linewidth=2)
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax1.set_ylabel('Average Return')
            ax1.set_title(f'Lag Response for {pattern}')
            
            # Highlight optimal lag
            opt_lag = stats['optimal_lag']
            ax1.axvline(x=opt_lag, color='g', linestyle='--', alpha=0.8)
            
            # Plot win rate
            ax2.plot(lags, win_rates, 'g-o', linewidth=2)
            ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Lag (minutes)')
            ax2.set_ylabel('Win Rate (%)')
            
            # Set up grid
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # Add annotation for optimal lag
            opt_return = stats['returns'].get(opt_lag, 0)
            opt_win_rate = stats['win_rates'].get(opt_lag, 0) * 100
            
            ax1.annotate(f'Optimal lag: {opt_lag} min\nReturn: {opt_return:.6f}\nWin rate: {opt_win_rate:.1f}%',
                        xy=(opt_lag, opt_return),
                        xytext=(opt_lag + 1, opt_return),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7))
            
            # Add correlation info
            corr = stats.get('correlation', 0)
            plt.figtext(0.5, 0.01, f"Correlation: {corr:.4f}", ha="center", 
                       bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.03, 1, 1])
            plt.savefig(os.path.join(output_dir, f'lag_response_{pattern}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error creating lag plot for {pattern}: {e}")
            # Create a placeholder error plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating lag plot: {e}", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(os.path.join(output_dir, f'lag_response_{pattern}_error.png'))
            plt.close()

def create_interactive_overview(combined_data, pattern_stats, output_dir):
    """Create a comprehensive interactive visualization of patterns and responses."""
    print("Creating interactive overview visualization...")
    
    # Detect altcoin