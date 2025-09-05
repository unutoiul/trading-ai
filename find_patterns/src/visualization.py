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
    
    # Create subdirectories for organized structure
    reports_dir = os.path.join(result_dir, 'reports')
    charts_dir = os.path.join(result_dir, 'charts')  
    html_dir = os.path.join(result_dir, 'html')
    data_dir = os.path.join(result_dir, 'data')
    
    # Create all directories
    os.makedirs(reports_dir)
    os.makedirs(charts_dir)
    os.makedirs(html_dir)
    os.makedirs(data_dir)
    
    return {
        'base': result_dir,
        'charts': charts_dir,
        'html': html_dir,
        'reports': reports_dir,
        'data': data_dir
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
    altcoin_name = "altcoin"
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            altcoin_name = col.split('_')[0]
            break
    
    try:
        # Create a simple overview plot
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f"Pattern Overview for {altcoin_name.upper()}", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'overview.png'))
        plt.close()
        print(f"Created overview visualization")
    except Exception as e:
        print(f"Error creating overview: {e}")

def generate_cross_asset_report(cross_asset_results, output_dir, altcoin_name):
    """Generate cross-asset analysis report."""
    print("Generating cross-asset analysis report...")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a simple HTML report
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cross-Asset Analysis: BTC-{altcoin_name.upper()}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>Cross-Asset Analysis: BTC-{altcoin_name.upper()}</h1>
        <div class="alert alert-info mt-4">
            <p>Cross-asset relationship analysis has been completed.</p>
            <p>This report shows the correlation and impact patterns between BTC and {altcoin_name.upper()}.</p>
        </div>
        <a href="index.html" class="btn btn-primary">Back to Main Report</a>
    </div>
</body>
</html>"""
        
        with open(os.path.join(output_dir, f'cross_asset_{altcoin_name}.html'), 'w') as f:
            f.write(html_content)
        
        print(f"Generated cross-asset report for {altcoin_name}")
    except Exception as e:
        print(f"Error generating cross-asset report: {e}")

def generate_index_html(results_dirs, altcoin_name):
    """Generate main index.html for results."""
    print("Generating main index.html...")
    
    try:
        # Check what files actually exist
        html_dir = results_dirs['html']
        charts_dir = results_dirs['charts']
        base_dir = results_dirs['base']
        
        # Common file patterns to check for
        possible_files = {
            'directional_strategies': 'directional_strategies.html',
            'cross_asset': f'cross_asset_{altcoin_name}.html',
            'strategy_optimization': f'strategy_optimization_{altcoin_name}.html',
            'pattern_analysis': f'pattern_analysis_{altcoin_name}.html'
        }
        
        available_reports = []
        chart_files = []
        vectorbt_reports = []
        
        # Check which files exist
        for key, filename in possible_files.items():
            file_path = os.path.join(html_dir, filename)
            if os.path.exists(file_path):
                available_reports.append((key, filename, os.path.getsize(file_path)))
        
        # Also scan for any other HTML files in the html directory
        if os.path.exists(html_dir):
            for html_file in os.listdir(html_dir):
                if html_file.endswith('.html'):
                    # Check if it's not already in our list
                    already_added = any(filename == html_file for _, filename, _ in available_reports)
                    if not already_added:
                        # Add it with a generic key derived from filename
                        key = html_file.replace('.html', '').replace('_', ' ')
                        file_path = os.path.join(html_dir, html_file)
                        available_reports.append((html_file.replace('.html', ''), html_file, os.path.getsize(file_path)))
        
        # Check for VectorBT optimization reports in subdirectories
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and item == 'optimization':
                    # Check if this optimization folder has an optimization_report.html
                    vectorbt_report = os.path.join(item_path, 'optimization_report.html')
                    if os.path.exists(vectorbt_report):
                        vectorbt_reports.append((item, os.path.getsize(vectorbt_report)))
        
        # Also check for optimization_report.html directly in the base directory
        direct_optimization_report = os.path.join(base_dir, 'optimization_report.html')
        if os.path.exists(direct_optimization_report):
            vectorbt_reports.append(('optimization', os.path.getsize(direct_optimization_report)))
        
        # Check for chart files
        if os.path.exists(charts_dir):
            chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>BTC-{altcoin_name.upper()} Analysis Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .analysis-card {{ transition: all 0.3s ease; }}
        .analysis-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        .chart-item {{ border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; }}
        .chart-item img {{ width: 100%; height: auto; min-height: 400px; object-fit: contain; }}
    </style>
</head>
<body>
    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <div class="card border-0 shadow-sm mb-4">
                    <div class="card-body">
                        <h1 class="h2 mb-0">BTC-{altcoin_name.upper()} Analysis Results</h1>
                        <p class="text-muted">Pattern analysis and trading strategy performance</p>
                    </div>
                </div>
            </div>
        </div>"""
        
        if available_reports or vectorbt_reports:
            html_content += """
        <div class="row mb-4">
            <div class="col-12">
                <h3 class="mb-3">Analysis Reports</h3>
                <div class="row">"""
            
            # Add regular reports - exclude directional_strategies as they're mostly empty templates
            for key, filename, file_size in available_reports:
                # Skip directional_strategies as they don't contain meaningful data
                if key == 'directional_strategies':
                    continue
                    
                title = key.replace('_', ' ').title()
                if key == 'cross_asset':
                    title = 'Cross-Asset Analysis'
                    description = 'Analyze correlations and relationships between BTC and altcoin'
                    btn_class = 'btn-primary'
                elif key == 'strategy_optimization':
                    title = 'Strategy Optimization'
                    description = 'Parameter optimization and performance analysis'
                    btn_class = 'btn-info'
                else:
                    title = key.replace('_', ' ').title()
                    description = 'View detailed analysis results'
                    btn_class = 'btn-outline-primary'
                
                size_kb = file_size / 1024
                
                html_content += f"""
                    <div class="col-md-6 col-lg-4 mb-3">
                        <div class="card analysis-card border-0 shadow-sm h-100">
                            <div class="card-body">
                                <h5 class="card-title">{title}</h5>
                                <p class="card-text">{description}</p>
                                <p class="small text-muted">File size: {size_kb:.1f} KB</p>
                                <a href="../html/{filename}" class="btn {btn_class}">View Report</a>
                            </div>
                        </div>
                    </div>"""
            
            # Add VectorBT optimization reports
            for vectorbt_folder, file_size in vectorbt_reports:
                size_kb = file_size / 1024
                
                html_content += f"""
                    <div class="col-md-6 col-lg-4 mb-3">
                        <div class="card analysis-card border-0 shadow-sm h-100">
                            <div class="card-body">
                                <h5 class="card-title">Strategy Optimization</h5>
                                <p class="card-text">Advanced parameter optimization and backtesting results</p>
                                <p class="small text-muted">Size: {size_kb:.1f} KB</p>
                                <a href="../optimization_report.html" class="btn btn-warning">View Optimization</a>
                                <div class="mt-2">
                                    <small class="text-muted">Download data:</small><br>
                                    <a href="../data/all_results.csv" class="btn btn-outline-primary btn-sm me-1">📊 All Results CSV</a>
                                    <a href="../data/top_100_strategies.csv" class="btn btn-outline-success btn-sm me-1">🏆 Top 100 CSV</a>
                                    <a href="../data/parameter_analysis.csv" class="btn btn-outline-info btn-sm">⚙️ Parameters CSV</a>
                                    <br>
                                    <a href="../data/individual_trades.csv" class="btn btn-outline-warning btn-sm me-1 mt-1">📝 Individual Trades CSV</a>
                                    <a href="../data/trades_summary.csv" class="btn btn-outline-secondary btn-sm mt-1">📋 Trades Summary CSV</a>
                                </div>
                            </div>
                        </div>
                    </div>"""
            
            html_content += """
                </div>
            </div>
        </div>"""
        
        if chart_files:
            html_content += f"""
        <div class="row">
            <div class="col-12">
                <h3 class="mb-3">Generated Charts ({len(chart_files)} charts)</h3>
                <div class="chart-grid">"""
            
            for chart_file in sorted(chart_files):
                chart_name = chart_file.replace('.png', '').replace('_', ' ').title()
                html_content += f"""
                    <div class="chart-item">
                        <h6>{chart_name}</h6>
                        <img src="../charts/{chart_file}" class="img-fluid" alt="{chart_name}">
                    </div>"""
            
            html_content += """
                </div>
            </div>
        </div>"""
        
        if not available_reports and not chart_files and not vectorbt_reports:
            html_content += """
        <div class="row">
            <div class="col-12">
                <div class="alert alert-warning">
                    <h4>No Reports Generated</h4>
                    <p>The analysis completed but no report files were found. This could indicate an error during report generation.</p>
                </div>
            </div>
        </div>"""
        
        html_content += """
        <div class="row mt-4">
            <div class="col-12">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h5 class="mb-3">Quick Actions</h5>
                        <a href="../../index.html" class="btn btn-outline-primary me-2">← Back to All Results</a>
                        <a href="/" class="btn btn-outline-secondary">Run New Analysis</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        with open(os.path.join(results_dirs['reports'], 'index.html'), 'w') as f:
            f.write(html_content)
        
        # Also update the master index in the main results directory
        update_master_index()
        
        print("Generated main index.html")
    except Exception as e:
        print(f"Error generating index.html: {e}")

def update_master_index():
    """Update the master index.html that lists all analysis results."""
    import glob
    
    try:
        # Get the results base directory (relative to the current working directory)
        results_base = "results"
        
        # Find all timestamped directories
        timestamp_dirs = []
        if os.path.exists(results_base):
            for item in os.listdir(results_base):
                item_path = os.path.join(results_base, item)
                if os.path.isdir(item_path) and "." in item and "-" in item:
                    # This looks like a timestamp directory (DD.MM.YYYY-HH.MM.SS)
                    try:
                        # Try to parse the timestamp to validate it
                        parts = item.split('-')
                        if len(parts) == 2:
                            date_part, time_part = parts
                            datetime.strptime(date_part, "%d.%m.%Y")
                            datetime.strptime(time_part, "%H.%M.%S")
                            timestamp_dirs.append(item)
                    except ValueError:
                        continue
        
        # Sort by timestamp (newest first)
        timestamp_dirs.sort(reverse=True)
        
        # Generate HTML content with simplified interface
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Analysis Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f8f9fa; }}
        .analysis-card {{ transition: all 0.3s ease; border-radius: 12px; }}
        .analysis-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.1); }}
        .timestamp {{ font-family: 'Courier New', monospace; color: #6c757d; }}
        .analysis-badge {{ font-size: 0.85em; }}
        .intro-text {{ font-size: 1.1em; line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="container mt-4">
        <!-- Header -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-0 shadow-sm">
                    <div class="card-body p-4">
                        <h1 class="h2 mb-3 text-primary">📈 Trading Analysis Dashboard</h1>
                        <p class="intro-text text-muted mb-3">
                            Comprehensive cryptocurrency trading pattern analysis and strategy optimization results. 
                            Each analysis provides insights into BTC-altcoin correlations, pattern recognition, 
                            strategy backtesting, and performance optimization.
                        </p>
                        <div class="alert alert-info border-0 mb-0">
                            <h6 class="mb-2">📊 What's inside each analysis:</h6>
                            <div class="row">
                                <div class="col-md-6">
                                    <ul class="mb-0 small">
                                        <li><strong>Advanced Optimization:</strong> Comprehensive backtesting with 2000+ parameter combinations</li>
                                        <li><strong>Pattern Analysis:</strong> Statistical correlations & trading signals</li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <ul class="mb-0 small">
                                        <li><strong>Individual Trades:</strong> Detailed trade-by-trade analysis</li>
                                        <li><strong>Performance Charts:</strong> Visual analysis & parameter trends</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Results List -->
        <div class="row">
            <div class="col-12">
                <h3 class="mb-3">Recent Analysis Results 
                    <span class="badge bg-primary analysis-badge">{} total</span>
                </h3>
                
                {}
                
                <!-- Simple Footer -->
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="text-center">
                            <a href="/" class="btn btn-outline-primary btn-lg me-3">
                                <i class="fas fa-plus me-2"></i>Run New Analysis
                            </a>
                            <a href="javascript:location.reload()" class="btn btn-outline-secondary btn-lg">
                                <i class="fas fa-refresh me-2"></i>Refresh Results
                            </a>
                        </div>
                        <hr class="my-4">
                        <div class="text-center text-muted small">
                            <p class="mb-1">Total Analyses: <strong>{}</strong> • Latest: <strong>{}</strong></p>
                            <p class="mb-0">💾 Results stored in <code>results/</code> directory</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://kit.fontawesome.com/your-fontawesome-kit.js" crossorigin="anonymous"></script>
</body>
</html>"""
        
        # Generate cards for each analysis
        if timestamp_dirs:
            cards_html = ""
            for i, timestamp_dir in enumerate(timestamp_dirs):
                # Parse timestamp for display
                try:
                    parts = timestamp_dir.split('-')
                    date_part, time_part = parts
                    dt = datetime.strptime(f"{date_part} {time_part}", "%d.%m.%Y %H.%M.%S")
                    formatted_date = dt.strftime("%B %d, %Y")
                    formatted_time = dt.strftime("%I:%M %p")
                    
                    # Check what files exist in this directory
                    dir_path = os.path.join(results_base, timestamp_dir)
                    has_index = os.path.exists(os.path.join(dir_path, "reports", "index.html"))
                    has_charts = os.path.exists(os.path.join(dir_path, "charts")) and len(os.listdir(os.path.join(dir_path, "charts"))) > 0
                    has_html = os.path.exists(os.path.join(dir_path, "html"))
                    
                    # Determine status
                    if has_index and has_charts:
                        status = '<span class="badge bg-success status-badge">Complete</span>'
                    elif has_index:
                        status = '<span class="badge bg-warning status-badge">Partial</span>'
                    else:
                        status = '<span class="badge bg-secondary status-badge">Basic</span>'
                    
                    # Create simplified card
                    card_html = f"""
                <div class="card analysis-card border-0 shadow-sm mb-3">
                    <div class="card-body p-4">
                        <div class="row align-items-center">
                            <div class="col-md-9">
                                <h5 class="card-title mb-2">
                                    📊 Analysis #{len(timestamp_dirs) - i:02d} - {formatted_date}
                                    {status}
                                </h5>
                                <p class="text-muted mb-2">
                                    <span class="timestamp">{formatted_time}</span> • 
                                    <small class="text-muted">{timestamp_dir}</small>
                                </p>
                                <div class="small">
                                    {'<span class="text-success">✅ Charts Available</span>' if has_charts else '<span class="text-muted">❌ No Charts</span>'} • 
                                    {'<span class="text-success">✅ Strategy Reports</span>' if has_html else '<span class="text-muted">❌ No Reports</span>'} • 
                                    {'<span class="text-success">✅ Full Analysis</span>' if has_index else '<span class="text-muted">❌ Incomplete</span>'}
                                </div>
                            </div>
                            <div class="col-md-3 text-end">"""
                    
                    if has_index:
                        card_html += f'''
                                <a href="{timestamp_dir}/reports/index.html" class="btn btn-primary btn-lg">
                                    <i class="fas fa-chart-line me-2"></i>View Results
                                </a>'''
                    else:
                        card_html += '''
                                <button class="btn btn-outline-secondary btn-lg" disabled>
                                    <i class="fas fa-exclamation-triangle me-2"></i>Incomplete
                                </button>'''
                    
                    card_html += """
                            </div>
                        </div>
                    </div>
                </div>"""
                    
                    cards_html += card_html
                    
                except ValueError:
                    continue
            
            # Format the final HTML
            latest_analysis = timestamp_dirs[0] if timestamp_dirs else "None"
            try:
                latest_dt = datetime.strptime(latest_analysis, "%d.%m.%Y-%H.%M.%S")
                latest_formatted = latest_dt.strftime("%B %d, %Y at %I:%M %p")
            except:
                latest_formatted = latest_analysis
                
            final_html = html_content.format(
                len(timestamp_dirs),  # Number in badge
                cards_html if cards_html else '<div class="alert alert-info">No analysis results found. <a href="/">Run your first analysis</a>.</div>',
                len(timestamp_dirs),  # Total analyses
                latest_formatted     # Latest analysis time
            )
        else:
            # No analyses found
            final_html = html_content.format(
                0,  # Number in badge
                '<div class="alert alert-info">No analysis results found. <a href="/" class="alert-link">Run your first analysis</a> to get started.</div>',
                0,  # Total analyses  
                "None"  # Latest analysis time
            )
        
        # Write the master index
        master_index_path = os.path.join(results_base, "index.html")
        with open(master_index_path, 'w') as f:
            f.write(final_html)
        
        print(f"Updated master index at: {master_index_path}")
        
    except Exception as e:
        print(f"Error updating master index: {e}")

def update_index_html(results_dirs, altcoin_name):
    """Update the main index.html with latest results."""
    print("Updating index.html...")
    
    try:
        # Simply call generate_index_html to create/update
        generate_index_html(results_dirs, altcoin_name)
        print("Updated index.html")
    except Exception as e:
        print(f"Error updating index.html: {e}")