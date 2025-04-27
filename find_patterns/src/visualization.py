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
    
    # Detect altcoin name from the columns in combined_data
    altcoin_name = "unknown"
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            altcoin_name = col.split('_')[0].upper()
            break
    
    # Make altcoin name lowercase for file paths
    altcoin_lowercase = altcoin_name.lower()
    
    # Create the output filename based on altcoin name
    output_file = os.path.join(output_dir, f'btc_{altcoin_lowercase}_pattern_analysis.html')
    print(f"Creating pattern analysis HTML at: {output_file}")
    
    try:
        # Try to use plotly if available
        # ... (existing plotly code)
        pass
    except Exception as e:
        print(f"Error creating interactive visualization: {e}")
        print("Creating simple HTML report instead")
        
        # Create a simple HTML report with dynamic altcoin name
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>BTC-{altcoin_name} Pattern Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">BTC-{altcoin_name} Pattern Analysis</h1>
        
        <div class="alert alert-info">
            <p>This report shows the relationships between Bitcoin patterns and {altcoin_name} price movements.</p>
        </div>
        
        <h2>Top Patterns by Correlation</h2>
        <div class="table-responsive">
            <table class="table table-bordered">
                <thead>
                    <tr>
                        <th>Pattern</th>
                        <th>Instances</th>
                        <th>Optimal Lag</th>
                        <th>Correlation</th>
                        <th>Avg Return</th>
                        <th>Win Rate</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Sort patterns by correlation strength
        sorted_patterns = sorted(pattern_stats.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
        
        # Add top patterns to table
        for pattern, stats in sorted_patterns:
            html_content += f"""
                    <tr>
                        <td>{pattern.replace('_', ' ').title()}</td>
                        <td>{stats['instances']}</td>
                        <td>{stats['optimal_lag']} min</td>
                        <td>{stats['correlation']:.4f}</td>
                        <td>{stats['avg_return']:.6f}</td>
                        <td>{stats['win_rate']*100:.1f}%</td>
                    </tr>"""
        
        html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="mt-4">
            <p>View chart images in the Charts directory for visual analysis of these patterns.</p>
            <a href="../charts/index.html" class="btn btn-primary">View Charts</a>
"""
        
        # Use the actual altcoin name in the file path instead of "altcoin"
        html_content += f"""            <a href="../reports/btc_{altcoin_lowercase}_pattern_report.txt" class="btn btn-secondary" target="_blank">View Full Report</a>
            <a href="index.html" class="btn btn-outline-secondary">Back to Overview</a>
        </div>
    </div>
</body>
</html>"""
        
        # Write HTML to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
    print(f"Saved interactive overview to {output_file}")
    return altcoin_name.lower()  # Return detected altcoin name for use in other functions

def create_pattern_specific_plots(combined_data, pattern_stats, output_dir):
    print("Creating pattern-specific visualizations...")
    
    # Sort patterns by correlation strength
    top_patterns = sorted(pattern_stats.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)[:3]
    
    for pattern, stats in top_patterns:
        pattern_name = pattern.replace('btc_pattern_', '').replace('_', ' ').title()
        print(f"Creating visualization for pattern: {pattern_name}")
        
        try:
            # Create the plot (implementation might be missing)
            fig = plt.figure(figsize=(10, 6))
            plt.title(f"Pattern: {pattern_name}")
            
            # Try to plot the lag returns
            if 'returns' in stats and stats['returns']:
                plt.plot(list(stats['returns'].keys()), 
                         list(stats['returns'].values()), 
                         marker='o', linestyle='-')
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                plt.axvline(x=stats['optimal_lag'], color='g', linestyle='--')
                plt.xlabel('Lag (minutes)')
                plt.ylabel('Average Return')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, "No data available for this pattern", 
                         ha='center', va='center')
            
            # Save the figure
            output_file = os.path.join(output_dir, f'pattern_{pattern}.png')
            plt.savefig(output_file)
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualization for {pattern}: {e}")
    
    print(f"Saved pattern-specific visualizations to {output_dir}")

def generate_report(pattern_stats, output_dir, altcoin_name="altcoin"):
    """Generate a comprehensive text report of pattern analysis results."""
    # Create output file with dynamic name
    output_file = os.path.join(output_dir, f'btc_{altcoin_name.lower()}_pattern_report.txt')
    
    with open(output_file, 'w') as f:
        f.write(f"BTC-{altcoin_name.upper()} PATTERN ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        
        # Sort patterns by correlation strength
        sorted_patterns = sorted(pattern_stats.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
        
        # Write summary of most predictive patterns
        f.write("SUMMARY OF PREDICTIVE PATTERNS\n")
        f.write("-" * 30 + "\n\n")
        
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
            
        # Rest of the function continues as before...
    
    print(f"Saved comprehensive report to {output_file}")
    return output_file

def generate_cross_asset_report(cross_asset_results, output_dir, altcoin_name):
    """Generate report on cross-asset relationships between BTC and altcoin."""
    if not cross_asset_results:
        return
        
    output_file = os.path.join(output_dir, 'cross_asset_relationship_report.txt')
    
    with open(output_file, 'w') as f:
        f.write(f"BTC TO {altcoin_name.upper()} RELATIONSHIP ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        # For each BTC pattern, show its effect on the altcoin
        for btc_pattern, lag_data in cross_asset_results.items():
            f.write(f"\n{btc_pattern.upper()}\n")
            f.write("-" * len(btc_pattern) + "\n")
            f.write(f"Total instances: {lag_data.get(list(lag_data.keys())[0], {}).get('sample_size', 0) if lag_data else 'Unknown'}\n\n")
            
            # Create a table of lag results
            f.write(f"{'Lag (min)':<10}{'Avg Return %':<15}{'Win Rate %':<15}{'Top Triggered Patterns':<50}\n")
            f.write("-" * 90 + "\n")
            
            # Sort lags by minute value
            for lag_min in sorted(lag_data.keys()):
                stats = lag_data[lag_min]
                top_patterns = ", ".join([f"{p[0]} ({p[1]*100:.1f}%)" for p in stats.get('top_patterns', [])[:3]])
                f.write(f"{lag_min:<10}{stats.get('avg_return', 0)*100:<15.4f}{stats.get('win_rate', 0)*100:<15.1f}{top_patterns:<50}\n")
            
            f.write("\n")
    
    print(f"Generated cross-asset relationship report: {output_file}")
    return output_file

def generate_index_html(result_dirs, base_dir):
    """Generate index.html with links to all result directories."""
    print("Generating main results index...")
    
    # Create HTML content directly instead of loading from template file
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Pattern Analysis Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .card {{ margin-bottom: 20px; }}
        h1 {{ margin-bottom: 30px; }}
        .report-buttons {{ margin-top: 10px; }}
        .report-btn {{ margin-right: 5px; margin-bottom: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1>BTC-Altcoin Pattern Analysis Results</h1>
            <a href="/" class="btn btn-primary">Back to Analysis Tool</a>
        </div>
        
        <div class="alert alert-info mb-4">
            <p>Select an analysis result to view detailed reports, charts, and strategy recommendations.</p>
        </div>
        
        <h2 class="mb-3">Analysis Results</h2>
        <div id="results-container">
"""
    
    # Add each result directory as a card
    for result_dir in sorted(result_dirs, reverse=True):
        # Format timestamp for display
        try:
            timestamp = datetime.strptime(result_dir, '%d.%m.%Y-%H.%M.%S')
            formatted_date = timestamp.strftime('%B %d, %Y %I:%M %p')
        except:
            formatted_date = result_dir
        
        # Check if there's an index.html in the html subfolder
        result_path = os.path.join(base_dir, result_dir)
        html_dir = os.path.join(result_path, 'html')
        index_path = os.path.join(html_dir, 'index.html')
        
        html_content += f'<div class="card mb-3">\n'
        html_content += f'  <div class="card-header bg-light">{formatted_date}</div>\n'
        html_content += f'  <div class="card-body">\n'
        
        # Primary button for the main results index
        if os.path.exists(index_path):
            html_content += f'    <a href="{result_dir}/html/index.html" class="btn btn-primary mb-2">View Analysis Results</a>\n'
        
        # Check for strategy optimization results specifically
        strategy_file = os.path.join(html_dir, 'strategy_optimization_results.html')
        if os.path.exists(strategy_file):
            html_content += f'    <a href="{result_dir}/html/strategy_optimization_results.html" class="btn btn-success me-2 mb-2">Strategy Optimization Results</a>\n'
        
        # Add individual buttons for reports directory
        reports_dir = os.path.join(result_path, 'reports')
        if os.path.exists(reports_dir):
            html_content += f'    <div class="report-buttons">\n'
            html_content += f'      <strong>Reports:</strong>\n'
            
            for file in sorted(os.listdir(reports_dir)):
                if file.endswith(('.txt', '.json', '.csv', '.pine')):
                    display_name = file.replace('_', ' ').title().split('.')[0]
                    button_color = 'btn-outline-info'
                    
                    # Different button styles based on file type
                    if file.endswith('.txt'):
                        button_color = 'btn-outline-primary'
                    elif file.endswith('.json'):
                        button_color = 'btn-outline-success'
                    elif file.endswith('.pine'):
                        button_color = 'btn-outline-warning'
                    
                    html_content += f'      <a href="{result_dir}/reports/{file}" class="btn btn-sm {button_color} report-btn" target="_blank">{display_name}</a>\n'
            
            html_content += f'    </div>\n'
        
        # Add link to charts directory
        charts_dir = os.path.join(result_path, 'charts')
        if os.path.exists(charts_dir) and any(file.endswith(('.png', '.jpg', '.jpeg')) for file in os.listdir(charts_dir)):
            html_content += f'    <hr>\n'
            html_content += f'    <a href="{result_dir}/charts/index.html" class="btn btn-outline-secondary me-2 mt-2">View All Charts</a>\n'
        
        html_content += f'  </div>\n'
        html_content += f'</div>\n'
    
    # Complete the HTML document
    html_content += """
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
    
    # Write index.html
    with open(os.path.join(base_dir, 'index.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Generated index.html with links to {len(result_dirs)} result directories")

def plot_feature_importance(model, feature_names, output_dir):
    """Create plot of feature importance from XGBoost model."""
    try:
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
        
        print(f"Feature importance chart saved to {output_file}")
        
        # Return sorted feature importance for reporting
        return [(feature_names[i], importance[i]) for i in indices]
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
        # Create a placeholder chart
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Feature importance calculation failed", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.tight_layout()
        
        # Save placeholder
        output_file = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(output_file)
        plt.close('all')
        
        return []  # Return empty list on error
    


def update_index_html(results_dirs, altcoin_name):
    """Update or create index.html files for a specific analysis run.
    
    Args:
        results_dirs: Dictionary containing paths to different result directories
        altcoin_name: Name of the altcoin being analyzed
    
    Returns:
        Tuple of (main_index_path, html_index_path) with the paths to the created files
    """
    import os
    from datetime import datetime
    
    # Get the base directory and HTML directory
    base_dir = results_dirs['base']
    html_dir = results_dirs['html']
    reports_dir = results_dirs['reports']
    charts_dir = results_dirs['charts']
    
    # Get current timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check what files are available
    has_directional_strategies = os.path.exists(os.path.join(html_dir, 'directional_strategies.html'))
    has_pattern_analysis = os.path.exists(os.path.join(html_dir, 'btc_' + altcoin_name.lower() + '_pattern_analysis.html'))
    has_strategy_optimization = os.path.exists(os.path.join(html_dir, 'strategy_optimization_results.html'))
    
    # Check for report files
    report_files = []
    if os.path.exists(reports_dir):
        report_files = [f for f in os.listdir(reports_dir) 
                      if os.path.isfile(os.path.join(reports_dir, f)) and 
                      f.endswith(('.txt', '.json', '.csv'))]
    
    # Check for chart files
    chart_files = []
    if os.path.exists(charts_dir):
        chart_files = [f for f in os.listdir(charts_dir) 
                     if os.path.isfile(os.path.join(charts_dir, f)) and 
                     f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 1. First create the main index.html in the base directory
    main_index_path = os.path.join(base_dir, 'index.html')
    with open(main_index_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>BTC-{altcoin_name.upper()} Analysis Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .card {{ margin-bottom: 20px; transition: all 0.3s; }}
        .card:hover {{ transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="mb-4">BTC-{altcoin_name.upper()} Analysis Dashboard</h1>
        
        <div class="alert alert-info">
            <p>Analysis completed on {now}</p>
            <a href="../index.html" class="btn btn-sm btn-outline-primary">Back to All Reports</a>
        </div>
        
        <div class="row">
""")

        # Add cards for each available report type
        if has_directional_strategies:
            f.write(f"""
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Trading Strategies</h5>
                    </div>
                    <div class="card-body">
                        <p>View machine learning-based trading strategies that analyze how BTC movements affect {altcoin_name.upper()}</p>
                        <a href="html/directional_strategies.html" class="btn btn-primary">View Strategies</a>
                    </div>
                </div>
            </div>
""")

        if has_pattern_analysis:
            f.write(f"""
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    <div class="card-header bg-success text-white">
                        <h5 class="mb-0">Pattern Analysis</h5>
                    </div>
                    <div class="card-body">
                        <p>View detailed analysis of BTC price patterns and their impact on {altcoin_name.upper()}</p>
                        <a href="html/btc_{altcoin_name.lower()}_pattern_analysis.html" class="btn btn-success">View Patterns</a>
                    </div>
                </div>
            </div>
""")

        if has_strategy_optimization:
            f.write(f"""
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    <div class="card-header bg-info text-white">
                        <h5 class="mb-0">Strategy Optimization</h5>
                    </div>
                    <div class="card-body">
                        <p>View optimization results for trading parameters like stop loss and take profit levels</p>
                        <a href="html/strategy_optimization_results.html" class="btn btn-info">View Optimization</a>
                    </div>
                </div>
            </div>
""")

        # Add reports section if there are any reports
        if report_files:
            f.write(f"""
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    <div class="card-header bg-secondary text-white">
                        <h5 class="mb-0">Analysis Reports</h5>
                    </div>
                    <div class="card-body">
                        <p>View detailed technical reports and data analysis</p>
                        <div class="list-group" style="max-height: 200px; overflow-y: auto;">
""")
            
            # Add links to top 5 reports
            for report in sorted(report_files)[:5]:
                display_name = report.replace('_', ' ').title().split('.')[0]
                f.write(f"""
                            <a href="reports/{report}" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                {display_name}
                                <span class="badge bg-secondary">{report.split('.')[-1].upper()}</span>
                            </a>
""")
                
            # Add a view all link if there are more than 5 reports
            if len(report_files) > 5:
                # Create reports index.html on the fly
                reports_index_path = os.path.join(reports_dir, 'index.html')
                with open(reports_index_path, 'w') as rf:
                    rf.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Analysis Reports</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Analysis Reports for BTC-{altcoin_name.upper()}</h1>
        <div class="mb-3">
            <a href="../index.html" class="btn btn-primary">Back to Dashboard</a>
        </div>
        <div class="list-group">
""")
                    for report in sorted(report_files):
                        display_name = report.replace('_', ' ').title().split('.')[0]
                        rf.write(f"""
            <a href="{report}" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                {display_name}
                <span class="badge bg-primary rounded-pill">{report.split('.')[-1].upper()}</span>
            </a>
""")
                    rf.write("""
        </div>
    </div>
</body>
</html>
""")
                
                # Add link to the reports index
                f.write(f"""
                            <a href="reports/index.html" class="list-group-item list-group-item-action text-center">
                                View all {len(report_files)} reports
                            </a>
""")
                
            f.write("""
                        </div>
                    </div>
                </div>
            </div>
""")

        # Add charts section if there are any charts
        if chart_files:
            f.write(f"""
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    <div class="card-header bg-danger text-white">
                        <h5 class="mb-0">Analysis Charts</h5>
                    </div>
                    <div class="card-body">
                        <p>View charts and visualizations of patterns and relationships</p>
""")
            
            # Create charts index.html on the fly
            charts_index_path = os.path.join(charts_dir, 'index.html')
            with open(charts_index_path, 'w') as cf:
                cf.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Analysis Charts</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Analysis Charts for BTC-{altcoin_name.upper()}</h1>
        <div class="mb-3">
            <a href="../index.html" class="btn btn-primary">Back to Dashboard</a>
        </div>
        <div class="row">
""")
                for chart in sorted(chart_files):
                    nice_name = chart.replace('_', ' ').split('.')[0].title()
                    cf.write(f"""
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header">{nice_name}</div>
                    <div class="card-body p-0">
                        <img src="{chart}" class="img-fluid" alt="{nice_name}">
                    </div>
                </div>
            </div>
""")
                cf.write("""
        </div>
    </div>
</body>
</html>
""")
            
            # Add thumbnail of first chart and link to all charts
            if chart_files:
                first_chart = sorted(chart_files)[0]
                f.write(f"""
                        <div class="text-center mb-3">
                            <img src="charts/{first_chart}" class="img-fluid img-thumbnail" style="max-height: 150px;" alt="Sample Chart">
                        </div>
                        <a href="charts/index.html" class="btn btn-danger">View All Charts ({len(chart_files)})</a>
""")
            else:
                f.write(f"""
                        <a href="charts/index.html" class="btn btn-danger">View All Charts</a>
""")
                
            f.write("""
                    </div>
                </div>
            </div>
""")

        f.write("""
        </div>
    </div>
</body>
</html>
""")

    # 2. Create a simplified version of the index in the HTML directory too
    html_index_path = os.path.join(html_dir, 'index.html')
    with open(html_index_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>BTC-{altcoin_name.upper()} Analysis Menu</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>BTC-{altcoin_name.upper()} Analysis Results</h1>
        
        <div class="alert alert-info">
            <a href="../index.html" class="btn btn-primary">Back to Main Dashboard</a>
        </div>
        
        <div class="list-group">
""")

        if has_directional_strategies:
            f.write(f"""
            <a href="directional_strategies.html" class="list-group-item list-group-item-action">Trading Strategies</a>
""")

        if has_pattern_analysis:
            f.write(f"""
            <a href="btc_{altcoin_name.lower()}_pattern_analysis.html" class="list-group-item list-group-item-action">Pattern Analysis</a>
""")

        if has_strategy_optimization:
            f.write(f"""
            <a href="strategy_optimization_results.html" class="list-group-item list-group-item-action">Strategy Optimization</a>
""")

        f.write("""
        </div>
    </div>
</body>
</html>
""")
    
    print(f"Updated main index.html at {main_index_path}")
    print(f"Updated HTML index.html at {html_index_path}")
    
    return main_index_path, html_index_path

def generate_pattern_html_reports(pattern_stats, output_dir, altcoin_name="altcoin", 
                                 ml_results=None, cross_asset_results=None, directional_impact=None):
    """Generate HTML reports for each pattern analysis with ML and cross-asset insights."""
    print("Generating pattern HTML reports...")
    
    # Create HTML directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First, create the main pattern analysis page
    main_output_file = os.path.join(output_dir, f"btc_{altcoin_name.lower()}_pattern_analysis.html")
    
    # Generate main page content with ML insights if available
    main_html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>BTC-{altcoin_name.upper()} Pattern Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .pattern-card {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metrics-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }}
        .metric-box {{ flex: 1; min-width: 120px; padding: 10px; background: #f8f9fa; border-radius: 4px; text-align: center; }}
        .pattern-link {{ color: #007bff; text-decoration: none; }}
        .pattern-link:hover {{ text-decoration: underline; }}
        .up-value {{ color: #28a745; font-weight: bold; }}
        .down-value {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">BTC-{altcoin_name.upper()} Pattern Analysis</h1>
        
        <!-- ML Insights Section (if available) -->
        {f'''
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h2 class="h4 mb-0">Machine Learning Insights</h2>
            </div>
            <div class="card-body">
                <h4>Top Influential Features:</h4>
                <ul class="list-group mb-3">
                    {"".join([f'<li class="list-group-item d-flex justify-content-between align-items-center">{feature} <span class="badge bg-primary rounded-pill">{importance:.4f}</span></li>' for feature, importance in ml_results["feature_importance"][:5]])}
                </ul>
                <p><strong>Model Accuracy:</strong> {ml_results["metrics"]["directional_accuracy"]:.1%}</p>
                <a href="../reports/xgboost_analysis_report.txt" class="btn btn-sm btn-outline-primary">View Full ML Report</a>
            </div>
        </div>
        ''' if ml_results else ''}
        
        <!-- Directional Impact Analysis Section (new) -->
        {f'''
        <div class="card mb-4">
            <div class="card-header bg-warning text-dark">
                <h2 class="h4 mb-0">BTC Directional Impact on {altcoin_name.upper()}</h2>
            </div>
            <div class="card-body">
                <p>How {altcoin_name.upper()} reacts to different Bitcoin price moves:</p>
                
                <div class="table-responsive">
                    <table class="table table-sm table-hover">
                        <thead>
                            <tr>
                                <th>BTC Movement</th>
                                <th>1m Impact</th>
                                <th>5m Impact</th>
                                <th>15m Impact</th>
                                <th>Instances</th>
                            </tr>
                        </thead>
                        <tbody>
                        {"".join([f"""
                            <tr>
                                <td><strong>{scenario.replace('btc_', '').replace('_', ' ').title()}</strong></td>
                                <td class="{'up-value' if stats['lags'].get(1, {}).get('avg_return', 0) > 0 else 'down-value'}">{stats['lags'].get(1, {}).get('avg_return', 0)*100:.4f}% ({stats['lags'].get(1, {}).get('win_rate', 0)*100:.1f}%)</td>
                                <td class="{'up-value' if stats['lags'].get(5, {}).get('avg_return', 0) > 0 else 'down-value'}">{stats['lags'].get(5, {}).get('avg_return', 0)*100:.4f}% ({stats['lags'].get(5, {}).get('win_rate', 0)*100:.1f}%)</td>
                                <td class="{'up-value' if stats['lags'].get(15, {}).get('avg_return', 0) > 0 else 'down-value'}">{stats['lags'].get(15, {}).get('avg_return', 0)*100:.4f}% ({stats['lags'].get(15, {}).get('win_rate', 0)*100:.1f}%)</td>
                                <td>{stats['instances']}</td>
                            </tr>
                        """ for scenario, stats in directional_impact.items()])}
                        </tbody>
                    </table>
                </div>
                <div class="mt-3">
                    <small class="text-muted">Note: Values show average return % followed by win rate % in parentheses.</small>
                </div>
                <a href="../reports/btc_directional_impact_report.txt" class="btn btn-sm btn-outline-warning mt-2">View Full Directional Analysis</a>
            </div>
        </div>
        ''' if directional_impact else ''}
        
        <!-- Cross-Asset Relationship Insights -->
        {f'''
        <div class="card mb-4">
            <div class="card-header bg-success text-white">
                <h2 class="h4 mb-0">Cross-Asset Relationships</h2>
            </div>
            <div class="card-body">
                <p>How BTC patterns affect {altcoin_name.upper()} price movements:</p>
                <div class="table-responsive">
                    <table class="table table-sm table-hover">
                        <thead>
                            <tr>
                                <th>BTC Pattern</th>
                                <th>Best Lag (min)</th>
                                <th>Avg Return</th>
                                <th>Win Rate</th>
                                <th>Top Triggered Patterns</th>
                            </tr>
                        </thead>
                        <tbody>
                        {"".join([f"""
                            <tr>
                                <td>{pattern}</td>
                                <td>{max(stats.keys(), key=lambda k: stats[k]['avg_return']) if stats else 'N/A'}</td>
                                <td>{stats.get(max(stats.keys(), key=lambda k: stats[k]['avg_return']), {}).get('avg_return', 0)*100:.2f}%</td>
                                <td>{stats.get(max(stats.keys(), key=lambda k: stats[k]['avg_return']), {}).get('win_rate', 0)*100:.1f}%</td>
                                <td>{"N/A" if not stats else ', '.join([p[0] for p in stats.get(max(stats.keys(), key=lambda k: stats[k]['avg_return']), {}).get('top_patterns', [])])}</td>
                            </tr>
                        """ for pattern, stats in list(cross_asset_results.items())[:5] if stats])}
                        </tbody>
                    </table>
                </div>
                <a href="../reports/cross_asset_relationship_report.txt" class="btn btn-sm btn-outline-success">View Full Report</a>
            </div>
        </div>
        ''' if cross_asset_results else ''}
        
        <!-- Pattern Cards -->
        <h2>Detected Patterns</h2>
        <div class="row">
"""
    # Rest of the function remains the same...

def generate_master_index():
    """Generate a master index.html file in the results directory that links to all analysis runs."""
    import os
    from datetime import datetime
    import glob
    
    results_dir = "results"
    index_path = os.path.join(results_dir, 'index.html')
    
    # Make sure results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Find all date directories - use glob pattern for more reliable detection
    date_dirs = []
    pattern = os.path.join(results_dir, "??.*.*-??.*.*")
    for full_path in glob.glob(pattern):
        if os.path.isdir(full_path):
            dir_name = os.path.basename(full_path)
            date_dirs.append(dir_name)
    
    # Alternative method to find all directories that might be reports
    if not date_dirs:
        print("No directories matched the date pattern, scanning all directories...")
        for item in os.listdir(results_dir):
            full_path = os.path.join(results_dir, item)
            if os.path.isdir(full_path):
                # Check if it has html, charts, or reports subdirectories
                if (os.path.exists(os.path.join(full_path, 'html')) or
                    os.path.exists(os.path.join(full_path, 'charts')) or
                    os.path.exists(os.path.join(full_path, 'reports'))):
                    date_dirs.append(item)
    
    # Sort directories by date (newest first)
    try:
        date_dirs.sort(key=lambda x: datetime.strptime(x, '%d.%m.%Y-%H.%M.%S'), reverse=True)
    except:
        date_dirs.sort(reverse=True)
    
    print(f"Found {len(date_dirs)} report directories: {date_dirs[:5]}")
    
    # Create HTML content
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Trading AI Analysis Reports</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .card { margin-bottom: 20px; transition: all 0.3s; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .date-header { font-weight: bold; font-size: 1.2em; }
        .time-span { font-size:0.8em; }
        .badge-report { margin-right: 5px; margin-bottom: 5px; }
        #noResults { display: none; text-align: center; margin-top: 30px; }
        #reportStats { font-weight: bold; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="mb-4">Trading AI Analysis Reports</h1>
        
        <div class="alert alert-info">
            <p>All analysis reports are listed below. Use the search box to filter reports.</p>
        </div>
        
        <div class="row mb-4">
            <div class="col-12">
                <div class="input-group">
                    <input type="text" class="form-control" id="searchInput" placeholder="Search reports...">
                    <button class="btn btn-primary" id="searchButton" type="button">Search</button>
                    <button class="btn btn-secondary" id="clearButton" type="button">Clear</button>
                </div>
                <div id="reportStats" class="mt-2">
                    Showing all {len(date_dirs)} reports
                </div>
            </div>
        </div>
        
        <div id="noResults" class="alert alert-warning">
            <h4>No matching reports found</h4>
            <p>Try a different search term or clear the search</p>
        </div>
        
        <div class="row" id="reportsList">
"""
    
    # Add a card for each report
    for dir_name in date_dirs:
        try:
            # Format the date for display
            date_parts = dir_name.split('-')
            date_str = date_parts[0]
            time_str = date_parts[1].replace('.', ':') if len(date_parts) > 1 else ''
            
            # Try to parse the date more nicely
            try:
                date_obj = datetime.strptime(dir_name, '%d.%m.%Y-%H.%M.%S')
                nice_date = date_obj.strftime('%B %d, %Y')
                nice_time = date_obj.strftime('%I:%M %p')
            except:
                nice_date = date_str
                nice_time = time_str
            
            # Check what's actually available
            dir_path = os.path.join(results_dir, dir_name)
            
            # Find available HTML files
            has_main_index = os.path.exists(os.path.join(dir_path, 'index.html'))
            has_strategies = os.path.exists(os.path.join(dir_path, 'html', 'directional_strategies.html'))
            
            # Find chart files (don't just check if directory exists)
            charts = []
            charts_dir = os.path.join(dir_path, 'charts')
            if os.path.exists(charts_dir):
                charts = [f for f in os.listdir(charts_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(charts_dir, f))]
            
            # Find report files (don't just check if directory exists)
            report_files = []
            reports_dir = os.path.join(dir_path, 'reports')
            if os.path.exists(reports_dir):
                report_files = [f for f in os.listdir(reports_dir) 
                              if f.endswith(('.txt', '.json', '.csv')) and os.path.isfile(os.path.join(reports_dir, f))]
            
            html_content += f"""
            <div class="col-md-6 mb-4 report-card">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <div class="date-header">{nice_date} <span class="time-span">-at {nice_time}</span></div>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <span class="badge bg-secondary badge-report">Report #{len(date_dirs) - date_dirs.index(dir_name)}</span>
                                    <span class="badge bg-info badge-report">{dir_name}</span>
                                </div>
                            </div>
                        </div>
"""
            # Only show main dashboard button if index.html exists
            if has_main_index:
                html_content += f"""
                        <div class="d-grid gap-2">
                            <a href="{dir_name}/index.html" class="btn btn-primary mb-2">View Analysis Dashboard</a>
                        </div>
"""
            
            # Add buttons for HTML files that actually exist
            if has_strategies or charts:
                html_content += """
                        <div class="d-flex flex-wrap mt-2">
"""
                if has_strategies:
                    html_content += f"""
                            <a href="{dir_name}/html/directional_strategies.html" class="btn btn-sm btn-success me-2 mb-2">Trading Strategies</a>
"""
                # Create an actual charts index.html if there are charts
                if charts:
                    # Create charts index file on the fly
                    charts_index_path = os.path.join(charts_dir, 'index.html')
                    with open(charts_index_path, 'w') as charts_file:
                        charts_file.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Analysis Charts</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Analysis Charts</h1>
        <div class="mb-3">
            <a href="../index.html" class="btn btn-primary">Back to Dashboard</a>
        </div>
        <div class="row">
""")
                        for chart in sorted(charts):
                            nice_name = chart.replace('_', ' ').split('.')[0].title()
                            charts_file.write(f"""
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header">{nice_name}</div>
                    <div class="card-body p-0">
                        <img src="{chart}" class="img-fluid" alt="{nice_name}">
                    </div>
                </div>
            </div>
""")
                        charts_file.write("""
        </div>
    </div>
</body>
</html>
""")
                    html_content += f"""
                            <a href="{dir_name}/charts/index.html" class="btn btn-sm btn-info me-2 mb-2">View Charts ({len(charts)})</a>
"""
                html_content += """
                        </div>
"""
            
            # Add reports list if report files exist
            if report_files:
                html_content += """
                        <div class="mt-3">
                            <p class="mb-1"><strong>Available Reports:</strong></p>
                            <ul class="list-group">
"""
                for i, report in enumerate(sorted(report_files)[:8]):  # Show up to 8 reports
                    display_name = report.replace('_', ' ').title().split('.')[0]
                    html_content += f"""
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    {display_name}
                                    <a href="{dir_name}/reports/{report}" class="btn btn-sm btn-outline-primary" target="_blank">View</a>
                                </li>"""
                
                # Add "more reports" message if there are more reports
                if len(report_files) > 8:
                    # Create a simple report index file
                    report_index_path = os.path.join(reports_dir, 'index.html')
                    with open(report_index_path, 'w') as report_file:
                        report_file.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Analysis Reports</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Analysis Reports</h1>
        <div class="mb-3">
            <a href="../index.html" class="btn btn-primary">Back to Dashboard</a>
        </div>
        <div class="list-group">
""")
                        for report in sorted(report_files):
                            display_name = report.replace('_', ' ').title().split('.')[0]
                            report_file.write(f"""
            <a href="{report}" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                {display_name}
                <span class="badge bg-primary rounded-pill">{report.split('.')[-1].upper()}</span>
            </a>
""")
                        report_file.write("""
        </div>
    </div>
</body>
</html>
""")
                    html_content += f"""
                                <li class="list-group-item text-center">
                                    <a href="{dir_name}/reports/index.html" class="text-primary">View all {len(report_files)} reports</a>
                                </li>"""
                
                html_content += """
                            </ul>
                        </div>"""
            
            html_content += """
                    </div>
                </div>
            </div>
"""
        except Exception as e:
            html_content += f"""
            <div class="col-md-6 mb-4 report-card">
                <div class="card">
                    <div class="card-header bg-warning">
                        <div class="date-header">{dir_name}</div>
                    </div>
                    <div class="card-body">
                        <p>Error loading report details: {str(e)}</p>
                        <a href="{dir_name}/index.html" class="btn btn-primary">Try View Report</a>
                    </div>
                </div>
            </div>
"""
    
    # Complete the HTML with improved search functionality
    html_content += """
        </div>
    </div>
    
    <script>
    // Improved search functionality
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const clearButton = document.getElementById('clearButton');
    const cards = document.querySelectorAll('.report-card');
    const noResults = document.getElementById('noResults');
    const reportStats = document.getElementById('reportStats');
    
    function performSearch() {
        const input = searchInput.value.toLowerCase();
        let visibleCount = 0;
        
        cards.forEach(function(card) {
            const text = card.textContent.toLowerCase();
            if (text.includes(input)) {
                card.style.display = '';
                visibleCount++;
            } else {
                card.style.display = 'none';
            }
        });
        
        // Update stats and show no results message if needed
        reportStats.textContent = `Showing ${visibleCount} of ${cards.length} reports`;
        noResults.style.display = visibleCount === 0 ? 'block' : 'none';
    }
    
    function clearSearch() {
        searchInput.value = '';
        cards.forEach(card => card.style.display = '');
        reportStats.textContent = `Showing all ${cards.length} reports`;
        noResults.style.display = 'none';
    }
    
    searchButton.addEventListener('click', performSearch);
    clearButton.addEventListener('click', clearSearch);
    
    searchInput.addEventListener('keyup', function(event) {
        if (event.key === 'Enter') {
            performSearch();
        } else if (event.key === 'Escape') {
            clearSearch();
        }
    });
    </script>
</body>
</html>"""
    
    # Write the index file
    with open(index_path, 'w') as f:
        f.write(html_content)
        
    print(f"Generated master index at {index_path} with {len(date_dirs)} report directories")
    return index_path