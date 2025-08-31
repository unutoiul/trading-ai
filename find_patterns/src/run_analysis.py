"""Main script to run the BTC-DOGE pattern analysis and serve results."""

import json
import os
import argparse
import shutil
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
import importlib
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Add the current directory to path if not already there
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import from src modules with explicit relative imports
from src import ml_models
from src import strategy_optimizer
from src import pattern_analysis
from src.data_processing import load_and_preprocess_data
from src.pattern_analysis import classify_momentum_patterns, analyze_lag_relationships, analyze_cross_asset_relationships
from src.ml_models import analyze_directional_impact_ml
from src.visualization import (
    create_results_directory,
    generate_cross_asset_report,
    generate_index_html,
    update_index_html
)
# Fixed import for server module
from src.server import start_server
from src.config import RESULTS_DIR, MAX_LAG
from src.strategy_generator import DirectionalImpactStrategies
from src.feature_engineering import (
    add_enhanced_momentum_features,
    add_price_action_features,
    add_relationship_features,
    add_composite_features
)
from src.conditions_filter import ConditionsFilter

# Update the ML report generation
def generate_ml_report(ml_results, output_dir):
    """Generate analysis of pattern-return relationships using price action insights."""
    output_file = os.path.join(output_dir, 'price_action_analysis_report.txt')
    
    with open(output_file, 'w') as f:
        f.write("PRICE ACTION ANALYSIS\n")
        f.write("=======================\n\n")
        
        # Check if feature importance exists in results
        if 'feature_importance' in ml_results:
            # Feature importance analysis
            f.write("PATTERN IMPORTANCE RANKING\n")
            f.write("Which BTC patterns most strongly influence altcoin returns:\n\n")
            for feature, importance in ml_results['feature_importance']:
                f.write(f"{feature}: {importance:.6f}\n")
        else:
            # Handle price action scenarios
            f.write("PRICE ACTION SCENARIOS\n")
            f.write("How different BTC movements influence altcoin price:\n\n")
            
            # List all scenarios if available
            scenarios = [s for s in ml_results.keys() if s.startswith('btc_')]
            for scenario in scenarios:
                f.write(f"\n{scenario.upper()}\n")
                if isinstance(ml_results[scenario], dict):
                    for lag, stats in sorted(ml_results[scenario].items()):
                        if isinstance(stats, dict) and 'mean_return' in stats:
                            f.write(f"  Lag {lag}: Return = {stats['mean_return']:.4f}%, "
                                    f"Win Rate = {stats['win_rate']*100 if 'win_rate' in stats else 0:.1f}%\n")
        
        f.write("\n\nINTERPRETATION\n")
        f.write("The analysis shows which BTC price action patterns have the strongest\n")
        f.write("relationship with future altcoin price movements.\n\n")
        
        # Include performance metrics if available
        if 'metrics' in ml_results:
            f.write("\nPERFORMANCE METRICS\n")
            metrics = ml_results['metrics']
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
                else:
                    f.write(f"{metric_name}: {metric_value}\n")
        
    print(f"Saved price action analysis report to {output_file}")
    return output_file

def ensure_html_reports_exist(results_dirs, pattern_stats, altcoin_name):
    """Make sure all pattern HTML reports exist, even if visualization functions failed."""
    html_dir = results_dirs['html']
    os.makedirs(html_dir, exist_ok=True)
    
    # Create placeholder HTML reports for each pattern
    for pattern in pattern_stats.keys():
        if pattern == 'no_patterns_detected':
            continue
            
        pattern_file = os.path.join(html_dir, f"{pattern}_analysis.html")
        if not os.path.exists(pattern_file):
            print(f"Creating placeholder HTML for {pattern}")
            with open(pattern_file, 'w') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{pattern.replace('_', ' ').title()} Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>{pattern.replace('_', ' ').title()} Pattern Analysis</h1>
        <div class="alert alert-info mt-4">
            <p>Pattern statistics:</p>
            <ul>
                <li>Instances: {pattern_stats[pattern].get('instances', 0)}</li>
                <li>Optimal Lag: {pattern_stats[pattern].get('optimal_lag', 0)} minutes</li>
                <li>Win Rate: {pattern_stats[pattern].get('win_rate', 0)*100:.1f}%</li>
            </ul>
        </div>
        <a href="./btc_{altcoin_name.lower()}_pattern_analysis.html" class="btn btn-primary">Back to Pattern Analysis</a>
    </div>
</body>
</html>""")

def apply_condition_filters(combined_data, conditions=None):
    """Apply condition filters to data for segmentation."""
    print("\nApplying condition filters for data segmentation...")
    
    # Create condition filter
    filter_manager = ConditionsFilter(combined_data)
    
    # Define standard conditions
    available_conditions = filter_manager.define_standard_conditions()
    
    # Apply filters if provided
    if conditions:
        filtered_data = filter_manager.apply_filters(conditions)
        print(f"Applied filters: {conditions}")
    else:
        filtered_data = combined_data
        print("No filters applied, using complete dataset")
    
    # Report on filtered data size
    print(f"Filtered dataset size: {len(filtered_data)} rows")
    
    # Return both the filtered data and the filter manager for further use
    return filtered_data, filter_manager

def run_directional_analysis_only(btc_file, alt_file, conditions=None, condition_type="any"):
    """Run only the directional impact ML analysis and strategy generation."""
    print("\n=== RUNNING DIRECTIONAL IMPACT ANALYSIS ONLY ===")
    
    # Step 1: Create results directory structure
    results_dirs = create_results_directory(RESULTS_DIR)
    
    # Step 2: Load and preprocess data (essential step)
    combined_data = load_and_preprocess_data(btc_file, alt_file)
    
    # Extract altcoin name from combined_data
    altcoin_name = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            altcoin_name = col.split('_')[0]
            break
    
    if not altcoin_name and 'altcoin_name' in combined_data.columns:
        altcoin_name = combined_data['altcoin_name'].iloc[0]
    
    if not altcoin_name:
        altcoin_name = "altcoin"
    
    print(f"\nRunning Directional Impact Analysis for BTC and {altcoin_name.upper()}")
    
    # Fix column naming inconsistencies - this is crucial!
    # Ensure all required columns exist with consistent naming
    column_mappings = {
        # Map price columns
        'close_btc': 'btc_close',
        'high_btc': 'btc_high',
        'low_btc': 'btc_low',
        'open_btc': 'btc_open',
        'volume_btc': 'btc_volume',
        
        # Map altcoin columns
        f'close_{altcoin_name}': f'{altcoin_name}_close',
        f'high_{altcoin_name}': f'{altcoin_name}_high',
        f'low_{altcoin_name}': f'{altcoin_name}_low',
        f'open_{altcoin_name}': f'{altcoin_name}_open',
        f'volume_{altcoin_name}': f'{altcoin_name}_volume'
    }
    
    # Apply all column mappings if source exists and target doesn't
    print("Standardizing column names...")
    for source, target in column_mappings.items():
        if source in combined_data.columns and target not in combined_data.columns:
            combined_data[target] = combined_data[source]
            print(f"  Mapped '{source}' to '{target}'")
    
    # Add enhanced features to data - DON'T reassign the DataFrame!
    print("Adding enhanced momentum features...")
    add_enhanced_momentum_features(combined_data, prefix='btc')
    
    print("Adding price action features...")
    add_price_action_features(combined_data, prefix='btc')
    
    print("Adding relationship features...")
    add_relationship_features(combined_data, btc_prefix='btc', alt_prefix=altcoin_name)
    
    print("Adding composite features...")
    add_composite_features(combined_data, btc_prefix='btc', alt_prefix=altcoin_name)
    
    # Apply condition filters if provided
    if conditions:
        filtered_data, filter_manager = apply_condition_filters(combined_data, conditions)
        # Save the filter configuration
        filter_report = os.path.join(results_dirs['reports'], 'filter_config.json')
        with open(filter_report, 'w') as f:
            json.dump({
                'conditions': conditions,
                'condition_type': condition_type,
                'filtered_size': len(filtered_data),
                'original_size': len(combined_data),
                'percentage': len(filtered_data) / len(combined_data) * 100
            }, f, indent=2)
    else:
        # If no conditions provided, use all data
        filtered_data = combined_data
        print("No filters applied, using complete dataset")
    
    # Step 3: Run directional impact ML analysis
    print("\nRunning directional impact ML analysis...")
    try:
        # First, analyze BTC directional impact on altcoin
        print("Analyzing BTC directional impact on altcoin...")
        directional_impact = pattern_analysis.analyze_btc_directional_impact(filtered_data)
        
        # Now use the directional impact results in ML analysis
        print("Running ML analysis with directional impact data...")
        directional_impact_results = analyze_directional_impact_ml(
            filtered_data, 
            directional_impact=directional_impact,
            results_dirs=results_dirs
        )
        
        # Step 4: Generate trading strategies based on directional impact
        print("\nGenerating trading strategies...")
        strategy_generator = DirectionalImpactStrategies(
            filtered_data, 
            directional_impact_results,
            altcoin_name
        )
        
        # Generate strategies
        strategy_generator.generate_strategies()
        
        # Backtest strategies
        strategy_generator.backtest_strategies()
        
        # Generate reports
        strategy_generator.generate_strategy_reports(results_dirs)
        
        print("\nDirectional analysis and strategy generation complete!")
        print(f"Results saved to: {results_dirs['base']}")
        
    except Exception as e:
        print(f"Error in directional analysis: {str(e)}")
        traceback.print_exc()
        print("Continuing with basic analysis...")
    
    # Generate index.html
    try:
        update_index_html(results_dirs, altcoin_name)
    except Exception as e:
        print(f"Error generating index HTML: {str(e)}")
    
    # Return the results directory
    return results_dirs['base']

def run_price_action_optimization(btc_file, alt_file, optimize_count=100):
    """Run price action optimization process to find the best strategy parameters."""
    print("\n=== RUNNING PRICE ACTION STRATEGY OPTIMIZATION ===")
    
    # Step 1: Create results directory structure
    results_dirs = create_results_directory(RESULTS_DIR)
    
    # Step 2: Load and preprocess data
    combined_data = load_and_preprocess_data(btc_file, alt_file)
    
    # Extract altcoin name
    altcoin_name = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            altcoin_name = col.split('_')[0]
            break
    
    if not altcoin_name and 'altcoin_name' in combined_data.columns:
        altcoin_name = combined_data['altcoin_name'].iloc[0]
    
    if not altcoin_name:
        altcoin_name = "altcoin"
    
    print(f"\nRunning Price Action Strategy Optimization for BTC and {altcoin_name.upper()}")
    
    # Add price action features
    print("Adding price action features...")
    add_price_action_features(combined_data, prefix='btc')
    add_price_action_features(combined_data, prefix=altcoin_name)
    add_enhanced_momentum_features(combined_data, prefix='btc')
    add_relationship_features(combined_data, btc_prefix='btc', alt_prefix=altcoin_name)
    
    # Step 3: Identify patterns using price action only
    print("\nIdentifying price action patterns...")
    patterns = classify_momentum_patterns(combined_data)
    
    # Step 4: Run strategy optimization
    print("\nRunning strategy optimization...")
    optimizer = strategy_optimizer.StrategyOptimizer(combined_data, patterns)
    
    # Run optimization with focus on price action
    optimization_results = optimizer.ml_optimization(price_action_focus=True, optimize_count=optimize_count)
    
    # Generate HTML report
    html_output = os.path.join(results_dirs['html'], 'strategy_optimization_results.html')
    optimizer.generate_html_report(html_output, results_dirs['charts'])
    
    # Update index HTML
    try:
        update_index_html(results_dirs, altcoin_name)
    except Exception as e:
        print(f"Error updating index HTML: {str(e)}")
    
    # Return results directory
    print(f"\nStrategy optimization complete! Results saved to: {results_dirs['base']}")
    return results_dirs['base']

def main(btc_file=None, alt_file=None, use_ml=True, optimize_strategy=True, serve=False, port=8000, 
         return_results_dir=False, conditions=None, condition_type='any'):
    """
    Main entry point that can be called directly or via command line.
    
    Args:
        btc_file: Path to BTC CSV file
        alt_file: Path to altcoin CSV file
        use_ml: Whether to use ML models
        optimize_strategy: Whether to optimize strategy parameters
        serve: Whether to start a web server
        port: Port for web server
        return_results_dir: Whether to return results directory
        conditions: List of market conditions to filter data by
        condition_type: 'any' or 'all' - whether any or all conditions must be met
    """
    # If called from command line, parse arguments
    if btc_file is None or alt_file is None:
        parser = argparse.ArgumentParser(description="Analyze cryptocurrency patterns")
        parser.add_argument('--btc', required=True, help='Path to BTC data CSV')
        parser.add_argument('--alt', required=True, help='Path to altcoin data CSV')
        parser.add_argument('--use-ml', action='store_true', help='Use ML models')
        parser.add_argument('--optimize-strategy', action='store_true', default=True, help='Optimize strategy parameters')
        parser.add_argument('--serve', action='store_true', help='Start web server')
        parser.add_argument('--port', type=int, default=8000, help='Port for web server')
        parser.add_argument('--condition', action='append', help='Market conditions to filter by (can be used multiple times)')
        parser.add_argument('--condition-type', choices=['any', 'all'], default='any', help='Whether any or all conditions must be met')
        args = parser.parse_args()
        
        btc_file = args.btc
        alt_file = args.alt
        use_ml = args.use_ml
        optimize_strategy = args.optimize_strategy
        serve = args.serve
        port = args.port
        conditions = args.condition
        condition_type = args.condition_type
    
    # Step 1: Create results directory structure - make sure it exists
    results_dirs = create_results_directory(RESULTS_DIR)
    
    # Step 2: Load and preprocess data
    combined_data = load_and_preprocess_data(btc_file, alt_file)
    
    # Extract altcoin name
    altcoin_name = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            altcoin_name = col.split('_')[0]
            break
            
    if not altcoin_name and 'altcoin_name' in combined_data.columns:
        altcoin_name = combined_data['altcoin_name'].iloc[0]
        
    if not altcoin_name:
        altcoin_name = "altcoin"
        
    print(f"Running analysis for BTC and {altcoin_name.upper()}")
    
    # Fix column naming inconsistencies - this is crucial!
    # Ensure all required columns exist with consistent naming
    column_mappings = {
        # Map price columns
        'close_btc': 'btc_close',
        'high_btc': 'btc_high',
        'low_btc': 'btc_low',
        'open_btc': 'btc_open',
        'volume_btc': 'btc_volume',
        
        # Map altcoin columns
        f'close_{altcoin_name}': f'{altcoin_name}_close',
        f'high_{altcoin_name}': f'{altcoin_name}_high',
        f'low_{altcoin_name}': f'{altcoin_name}_low',
        f'open_{altcoin_name}': f'{altcoin_name}_open',
        f'volume_{altcoin_name}': f'{altcoin_name}_volume'
    }
    
    # Apply all column mappings if source exists and target doesn't
    print("Standardizing column names...")
    for source, target in column_mappings.items():
        if source in combined_data.columns and target not in combined_data.columns:
            combined_data[target] = combined_data[source]
            print(f"  Mapped '{source}' to '{target}'")
            
    # Add enhanced features to data - DON'T reassign the DataFrame!
    print("Adding enhanced momentum features...")
    add_enhanced_momentum_features(combined_data, prefix='btc')
    
    # Apply condition filters if requested
    if conditions:
        filtered_data, filter_manager = apply_condition_filters(combined_data, conditions)
        # Save the filter configuration
        filter_report = os.path.join(results_dirs['reports'], 'filter_config.json')
        with open(filter_report, 'w') as f:
            json.dump({
                'conditions': conditions,
                'condition_type': condition_type,
                'filtered_size': len(filtered_data),
                'original_size': len(combined_data),
                'percentage': len(filtered_data) / len(combined_data) * 100
            }, f, indent=2)
    else:
        filtered_data = combined_data
        filter_manager = None
    
    # Step 3: Identify patterns
    print("\nIdentifying momentum patterns...")
    patterns = classify_momentum_patterns(filtered_data)
    filtered_data = pd.concat([filtered_data, patterns], axis=1)
    
    # Step 4: Pattern analysis
    print("\nAnalyzing lag relationships between patterns...")
    pattern_stats = analyze_lag_relationships(filtered_data, patterns, max_lag_seconds=600, lag_step_seconds=10)
    
    # Step 5: ML analysis (optional)
    ml_results = None
    if use_ml:
        try:
            print("\nApplying machine learning to pattern analysis...")
            # Additional price action features for ML
            add_price_action_features(filtered_data, prefix='btc')
            add_relationship_features(filtered_data, btc_prefix='btc', alt_prefix=altcoin_name)
            
            # Run directional impact analysis
            directional_impact = pattern_analysis.analyze_btc_directional_impact(filtered_data)
            
            ml_results = ml_models.analyze_directional_impact_ml(
                filtered_data,
                directional_impact=directional_impact,
                results_dirs=results_dirs
            )
            
            # Generate ML report
            generate_ml_report(ml_results, results_dirs['reports'])
            
        except Exception as e:
            print(f"Error in ML analysis: {str(e)}")
            traceback.print_exc()
            print("Continuing with pattern analysis only...")
    
    # Step 6: Strategy optimization (optional)
    if optimize_strategy:
        try:
            print("\nOptimizing trading strategy parameters...")
            optimizer = strategy_optimizer.StrategyOptimizer(filtered_data, pattern_stats)
            
            # Run ML optimization if ML results are available
            if ml_results:
                optimizer.ml_optimization()
            else:
                # Otherwise run standard backtest with default params
                test_params = {
                    'trailing_stop': 0.02,
                    'take_profit': 0.03,
                    'stop_loss': 0.015,
                    'entry_threshold': 0.7,
                    'position_size': 0.1
                }
                optimizer.backtest_strategy(test_params)
            
            # Generate charts and reports
            optimizer.generate_strategy_charts(results_dirs['charts'])
            optimizer.generate_html_report(
                os.path.join(results_dirs['html'], f'strategy_optimization_{altcoin_name}.html'),
                results_dirs['charts']
            )
            
        except Exception as e:
            print(f"Error in strategy optimization: {str(e)}")
            traceback.print_exc()
            print("Continuing without strategy optimization...")
    
    # Step 7: Generate visual reports
    try:
        # Generate cross-asset report
        cross_asset_results = analyze_cross_asset_relationships(filtered_data, patterns)
        generate_cross_asset_report(cross_asset_results, results_dirs['html'], altcoin_name)
        
        # Make sure all HTML reports are created
        ensure_html_reports_exist(results_dirs, pattern_stats, altcoin_name)
        
        # Create main index.html
        generate_index_html(results_dirs, altcoin_name)
        
    except Exception as e:
        print(f"Error generating reports: {str(e)}")
        traceback.print_exc()
    
    # Step 8: Serve results (optional)
    if serve:
        print(f"\nStarting web server on port {port}...")
        start_server(results_dirs['base'], port=port)
    
    print(f"\nAnalysis complete! Results saved to: {results_dirs['base']}")
    
    # Return results directory if requested (for API usage)
    if return_results_dir:
        return results_dirs['base']