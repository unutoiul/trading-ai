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
from src.ml_models import evaluate_model, prepare_features_targets, train_xgboost_model, plot_feature_importance, analyze_directional_impact_ml
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

# Update the ML report generation
def generate_ml_report(ml_results, output_dir):
    """Generate analysis of pattern-return relationships using XGBoost insights."""
    output_file = os.path.join(output_dir, 'xgboost_analysis_report.txt')
    
    with open(output_file, 'w') as f:
        f.write("XGBOOST FEATURE ANALYSIS\n")
        f.write("=======================\n\n")
        
        # Feature importance analysis
        f.write("PATTERN IMPORTANCE RANKING\n")
        f.write("Which BTC patterns most strongly influence DOGE returns:\n\n")
        for feature, importance in ml_results['feature_importance']:
            f.write(f"{feature}: {importance:.6f}\n")
        
        f.write("\n\nINTERPRETATION\n")
        f.write("The feature importance shows which BTC patterns and indicators have the strongest\n")
        f.write("relationship with future DOGE price movements. Higher values indicate stronger influence.\n\n")
        
        # Still include model performance metrics but de-emphasize them
        f.write("\nMODEL VALIDATION METRICS\n")
        f.write("(Used to verify the reliability of the feature importance analysis)\n")
        f.write(f"Directional Accuracy: {ml_results['metrics']['directional_accuracy']:.2%}\n")
        f.write(f"R² Score: {ml_results['metrics']['r2']:.4f}\n")
        
    print(f"Saved XGBoost analysis report to {output_file}")
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

def run_directional_analysis_only(btc_file, alt_file):
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
    
    if not altcoin_name:
        altcoin_name = "altcoin"
    
    print(f"\nRunning Directional Impact Analysis for BTC and {altcoin_name.upper()}")
    
    # Add BTC directional impact scenarios to the data
    for scenario in ['btc_up_strong', 'btc_up_medium', 'btc_up_small',
                    'btc_down_small', 'btc_down_medium', 'btc_down_strong']:
        combined_data[scenario] = False
    
    min_price_move = 0.0015  # 0.15% move threshold
    
    # Define the different BTC move scenarios directly on the combined_data
    combined_data['btc_up_strong'] = (combined_data['btc_returns'] > min_price_move)
    combined_data['btc_up_medium'] = (combined_data['btc_returns'] > min_price_move/2) & (combined_data['btc_returns'] <= min_price_move)
    combined_data['btc_up_small'] = (combined_data['btc_returns'] > 0) & (combined_data['btc_returns'] <= min_price_move/2)
    combined_data['btc_down_small'] = (combined_data['btc_returns'] < 0) & (combined_data['btc_returns'] >= -min_price_move/2)
    combined_data['btc_down_medium'] = (combined_data['btc_returns'] < -min_price_move/2) & (combined_data['btc_returns'] >= -min_price_move)
    combined_data['btc_down_strong'] = (combined_data['btc_returns'] < -min_price_move)
    
    # Step 3: Run directional impact analysis
    print("\nAnalyzing BTC directional impact on altcoin...")
    directional_impact = pattern_analysis.analyze_btc_directional_impact(
        combined_data,
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30],
        min_price_move=min_price_move
    )
    
    # Generate directional impact report
    if directional_impact:
        output_file = os.path.join(results_dirs['reports'], 'btc_directional_impact_report.txt')
        with open(output_file, 'w') as f:
            f.write(f"BTC DIRECTIONAL IMPACT ON {altcoin_name.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            for scenario, stats in directional_impact.items():
                f.write(f"\n{scenario.upper()}\n")
                f.write("-" * len(scenario) + "\n")
                f.write(f"Total instances: {stats['instances']}\n\n")
                
                # Create a table of lag results
                f.write(f"{'Lag (min)':<10}{'Avg Return %':<15}{'Win Rate %':<15}{'Sample Size':<10}\n")
                f.write("-" * 50 + "\n")
                
                # Sort lags by minute value
                for lag in sorted(stats['lags'].keys()):
                    lag_stats = stats['lags'][lag]
                    f.write(f"{lag:<10}{lag_stats['avg_return']*100:<15.4f}{lag_stats['win_rate']*100:<15.1f}{lag_stats['sample_size']:<10}\n")
                
                f.write("\n")
        print(f"Generated directional impact report: {output_file}")

    # Step 4: Apply ML to analyze directional impact
    print("\nApplying machine learning to directional impact analysis...")
    try:
        directional_impact_ml = ml_models.analyze_directional_impact_ml(
            combined_data,
            directional_impact
        )
        
        # Save ML analysis results
        ml_report_path = os.path.join(results_dirs['reports'], 'directional_impact_ml_report.txt')
        with open(ml_report_path, 'w') as f:
            f.write(f"ML ANALYSIS OF BTC-{altcoin_name.upper()} DIRECTIONAL IMPACT\n")
            f.write("=" * 50 + "\n\n")
            
            for scenario, lag_results in directional_impact_ml.items():
                f.write(f"\n{scenario.upper()}\n")
                f.write("-" * len(scenario) + "\n\n")
                
                for lag, results in lag_results.items():
                    f.write(f"Lag: {lag} minutes\n")
                    f.write(f"Prediction accuracy: {results['accuracy']*100:.1f}%\n\n")
                    
                    f.write("Top predictive factors:\n")
                    for feature, importance in results['feature_importance'][:7]:
                        f.write(f"  {feature}: {importance*100:.2f}%\n")
                    
                    f.write("\n")
        
        print(f"Generated ML directional impact report: {ml_report_path}")
        
        # Step 5: Generate and backtest trading strategies
        try:
            print("\nGenerating trading strategies from directional impact analysis...")
            from src.strategy_generator import DirectionalImpactStrategies
            
            strategy_generator = DirectionalImpactStrategies(
                combined_data, 
                directional_impact_ml, 
                altcoin_name
            )

            # Generate strategies
            strategies = strategy_generator.generate_strategies()

            # Backtest strategies
            backtest_results = strategy_generator.backtest_strategies()

            # Generate reports
            strategy_reports = strategy_generator.generate_strategy_reports(results_dirs)

            print(f"\nGenerated strategy reports at {strategy_reports['txt_report']}")
            print(f"Generated HTML strategy dashboard at {strategy_reports['html_report']}")
            update_index_html(results_dirs, altcoin_name)

            # Generate/update master index of all reports
            from src.visualization import generate_master_index
            generate_master_index()

        except Exception as e:
            print(f"Error in strategy generation: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error in ML directional impact analysis: {e}")
        traceback.print_exc()
    
    print("\n======= DIRECTIONAL ANALYSIS COMPLETE =======")
    print(f"Results saved to: {results_dirs['base']}")
    print(f"Main dashboard: {os.path.join(results_dirs['base'], 'index.html')}")
    
    return results_dirs['base']

def main(btc_file=None, alt_file=None, use_ml=True, optimize_strategy=True, serve=False, port=8000, return_results_dir=False):
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
        args = parser.parse_args()
        
        btc_file = args.btc
        alt_file = args.alt
        use_ml = args.use_ml
        optimize_strategy = args.optimize_strategy
        serve = args.serve
        port = args.port
    
    # Step 1: Create results directory structure - make sure it exists
    results_dirs = create_results_directory(RESULTS_DIR)
    
    # Step 2: Load and preprocess data
    combined_data = load_and_preprocess_data(btc_file, alt_file)
    
    # Add enhanced features to data
    print("Adding enhanced momentum features...")
    btc_momentum_features = add_enhanced_momentum_features(combined_data, prefix='btc')
    alt_momentum_features = add_enhanced_momentum_features(combined_data, prefix=altcoin_name)

    print("Adding price action features...")
    btc_price_features = add_price_action_features(combined_data, prefix='btc')
    alt_price_features = add_price_action_features(combined_data, prefix=altcoin_name)

    print("Adding relationship features...")
    relationship_features = add_relationship_features(combined_data, btc_prefix='btc', alt_prefix=altcoin_name)

    print("Adding composite features...")
    composite_features = add_composite_features(combined_data, btc_prefix='btc', alt_prefix=altcoin_name)

    # Log the number of features added
    total_features = len(btc_momentum_features) + len(alt_momentum_features) + \
                    len(btc_price_features) + len(alt_price_features) + \
                    len(relationship_features) + len(composite_features)
    print(f"Added {total_features} new features to enhance pattern detection")
    
    # Extract altcoin name from combined_data directly
    altcoin_name = getattr(combined_data, 'altcoin_name', None)

    # With this correct approach:
    if 'altcoin_name' in combined_data.columns:
        # Get the first non-null value from the column
        altcoin_name = combined_data['altcoin_name'].iloc[0] if not combined_data['altcoin_name'].isna().all() else None
    else:
        altcoin_name = None
        
    if not altcoin_name:
        print("WARNING: Could not detect altcoin name from data. Using default.")
        return

    print(f"\nAnalyzing relationship between BTC and {altcoin_name.upper()}")

    # Make sure the altcoin name is available to all analysis functions
    combined_data['altcoin_name'] = altcoin_name
    
    # Step 3: Identify patterns - use our enhanced pattern detection
    patterns = classify_momentum_patterns(combined_data)
    combined_data = pd.concat([combined_data, patterns], axis=1)
    
    # Step 4: Traditional pattern analysis - use both old and new analysis
    pattern_stats = analyze_lag_relationships(combined_data, patterns, max_lag_seconds=600, lag_step_seconds=10)
    
    # New: Add cross-asset relationship analysis
    print("\nAnalyzing cross-asset relationships...")
    cross_asset_results = analyze_cross_asset_relationships(
        combined_data, 
        patterns,
        max_lag_minutes=20,
        lag_step_minutes=1
    )

    # Generate cross-asset report
    if cross_asset_results:
        print("\nGenerating cross-asset relationship report...")
        cross_asset_report = generate_cross_asset_report(
            cross_asset_results, 
            results_dirs['reports'], 
            altcoin_name
        )
        print(f"Cross-asset report generated: {cross_asset_report}")
    
    # Add directional impact analysis
    print("\nAnalyzing BTC directional impact on altcoin...")
    directional_impact = pattern_analysis.analyze_btc_directional_impact(
        combined_data,
        lags=[1, 2, 3, 5, 10, 15, 20, 30],
        min_price_move=0.0015  # 0.15% move is significant
    )

    # Generate directional impact report
    directional_impact_report = None
    if directional_impact:
        output_file = os.path.join(results_dirs['reports'], 'btc_directional_impact_report.txt')
        with open(output_file, 'w') as f:
            f.write(f"BTC DIRECTIONAL IMPACT ON {altcoin_name.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            for scenario, stats in directional_impact.items():
                f.write(f"\n{scenario.upper()}\n")
                f.write("-" * len(scenario) + "\n")
                f.write(f"Total instances: {stats['instances']}\n\n")
                
                # Create a table of lag results
                f.write(f"{'Lag (min)':<10}{'Avg Return %':<15}{'Win Rate %':<15}{'Sample Size':<10}\n")
                f.write("-" * 50 + "\n")
                
                # Sort lags by minute value
                for lag in sorted(stats['lags'].keys()):
                    lag_stats = stats['lags'][lag]
                    f.write(f"{lag:<10}{lag_stats['avg_return']*100:<15.4f}{lag_stats['win_rate']*100:<15.1f}{lag_stats['sample_size']:<10}\n")
                
                f.write("\n")
        directional_impact_report = output_file
        print(f"Generated directional impact report: {directional_impact_report}")
    
    # Apply ML to analyze directional impact
    print("\nApplying machine learning to directional impact analysis...")
    directional_impact_ml = analyze_directional_impact_ml(
        combined_data,
        directional_impact
    )

    # Save ML analysis results
    if directional_impact_ml:
        ml_report_path = os.path.join(results_dirs['reports'], 'directional_impact_ml_report.txt')
        with open(ml_report_path, 'w') as f:
            f.write(f"ML ANALYSIS OF BTC-{altcoin_name.upper()} DIRECTIONAL IMPACT\n")
            f.write("=" * 50 + "\n\n")
            
            for scenario, lag_results in directional_impact_ml.items():
                f.write(f"\n{scenario.upper()}\n")
                f.write("-" * len(scenario) + "\n\n")
                
                for lag, results in lag_results.items():
                    f.write(f"Lag: {lag} minutes\n")
                    f.write(f"Prediction accuracy: {results['accuracy']*100:.1f}%\n\n")
                    
                    f.write("Top predictive factors:\n")
                    for feature, importance in results['feature_importance'][:7]:
                        f.write(f"  {feature}: {importance*100:.2f}%\n")
                    
                    f.write("\n")
        
        print(f"Generated ML directional impact report: {ml_report_path}")
    
    # Generate and backtest trading strategies based on directional impact ML
    print("\nGenerating trading strategies from directional impact analysis...")
    strategy_generator = DirectionalImpactStrategies(
        combined_data, 
        directional_impact_ml, 
        altcoin_name
    )

    # Generate strategies
    strategies = strategy_generator.generate_strategies()

    # Backtest strategies
    backtest_results = strategy_generator.backtest_strategies()

    # Generate reports
    strategy_reports = strategy_generator.generate_strategy_reports(results_dirs)

    print(f"\nGenerated strategy reports at {strategy_reports['txt_report']}")
    print(f"Generated HTML strategy dashboard at {strategy_reports['html_report']}")

    # Add link to home page
    with open(os.path.join(results_dirs['html'], 'index.html'), 'r') as f:
        index_content = f.read()

    # Add strategy link if not already present
    if 'directional_strategies.html' not in index_content:
        new_link = """<div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-info text-white">
                            <h5 class="mb-0">Directional Trading Strategies</h5>
                        </div>
                        <div class="card-body">
                            <p>View ML-based strategies from directional impact analysis</p>
                            <a href="directional_strategies.html" class="btn btn-info">View Strategies</a>
                        </div>
                    </div>
                </div>"""
                
        # Insert after the pattern analysis card
        index_content = index_content.replace('</div>\n            \n            <div class="col-md-6">', 
                                             f'</div>\n            \n            {new_link}\n            \n            <div class="col-md-6">')
        
        # Write updated index
        with open(os.path.join(results_dirs['html'], 'index.html'), 'w') as f:
            f.write(index_content)

    # Step 5: Machine learning analysis (if enabled)
    ml_results = None
    if use_ml:
        print("\nRunning XGBoost analysis...")

        # First detect altcoin name to use as target column
        print("\nDetecting altcoin for ML target...")
        target_col = None
        # Fix: Explicitly iterate over column names to avoid Series comparison
        for col in [c for c in combined_data.columns]:
            if col.endswith('_returns') and not col.startswith('btc'):
                target_col = col
                break

        if not target_col:
            print("WARNING: Could not detect altcoin returns column. Using default.")
            target_col = 'alt_returns'
        else:
            print(f"Using {target_col} as ML target")
        
        # Prepare features and target with the detected column
        X, y, feature_names = prepare_features_targets(
            combined_data, 
            target_col=target_col,  # Now dynamic, not hardcoded to 'doge_returns'
            lag_periods=range(1, MAX_LAG+1)
        )
        
        # Split data chronologically (respect time series nature)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Train XGBoost model
        model = train_xgboost_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Plot feature importance
        feature_importance = plot_feature_importance(model, feature_names, results_dirs['charts'])
        
        
        # Generate ML report
        ml_report = generate_ml_report({
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': feature_names
        }, results_dirs['reports'])
        
        # Store ML results for including in other reports
        ml_results = {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'report_path': ml_report
        }
    
    # Step 6: First detect altcoin name to use in all reports
    print("\nDetecting altcoin name from data columns...")
    altcoin_name = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            altcoin_name = col.split('_')[0]
            break
    
    if not altcoin_name:
        altcoin_name = "alt"  # Default fallback
    
    print(f"Detected altcoin: {altcoin_name.upper()}")
    
    # Now that we have the altcoin name, ensure all required charts exist
    print("\nVerifying chart files exist...")
   
    
    # Ensure HTML reports exist even if visualization functions failed
    ensure_html_reports_exist(results_dirs, pattern_stats, altcoin_name)
    
    # Step 6b: Run Strategy Optimization if enabled
    if optimize_strategy:
        print("\nOptimizing trading strategy parameters...")
        try:
            print(f"\nRunning strategy optimization for {altcoin_name}...")
            strategy_results = pattern_analysis.optimize_strategy_parameters(
                combined_data, patterns, max_lag=20, altcoin_name=altcoin_name,
                results_dirs=results_dirs  # Pass the directories explicitly
            )
            
            if strategy_results and 'best_params' in strategy_results and strategy_results['best_params']:
                print(f"Strategy optimization successful. Win rate: {strategy_results['performance_metrics']['win_rate']*100:.1f}%")
            else:
                print("Strategy optimization failed or returned no results. Using default parameters.")
                # Create default parameters
                strategy_results = {
                    'best_params': {
                        'use_pattern': patterns.columns[0] if not patterns.empty and len(patterns.columns) > 0 else 'strong_up',
                        'pattern_lag': 5,
                        'stop_loss_pct': 2.0,
                        'take_profit_pct': 1.0,
                        'max_holding_time': 60,
                        'position_size_pct': 10,
                        'entry_threshold': 0.0001
                    },
                    'performance_metrics': {
                        'win_rate': 0.5,
                        'total_trades': 0,
                        'profit_factor': 1.0,
                        'sharpe_ratio': 0,
                        'max_drawdown': 0,
                        'total_return_pct': 0
                    },
                    'strategy_summary': "Default strategy (no optimization performed)",
                }
            
            # Save strategy parameters JSON
            strategy_output = os.path.join(results_dirs['reports'], 'strategy_params.json')
            with open(strategy_output, 'w') as f:
                # Convert non-serializable objects to strings
                serializable_results = {
                    'best_params': strategy_results.get('best_params', {}),
                    'performance_summary': {
                        'total_trades': strategy_results.get('performance_metrics', {}).get('total_trades', 0),
                        'win_rate': float(strategy_results.get('performance_metrics', {}).get('win_rate', 0)),
                        'profit_factor': float(strategy_results.get('performance_metrics', {}).get('profit_factor', 1.0)),
                        'sharpe_ratio': float(strategy_results.get('performance_metrics', {}).get('sharpe_ratio', 0)),
                        'max_drawdown': float(strategy_results.get('performance_metrics', {}).get('max_drawdown', 0)),
                        'total_return_pct': float(strategy_results.get('performance_metrics', {}).get('total_return_pct', 0))
                    },
                    'strategy_summary': strategy_results.get('strategy_summary', 'No summary available')
                }
                json.dump(serializable_results, f, indent=2)
                
            # Generate HTML report
            html_output = os.path.join(results_dirs['html'], 'strategy_optimization_results.html')
            charts_dir = os.path.join(results_dirs['base'], 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            if 'optimizer' in strategy_results and hasattr(strategy_results['optimizer'], 'generate_html_report'):
                strategy_results['optimizer'].generate_html_report(html_output, charts_dir)
                print(f"Generated strategy optimization HTML report: {html_output}")
            else:
                # Create basic HTML report
                print("Creating basic strategy optimization report")
                with open(html_output, 'w') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Strategy Optimization Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Trading Strategy Optimization Results</h1>
        
        <div class="alert alert-info">
            <h4>Strategy Parameters</h4>
            <ul>
                <li><strong>Pattern:</strong> {strategy_results['best_params'].get('use_pattern', 'N/A')}</li>
                <li><strong>Stop Loss:</strong> {strategy_results['best_params'].get('stop_loss_pct', 0.0):.1f}%</li>
                <li><strong>Take Profit:</strong> {strategy_results['best_params'].get('take_profit_pct', 0.0):.1f}%</li>
                <li><strong>Pattern Lag:</strong> {strategy_results['best_params'].get('pattern_lag', 0)} minutes</li>
                <li><strong>Position Size:</strong> {strategy_results['best_params'].get('position_size_pct', 0.0)}%</li>
            </ul>
        </div>
        
        <div class="alert alert-success">
            <h4>Performance Metrics</h4>
            <ul>
                <li><strong>Win Rate:</strong> {strategy_results.get('performance_metrics', {}).get('win_rate', 0.0)*100:.1f}%</li>
                <li><strong>Profit Factor:</strong> {strategy_results.get('performance_metrics', {}).get('profit_factor', 0.0):.2f}</li>
                <li><strong>Total Return:</strong> {strategy_results.get('performance_metrics', {}).get('total_return_pct', 0.0):.2f}%</li>
                <li><strong>Max Drawdown:</strong> {strategy_results.get('performance_metrics', {}).get('max_drawdown', 0.0):.2f}%</li>
                <li><strong>Sharpe Ratio:</strong> {strategy_results.get('performance_metrics', {}).get('sharpe_ratio', 0.0):.2f}</li>
            </ul>
        </div>
        
        <div class="mt-4">
            <h4>Strategy Summary</h4>
            <pre class="bg-light p-3">{strategy_results.get('strategy_summary', 'No summary available')}</pre>
        </div>
        
        <div class="mt-4">
            <a href="../reports/strategy_params.json" class="btn btn-primary" target="_blank">View Raw JSON Data</a>
            <a href="../charts/" class="btn btn-secondary">View Strategy Charts</a>
        </div>
    </div>
</body>
</html>""")
            
            # Generate visualization files directly to the charts directory
            if 'optimizer' in strategy_results and hasattr(strategy_results['optimizer'], 'visualize_results'):
                viz_files = strategy_results['optimizer'].visualize_results(results_dirs['base'])
                print(f"Strategy visualization files saved directly to charts directory")
                    
            print(f"Strategy optimization results saved to {strategy_output}")
            
        except Exception as e:
            print(f"Error in strategy optimization: {e}")
            import traceback
            traceback.print_exc()
            # Create default parameters
            strategy_results = {
                'best_params': {
                    'use_pattern': patterns.columns[0] if not patterns.empty and len(patterns.columns) > 0 else 'strong_up',
                    'pattern_lag': 5,
                    'stop_loss_pct': 2.0,
                    'take_profit_pct': 1.0,
                    'max_holding_time': 60,
                    'position_size_pct': 10,
                    'entry_threshold': 0.0001
                },
                'performance_metrics': {
                    'win_rate': 0.5,
                    'total_trades': 0,
                    'profit_factor': 1.0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_return_pct': 0
                },
                'strategy_summary': "Default strategy (no optimization performed)",
            }
    
    # Generate results index LAST after all other files are created
    # generate_results_index(results_dirs['base'], pattern_stats, altcoin_name)
    
    # Generate the main index.html file with links to all results
    all_result_dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    generate_index_html(all_result_dirs, RESULTS_DIR)
    
    # Print final analysis completion message
    print("\n======= ANALYSIS COMPLETE =======")
    print(f"Results saved to: {results_dirs['base']}")
    if serve:
        print(f"Starting web server on port {port}...")
        start_server(RESULTS_DIR, port)
    else:
        print("To view results, run:")
        print(f"  python -m src.server {RESULTS_DIR}")
    
    # Return results directory if requested (for API usage)
    if return_results_dir:
        return results_dirs['base']

if __name__ == "__main__":
    main()