"""Main script to run the BTC-DOGE pattern analysis and serve results."""

import json
import os
import argparse
import shutil
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
from src import pattern_analysis
from src.data_processing import load_and_preprocess_data
from src.pattern_analysis import classify_momentum_patterns, analyze_lag_relationships
from src.ml_models import prepare_features_targets, train_xgboost_model, evaluate_model, plot_feature_importance
from src.visualization import (
    create_results_directory, 
    plot_lag_responses, 
    create_interactive_overview, 
    create_pattern_specific_plots, 
    generate_report, 
    generate_index_html,
    plot_ml_predictions
)
# Fixed import for server module
from src.server import start_server
from src.config import DEFAULT_BTC_PATH, DEFAULT_DOGE_PATH, RESULTS_DIR, MAX_LAG
from src.ai_analysis import generate_claude_reports

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

def main(return_results_dir=False):
    """Main function to run the BTC-DOGE pattern analysis."""
    parser = argparse.ArgumentParser(description="BTC-DOGE Momentum Pattern Analysis")
    parser.add_argument('--btc', default=DEFAULT_BTC_PATH,
                        help='Path to BTC data CSV')
    parser.add_argument('--doge', default=DEFAULT_DOGE_PATH,
                        help='Path to DOGE data CSV')
    parser.add_argument('--results', default=RESULTS_DIR,
                        help='Directory to store results')
    parser.add_argument('--max-lag', type=int, default=MAX_LAG,
                        help='Maximum lag (in minutes) to analyze')
    parser.add_argument('--serve', action='store_true',
                        help='Start a web server to view results after analysis')
    parser.add_argument('--port', type=int, default=None,
                        help='Port for the web server (random if not specified)')
    parser.add_argument('--use-ml', action='store_true',
                        help='Use XGBoost to analyze feature importance and relationships')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--optimize-strategy', action='store_true',
                        help='Optimize trading strategy parameters using ML techniques')
    
    args = parser.parse_args()
    
    # Step 1: Create results directory
    results_dirs = create_results_directory(args.results)
    
    # Step 2: Load and preprocess data
    combined_data = load_and_preprocess_data(args.btc, args.doge)
    
    # Step 3: Identify patterns
    patterns = classify_momentum_patterns(combined_data)
    combined_data = pd.concat([combined_data, patterns], axis=1)
    
    # Step 4: Traditional pattern analysis
    pattern_stats = analyze_lag_relationships(combined_data, patterns, args.max_lag)
    
    # Step 5: Machine learning analysis (if enabled)
    ml_results = None
    if args.use_ml:
        print("\nRunning XGBoost analysis...")
        
        # Prepare features and target
        X, y, feature_names = prepare_features_targets(
            combined_data, 
            target_col='doge_returns', 
            lag_periods=range(1, args.max_lag+1)
        )
        
        # Split data chronologically (respect time series nature)
        split_idx = int(len(X) * (1 - args.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Train XGBoost model
        model = train_xgboost_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Plot feature importance
        feature_importance = plot_feature_importance(model, feature_names, results_dirs['charts'])
        
        # plot_ml_predictions(y_test, metrics['predictions'], X_test.index, results_dirs['charts'])
        
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
    
    # Step 6: Generate traditional visualizations and reports
    print("\nGenerating visualizations and reports...")
    plot_lag_responses(pattern_stats, results_dirs['charts'])
    create_interactive_overview(combined_data, pattern_stats, results_dirs['html'])
    create_pattern_specific_plots(combined_data, pattern_stats, results_dirs['html'])
    generate_report(pattern_stats, results_dirs['reports'])
    
    # Step 6b: Optimize Trading Strategy (if requested)
    if args.optimize_strategy:
        print("\nOptimizing trading strategy parameters...")
        try:
            strategy_results = pattern_analysis.optimize_strategy_parameters(
                combined_data, patterns, max_lag=20
            )
            
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
            if 'optimizer' in strategy_results and hasattr(strategy_results['optimizer'], 'generate_html_report'):
                strategy_results['optimizer'].generate_html_report(html_output)
                print(f"Generated strategy optimization HTML report: {html_output}")
            
            # Copy visualization files to results directory
            for viz_file in ['sl_tp_heatmap.png', 'equity_curve.png', 'day_win_rate.png', 'parameter_importance.png']:
                if os.path.exists(viz_file):
                    shutil.copy(viz_file, os.path.join(results_dirs['charts'], viz_file))
                    
            print(f"Strategy optimization results saved to {strategy_output}")
            
        except Exception as e:
            print(f"Warning: Strategy optimization failed: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    # Step 7: Generate Claude AI analysis if API key is available
    claude_results = None  # No automatic Claude analysis
    
    # Step 8: Generate or update index.html
    all_result_dirs = [d for d in os.listdir(args.results) 
                     if os.path.isdir(os.path.join(args.results, d))]
    generate_index_html(all_result_dirs, args.results, claude_results)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {results_dirs['base']}")
    
    # Step 9: Start web server if requested
    if args.serve:
        print("\nStarting web server to view results...")
        start_server(args.results, args.port)
    else:
        print("\nTo view results, navigate to /results in your browser.")

    
    # Return results directory if requested (for API usage)
    if return_results_dir:
        return results_dirs['base']

if __name__ == "__main__":
    main()