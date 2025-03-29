"""Main script to run the BTC-DOGE pattern analysis and serve results."""

import os
import argparse
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
        
        # Plot predictions vs actual
        plot_ml_predictions(y_test, metrics['predictions'], X_test.index, results_dirs['charts'])
        
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
    
    # Step 7: Generate Claude AI analysis if API key is available
    claude_results = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\nGenerating Claude 3.7 AI analysis...")
        claude_results = generate_claude_reports(pattern_stats, ml_results, results_dirs)
        
        # Add claude results to index.html for easy access
        if claude_results and 'analysis_report' in claude_results:
            # This will be handled by the index generator
            pass
    else:
        print("\nSkipping Claude 3.7 analysis - no API key found")
        print("Set the ANTHROPIC_API_KEY environment variable to enable AI analysis")
    
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