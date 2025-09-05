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

# Configuration constants (formerly from config.py)
RESULTS_DIR = 'results'

from src.strategy_generator import DirectionalImpactStrategies
from src.feature_engineering import (
    add_enhanced_momentum_features,
    add_price_action_features,
    add_relationship_features,
    add_composite_features
)
from src.conditions_filter import ConditionsFilter
from src.vectorbt_optimizer import VectorBTOptimizer

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

def check_win_rates_threshold(directional_impact, min_threshold=0.50):
    """
    Check if any scenario in directional impact analysis has win rates above threshold.
    
    Args:
        directional_impact: Results from analyze_btc_directional_impact
        min_threshold: Minimum win rate threshold (default: 0.50 = 50%)
        
    Returns:
        bool: True if any scenario has win rates above threshold, False otherwise
    """
    if not directional_impact or len(directional_impact) == 0:
        print("❌ No directional impact data available")
        return False
    
    print(f"\n🔍 Checking win rates against {min_threshold*100:.0f}% threshold...")
    
    best_win_rate = 0.0
    best_scenario = None
    best_lag = None
    scenarios_above_threshold = 0
    total_scenarios_checked = 0
    
    for scenario_name, scenario_data in directional_impact.items():
        if 'lags' not in scenario_data:
            continue
            
        for lag, lag_data in scenario_data['lags'].items():
            total_scenarios_checked += 1
            win_rate = lag_data.get('win_rate', 0)
            count = lag_data.get('count', 0)
            
            # Only consider scenarios with sufficient data points
            if count >= 5:
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_scenario = scenario_name
                    best_lag = lag
                
                if win_rate >= min_threshold:
                    scenarios_above_threshold += 1
                    print(f"   ✅ {scenario_name} (lag {lag}min): {win_rate*100:.1f}% win rate ({count} trades)")
                else:
                    print(f"   ❌ {scenario_name} (lag {lag}min): {win_rate*100:.1f}% win rate ({count} trades)")
    
    print(f"\n📊 Win Rate Analysis Summary:")
    print(f"   • Best win rate: {best_win_rate*100:.1f}% ({best_scenario}, {best_lag}min lag)")
    print(f"   • Scenarios above {min_threshold*100:.0f}%: {scenarios_above_threshold}/{total_scenarios_checked}")
    print(f"   • Threshold check: {'PASSED' if scenarios_above_threshold > 0 else 'FAILED'}")
    
    return scenarios_above_threshold > 0

def apply_condition_filters(combined_data, altcoin_name, conditions=None):
    """Apply condition filters to data for segmentation."""
    print("\nApplying condition filters for data segmentation...")
    
    # Create condition filter
    filter_manager = ConditionsFilter(combined_data, altcoin_name)
    
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

def run_directional_analysis_only(btc_file, alt_file, conditions=None, custom_settings=None, vectorbt_settings=None):
    """Run only the directional impact ML analysis and strategy generation."""
    print("\n=== RUNNING DIRECTIONAL IMPACT ANALYSIS ONLY ===")
    
    # Log VectorBT settings if provided
    if vectorbt_settings:
        print("🚀 VectorBT Optimization Settings:")
        print(f"   • Max Combinations: {vectorbt_settings.get('max_combinations', 10000):,}")
        print(f"   • Parallel Jobs: {vectorbt_settings.get('parallel_jobs', 4)}")
        print(f"   • Trailing Stop Testing: {'ENABLED' if vectorbt_settings.get('enable_trailing_stop', True) else 'DISABLED'}")
        print(f"   • Detailed Reports: {'ENABLED' if vectorbt_settings.get('detailed_reports', True) else 'DISABLED'}")
    
    # Log custom settings if provided
    if custom_settings:
        print("🎯 Custom Analysis Settings Detected:")
        print(f"   • Strong threshold: {custom_settings.get('strongThreshold', 0.15)}%")
        print(f"   • Medium threshold: {custom_settings.get('mediumThreshold', 0.075)}%")
        print(f"   • Active timeframes: {', '.join(custom_settings.get('timeframes', ['1m']))}")
        print(f"   • Sustained moves: {'enabled' if custom_settings.get('sustainedMoves') else 'disabled'}")
        print(f"   • Volatility breakouts: {'enabled' if custom_settings.get('volatilityBreakouts') else 'disabled'}")
        print(f"   • Cross-timeframe: {'enabled' if custom_settings.get('crossTimeframe') else 'disabled'}")
        print(f"   • Min confidence: {custom_settings.get('minConfidence', 0.6) * 100:.0f}%")
        print(f"   • Lookback periods: {custom_settings.get('lookbackPeriods', 50)}")
        print(f"   • Selected conditions: {len(custom_settings.get('selectedConditions', []))}")
    
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
    
    # Apply condition filters if provided with custom settings support
    if conditions or custom_settings:
        print("\nApplying condition filters for data segmentation...")
        
        # Create condition filter with custom settings if provided
        filter_manager = ConditionsFilter(combined_data, altcoin_name)
        
        # Pass custom settings to condition generation if available
        if custom_settings:
            # Extract thresholds from custom settings
            strong_threshold = custom_settings.get('strongThreshold', 0.15) / 100  # Convert percentage to decimal
            medium_threshold = custom_settings.get('mediumThreshold', 0.075) / 100
            timeframes = [int(tf.replace('m', '')) for tf in custom_settings.get('timeframes', ['1', '2', '3', '5', '10'])]
            
            print(f"🎯 Using custom thresholds: Strong={strong_threshold*100:.2f}%, Medium={medium_threshold*100:.2f}%")
            print(f"🎯 Using custom timeframes: {timeframes}")
            
            # Generate conditions with custom settings
            available_conditions = filter_manager.define_standard_conditions(
                volatility_threshold=strong_threshold
            )
            
            # Create custom condition mapping based on settings
            if custom_settings.get('selectedConditions'):
                conditions = custom_settings['selectedConditions']
                print(f"🎯 Using {len(conditions)} custom-selected conditions")
        else:
            # Use standard condition generation
            available_conditions = filter_manager.define_standard_conditions()
        
        # Apply filters
        if conditions:
            filtered_data = filter_manager.apply_filters(conditions)
            print(f"Applied filters: {conditions}")
        else:
            filtered_data = combined_data
            print("No specific filters applied, using complete dataset")
        
        # Save the filter configuration with custom settings
        filter_config = {
            'conditions': conditions if conditions else [],
            'filtered_size': len(filtered_data),
            'original_size': len(combined_data),
            'percentage': len(filtered_data) / len(combined_data) * 100
        }
        
        if custom_settings:
            filter_config['custom_settings'] = custom_settings
            
        filter_report = os.path.join(results_dirs['reports'], 'filter_config.json')
        with open(filter_report, 'w') as f:
            json.dump(filter_config, f, indent=2)
            
    else:
        # If no conditions provided, use all data
        filtered_data = combined_data
        print("No filters applied, using complete dataset")
    
    # Report on filtered data size
    print(f"Filtered dataset size: {len(filtered_data)} rows")
    
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
        
        # Pass custom settings to strategy generator if available
        if custom_settings:
            strategy_generator.custom_settings = custom_settings
            print("🎯 Custom settings passed to strategy generator")
        
        # Generate strategies
        strategy_generator.generate_strategies()
        
        # Backtest strategies
        strategy_generator.backtest_strategies()
        
        # Generate reports
        strategy_generator.generate_strategy_reports(results_dirs)
        
        # Step 5: VectorBT Comprehensive Backtesting (if vectorbt_settings provided)
        if vectorbt_settings:
            try:
                print("\n" + "="*80)
                print("Step 5: VectorBT Comprehensive Backtesting")
                print("="*80)
                
                # Check win rates before running VectorBT optimization
                vbt_settings = vectorbt_settings or {}
                min_win_rate_threshold = vbt_settings.get('win_rate_threshold', 0.50)  # Default 50%
                should_run_vectorbt = check_win_rates_threshold(directional_impact, min_win_rate_threshold)
                
                if not should_run_vectorbt:
                    print("❌ SKIPPING VectorBT Optimization:")
                    print(f"   • Win rates are below {min_win_rate_threshold*100:.0f}% threshold")
                    print(f"   • No profitable patterns detected - VectorBT testing would be ineffective")
                    print(f"   • Consider adjusting analysis parameters or using different data")
                    
                    # Save a note about why VectorBT was skipped
                    skip_report = os.path.join(results_dirs['reports'], 'vectorbt_skipped.txt')
                    with open(skip_report, 'w') as f:
                        f.write("VectorBT Optimization Skipped\n")
                        f.write("=" * 35 + "\n\n")
                        f.write(f"Reason: Win rates below {min_win_rate_threshold*100:.0f}% threshold\n")
                        f.write("All tested scenarios showed poor win rates indicating:\n")
                        f.write("- Patterns are not predictive enough for profitable trading\n")
                        f.write("- Market conditions may not be suitable for these strategies\n")
                        f.write("- Consider using different timeframes or market conditions\n\n")
                        f.write("Recommendation: Review pattern analysis results and adjust parameters\n")
                    
                    print(f"📄 Skip report saved to: {skip_report}")
                else:
                    print("✅ Win rate check passed - proceeding with VectorBT optimization")
                    
                    # Initialize VectorBT optimizer with full dataset
                    vectorbt_optimizer = VectorBTOptimizer(filtered_data, directional_impact_results)
                    
                    # Prepare VectorBT optimization settings
                    max_combinations = vbt_settings.get('max_combinations', 2000)  # Reduced for browser stability
                    parallel_jobs = vbt_settings.get('parallel_jobs', 4)
                    enable_trailing_stop = vbt_settings.get('enable_trailing_stop', True)
                    detailed_reports = vbt_settings.get('detailed_reports', True)
                    
                    print(f"🚀 Starting Enhanced VectorBT Optimization:")
                    print(f"   • Max Combinations: {max_combinations:,}")
                    print(f"   • Parallel Jobs: {parallel_jobs}")
                    print(f"   • Trailing Stop: {'ENABLED' if enable_trailing_stop else 'DISABLED'}")
                    print(f"   • Detailed Reports: {'ENABLED' if detailed_reports else 'DISABLED'}")
                    
                    # Run enhanced comprehensive backtesting
                    vectorbt_results = vectorbt_optimizer.optimize_strategies(
                        ml_results=directional_impact_results if directional_impact_results else None,
                        momentum_results=directional_impact_results,
                        results_dir=results_dirs['base'],
                        enable_trailing_stop=enable_trailing_stop,
                        max_combinations=max_combinations,
                        n_jobs=parallel_jobs
                    )
                    
                    print(f"✅ VectorBT Enhanced Optimization completed!")
                    print(f"   • Total strategies tested: {vectorbt_results.get('total_tested', 0):,}")
                    print(f"   • Profitable strategies: {vectorbt_results.get('profitable_count', 0):,}")
                    print(f"   • Best return: {vectorbt_results.get('best_return_pct', 0):.2f}%")
                    
                    if detailed_reports and vectorbt_results.get('best_strategies'):
                        print(f"   • Detailed reports generated with comprehensive analysis")
                        print(f"   • Parameter insights and optimal ranges calculated")
                        if enable_trailing_stop and 'trailing_stop_analysis' in vectorbt_results:
                            print(f"   • Trailing stop analysis completed")
                        
            except Exception as e:
                print(f"Error running VectorBT backtesting: {str(e)}")
                traceback.print_exc()
        
        print("\n✅ Directional analysis and strategy generation complete!")
        print(f"📁 Results saved to: {results_dirs['base']}")
        
        if custom_settings:
            print(f"🎯 Analysis completed with custom settings:")
            print(f"   • Processed {len(filtered_data)} data points")
            print(f"   • Used {len(custom_settings.get('selectedConditions', []))} custom conditions")
            print(f"   • Applied {len(custom_settings.get('timeframes', []))} timeframes")
        
    except Exception as e:
        print(f"❌ Error in directional analysis: {str(e)}")
        traceback.print_exc()
        print("Continuing with basic analysis...")
    
    # Generate index.html
    try:
        update_index_html(results_dirs, altcoin_name)
    except Exception as e:
        print(f"Error generating index HTML: {str(e)}")
    
    # Return the results directory
    return results_dirs['base']


def main(btc_file=None, alt_file=None, use_ml=True, optimize_strategy=True, serve=False, port=8000, 
         return_results_dir=False, conditions=None, vectorbt_settings=None):
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
        args = parser.parse_args()
        
        btc_file = args.btc
        alt_file = args.alt
        use_ml = args.use_ml
        optimize_strategy = args.optimize_strategy
        serve = args.serve
        port = args.port
        conditions = args.condition
    
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
        
    print("\n" + "="*80)
    print("🚀 STARTING COMPREHENSIVE CRYPTO PATTERN ANALYSIS")
    print("="*80)
    print(f"📊 Analyzing: BTC and {altcoin_name.upper()}")
    print(f"📁 Results will be saved to: {results_dirs['base']}")
    print(f"📈 Dataset size: {len(combined_data):,} data points")
    
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
    print("\n🔧 STEP 1: STANDARDIZING DATA COLUMNS")
    print("─" * 50)
    mapped_count = 0
    for source, target in column_mappings.items():
        if source in combined_data.columns and target not in combined_data.columns:
            combined_data[target] = combined_data[source]
            print(f"  ✅ Mapped '{source}' → '{target}'")
            mapped_count += 1
    print(f"✅ Column standardization complete: {mapped_count} columns mapped")
            
    # Add enhanced features to data - DON'T reassign the DataFrame!
    print("\n🔧 STEP 2: FEATURE ENGINEERING")
    print("─" * 50)
    print("🔄 Adding enhanced momentum features...")
    add_enhanced_momentum_features(combined_data, prefix='btc')
    print(f"✅ Enhanced features added. New dataset shape: {combined_data.shape}")
    
    # Apply condition filters if requested
    if conditions:
        print("\n🔧 STEP 3: APPLYING CONDITION FILTERS")
        print("─" * 50)
        print(f"🎯 Applying {len(conditions)} custom conditions...")
        filtered_data, filter_manager = apply_condition_filters(combined_data, altcoin_name, conditions)
        # Save the filter configuration
        filter_report = os.path.join(results_dirs['reports'], 'filter_config.json')
        with open(filter_report, 'w') as f:
            json.dump({
                'conditions': conditions,
                'filtered_size': len(filtered_data),
                'original_size': len(combined_data),
                'percentage': len(filtered_data) / len(combined_data) * 100
            }, f, indent=2)
        print(f"✅ Filtered data: {len(filtered_data):,} points ({len(filtered_data)/len(combined_data)*100:.1f}% of original)")
    else:
        print("\n🔧 STEP 3: NO FILTERING APPLIED")
        print("─" * 50)
        filtered_data = combined_data
        filter_manager = None
        print("✅ Using full dataset without filters")
    
    # Step 3: Identify patterns
    print("\n🔧 STEP 4: PATTERN IDENTIFICATION")
    print("─" * 50)
    print("🔄 Identifying momentum patterns...")
    patterns = classify_momentum_patterns(filtered_data)
    filtered_data = pd.concat([filtered_data, patterns], axis=1)
    pattern_count = len([col for col in patterns.columns if patterns[col].sum() > 0])
    print(f"✅ Pattern identification complete: {pattern_count} active patterns found")
    
    # Step 4: Pattern analysis
    print("\n🔧 STEP 5: LAG RELATIONSHIP ANALYSIS")
    print("─" * 50)
    print("🔄 Analyzing lag relationships between patterns...")
    pattern_stats = analyze_lag_relationships(filtered_data, patterns, max_lag_seconds=600, lag_step_seconds=10)
    print("✅ Lag relationship analysis complete")
    
    # Step 5: ML analysis (optional)
    ml_results = None
    if use_ml:
        print("\n🔧 STEP 6: MACHINE LEARNING ANALYSIS")
        print("─" * 50)
        try:
            print("🔄 Applying machine learning to pattern analysis...")
            # Additional price action features for ML
            add_price_action_features(filtered_data, prefix='btc')
            add_relationship_features(filtered_data, btc_prefix='btc', alt_prefix=altcoin_name)
            
            # Run directional impact analysis
            print("🔄 Running directional impact analysis...")
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
    
    # Step 7.5: VectorBT Comprehensive Backtesting
    try:
        print("\n" + "="*80)
        print("Step 7.5: VectorBT Comprehensive Backtesting")
        print("="*80)
        
        # Initialize VectorBT optimizer with proper asset data
        assets = {
            "btc": combined_data[['btc_close', 'btc_high', 'btc_low', 'btc_open', 'btc_volume']].copy(),
            altcoin_name: combined_data[[f'{altcoin_name}_close', f'{altcoin_name}_high', 
                                       f'{altcoin_name}_low', f'{altcoin_name}_open', 
                                       f'{altcoin_name}_volume']].copy()
        }
        
        vectorbt_optimizer = VectorBTOptimizer(assets, pattern_stats)
        
        # Prepare VectorBT optimization settings
        vbt_settings = vectorbt_settings or {}
        max_combinations = vbt_settings.get('max_combinations', 10000)
        parallel_jobs = vbt_settings.get('parallel_jobs', 4)
        enable_trailing_stop = vbt_settings.get('enable_trailing_stop', True)
        detailed_reports = vbt_settings.get('detailed_reports', True)
        
        print(f"🚀 Starting Enhanced VectorBT Optimization:")
        print(f"   • Max Combinations: {max_combinations:,}")
        print(f"   • Parallel Jobs: {parallel_jobs}")
        print(f"   • Trailing Stop: {'ENABLED' if enable_trailing_stop else 'DISABLED'}")
        print(f"   • Detailed Reports: {'ENABLED' if detailed_reports else 'DISABLED'}")
        
        # Run enhanced comprehensive backtesting
        vectorbt_results = vectorbt_optimizer.optimize_strategies(
            ml_results=ml_results if ml_results else None,
            momentum_results=pattern_stats,
            results_dir=results_dirs['base'],
            enable_trailing_stop=enable_trailing_stop,
            max_combinations=max_combinations,
            n_jobs=parallel_jobs
        )
        
        print(f"✅ VectorBT Enhanced Optimization completed!")
        print(f"   • Total strategies tested: {vectorbt_results.get('total_tested', 0):,}")
        print(f"   • Profitable strategies: {vectorbt_results.get('profitable_count', 0):,}")
        print(f"   • Best return: {vectorbt_results.get('best_return_pct', 0):.2f}%")
        
        if detailed_reports and vectorbt_results.get('best_strategies'):
            print(f"   • Detailed reports generated with comprehensive analysis")
            print(f"   • Parameter insights and optimal ranges calculated")
            if enable_trailing_stop and 'trailing_stop_analysis' in vectorbt_results:
                print(f"   • Trailing stop analysis completed")
        
        # Log recommended parameters if available
        if 'recommended_params' in vectorbt_results:
            rec_params = vectorbt_results['recommended_params']
            if 'best_single_strategy' in rec_params:
                best = rec_params['best_single_strategy']
                print(f"🏆 Best Strategy Parameters:")
                print(f"   • Pattern: {best.get('pattern', 'N/A')}")
                print(f"   • Stop Loss: {best.get('stop_loss', 0)*100:.2f}%")
                print(f"   • Take Profit: {best.get('take_profit', 0)*100:.2f}%")
                print(f"   • Trailing Stop: {best.get('trailing_stop', 0)*100:.2f}%")
        
    except Exception as e:
        print(f"Error running VectorBT backtesting: {str(e)}")
        traceback.print_exc()
    
    # Step 8: Serve results (optional)
    if serve:
        print(f"\n🌐 STARTING WEB SERVER")
        print("─" * 50)
        print(f"🔄 Starting web server on port {port}...")
        start_server(results_dirs['base'], port=port)
    
    # Final comprehensive summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE - COMPREHENSIVE SUMMARY")
    print("="*80)
    print(f"📊 Dataset Analysis:")
    print(f"   • Original size: {len(combined_data):,} data points")
    if conditions:
        print(f"   • Filtered size: {len(filtered_data):,} data points ({len(filtered_data)/len(combined_data)*100:.1f}%)")
    print(f"   • Patterns identified: {pattern_count if 'pattern_count' in locals() else 'N/A'}")
    
    print(f"\n🎯 Analysis Components Completed:")
    print(f"   ✅ Data preprocessing and feature engineering")
    print(f"   ✅ Pattern identification and classification")
    print(f"   ✅ Lag relationship analysis")
    if ml_results:
        print(f"   ✅ Machine learning directional impact analysis")
        print(f"   ✅ Strategy generation and backtesting")
    if optimize_strategy:
        print(f"   ✅ VectorBT optimization with enhanced parameters")
    
    print(f"\n📁 Results Location:")
    print(f"   📂 Main directory: {results_dirs['base']}")
    print(f"   📊 Charts: {results_dirs['charts']}")
    print(f"   📄 Reports: {results_dirs['html']}")
    print(f"   🔧 Config: {results_dirs['reports']}")
    
    if not serve:
        print(f"\n💡 To view results:")
        print(f"   🌐 Open: {results_dirs['base']}/index.html")
        print(f"   🚀 Or run with --serve flag to start web server")
    
    print("="*80)
    
    print(f"\nAnalysis complete! Results saved to: {results_dirs['base']}")
    
    # Return results directory if requested (for API usage)
    if return_results_dir:
        return results_dirs['base']