import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that's thread-safe

"""Machine learning models for crypto pattern analysis and strategy optimization."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
    
# Import needed libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def prepare_features_targets(data, target_col=None, lag_periods=range(1, 21), 
                            feature_cols=None):
    """
    Prepare features and target variables for analysis.
    
    Args:
        data: DataFrame with combined BTC and altcoin data
        target_col: Column to predict (e.g., 'doge_returns') - will be auto-detected if None
        lag_periods: Range of lag periods to include
        feature_cols: List of columns to use as features
        
    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_names: List of feature names
    """
    # Auto-detect target column if not specified
    if target_col is None:
        for col in data.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                target_col = col
                print(f"Auto-detected target column: {target_col}")
                break
        
        if target_col is None:
            raise ValueError("Could not auto-detect target column. Please specify target_col.")
    
    # Extract altcoin prefix for later use
    alt_prefix = target_col.split('_')[0]
    
    # If no feature columns specified, use numeric columns except pattern columns
    if feature_cols is None:
        feature_cols = [col for col in data.select_dtypes(include=['number']).columns 
                       if not ('pattern' in col or 'volatility_' in col or 
                              'steady_' in col or col == target_col)]
    
    # Create lagged features
    feature_data = data[feature_cols].copy()
    
    # Add lagged BTC features
    for lag in lag_periods:
        if lag > 0:
            for col in ['btc_returns', f'btc_momentum_15m', 'rsi_btc']:
                if col in data.columns:
                    feature_data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    # Prepare target variable
    target_data = data[target_col]
    
    # Drop rows with NaN values
    feature_data = feature_data.dropna()
    target_data = target_data.loc[feature_data.index]
    
    return feature_data, target_data, feature_data.columns.tolist()

def prepare_features_for_analysis(data, feature_cols=None):
    """
    Prepare features for pattern analysis and strategy optimization.
    
    Args:
        data: DataFrame with combined BTC and altcoin data
        feature_cols: List of columns to use as features
        
    Returns:
        Processed feature DataFrame and feature names
    """
    # If no feature columns specified, use numeric columns except pattern columns
    if feature_cols is None:
        feature_cols = [col for col in data.select_dtypes(include=['number']).columns 
                       if not ('pattern' in col)]
    
    # Create feature DataFrame
    feature_data = data[feature_cols].copy()
    
    # Drop rows with NaN values
    feature_data = feature_data.dropna()
    
    return feature_data, feature_data.columns.tolist()

def analyze_feature_importance(data, target_col=None, output_dir=None):
    """
    Analyze feature importance using correlation with target variable.
    
    Args:
        data: DataFrame with features and target
        target_col: Target column name (will auto-detect if None)
        output_dir: Directory to save output (optional)
    
    Returns:
        DataFrame with correlation-based importance
    """
    # Auto-detect target column if not specified
    if target_col is None:
        for col in data.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                target_col = col
                print(f"Auto-detected target column: {target_col}")
                break
        
        if target_col is None:
            raise ValueError("Could not auto-detect target column. Please specify target_col.")
    
    # Calculate correlation with target
    correlations = data.corr()[target_col].drop(target_col)
    
    # Sort by absolute correlation
    importance = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values,
        'abs_correlation': abs(correlations.values)
    }).sort_values('abs_correlation', ascending=False)
    
    if output_dir:
        # Plot top 15 features
        plt.figure(figsize=(12, 8))
        top_features = importance.head(15)
        plt.barh(top_features['feature'], top_features['correlation'])
        plt.title(f'Feature Importance for {target_col}')
        plt.xlabel('Correlation')
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_file = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Saved feature importance plot to {output_file}")
    
    return importance

def analyze_optimal_lags(data, alt_prefix=None, max_lag=20):
    """
    Analyze optimal lag between BTC and altcoin returns using correlation.
    
    Args:
        data: DataFrame with time series data
        alt_prefix: Altcoin prefix (e.g., 'doge') - will auto-detect if None
        max_lag: Maximum lag to analyze
        
    Returns:
        Dictionary with lag correlations and optimal lag
    """
    # Auto-detect altcoin prefix if not specified
    if alt_prefix is None:
        for col in data.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                alt_prefix = col.split('_')[0]
                print(f"Auto-detected altcoin prefix: {alt_prefix}")
                break
        
        if alt_prefix is None:
            raise ValueError("Could not auto-detect altcoin. Please specify alt_prefix.")
    
    source_col = 'btc_returns'
    target_col = f'{alt_prefix}_returns'
    
    if source_col not in data.columns or target_col not in data.columns:
        raise ValueError(f"Could not find {source_col} or {target_col} in the data")
    
    lag_correlations = {}
    lag_win_rates = {}
    lag_returns = {}
    
    # Calculate correlation for each lag
    for lag in range(1, max_lag + 1):
        # Shift source column by lag
        lagged_source = data[source_col].shift(lag)
        
        # Calculate correlation with target
        valid_indices = ~(lagged_source.isna() | data[target_col].isna())
        if sum(valid_indices) > 10:
            correlation = np.corrcoef(
                lagged_source[valid_indices], 
                data[target_col][valid_indices]
            )[0, 1]
            
            lag_correlations[lag] = correlation if not np.isnan(correlation) else 0
            
            # Calculate win rate (% of times direction is predicted correctly)
            direction_match = (lagged_source > 0) == (data[target_col] > 0)
            lag_win_rates[lag] = direction_match[valid_indices].mean()
            
            # Calculate average return
            lag_returns[lag] = data[target_col][valid_indices].mean()
    
    # Find optimal lag
    if lag_correlations:
        optimal_lag = max(lag_correlations.items(), key=lambda x: abs(x[1]))[0]
    else:
        optimal_lag = 1
    
    result = {
        'lag_correlations': lag_correlations,
        'lag_win_rates': lag_win_rates,
        'lag_returns': lag_returns,
        'optimal_lag': optimal_lag,
        'optimal_correlation': lag_correlations.get(optimal_lag, 0),
        'optimal_win_rate': lag_win_rates.get(optimal_lag, 0),
        'optimal_return': lag_returns.get(optimal_lag, 0)
    }
    
    return result

def plot_lag_analysis(lag_analysis, output_dir=None, alt_prefix=None):
    """
    Plot lag analysis results.
    
    Args:
        lag_analysis: Result from analyze_optimal_lags
        output_dir: Directory to save plot (optional)
        alt_prefix: Altcoin prefix for title (will use "Altcoin" if None)
    
    Returns:
        None
    """
    if not lag_analysis or 'lag_correlations' not in lag_analysis:
        print("No lag analysis data to plot")
        return
    
    if alt_prefix is None:
        alt_prefix = "Altcoin"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot correlations
    lags = list(lag_analysis['lag_correlations'].keys())
    correlations = list(lag_analysis['lag_correlations'].values())
    
    ax1.plot(lags, correlations, marker='o', linewidth=2)
    ax1.set_title(f'BTC-{alt_prefix} Return Correlation by Lag')
    ax1.set_ylabel('Correlation')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Highlight optimal lag
    opt_lag = lag_analysis['optimal_lag']
    ax1.axvline(x=opt_lag, color='g', linestyle='--', alpha=0.8)
    
    # Plot win rates
    win_rates = [lag_analysis['lag_win_rates'].get(lag, 0) * 100 for lag in lags]
    
    ax2.plot(lags, win_rates, marker='o', color='g', linewidth=2)
    ax2.set_title(f'BTC-{alt_prefix} Direction Prediction Win Rate by Lag')
    ax2.set_xlabel('Lag (minutes)')
    ax2.set_ylabel('Win Rate (%)')
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation for optimal lag
    opt_corr = lag_analysis['optimal_correlation']
    opt_win = lag_analysis['optimal_win_rate'] * 100
    
    ax1.annotate(f'Optimal lag: {opt_lag} min\nCorrelation: {opt_corr:.4f}\nWin rate: {opt_win:.1f}%',
                xy=(opt_lag, correlations[lags.index(opt_lag)]),
                xytext=(opt_lag + 1, correlations[lags.index(opt_lag)]),
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_file = os.path.join(output_dir, f'lag_analysis_{alt_prefix}.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Saved lag analysis plot to {output_file}")
    else:
                                plt.show()


def evaluate_sklearn_model(model, X_test, y_test):
    """
    Evaluate sklearn model performance with proper metrics.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Determine if classification or regression
        is_classification = hasattr(model, 'predict_proba') or hasattr(model, 'classes_')
        
        if is_classification:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Convert to binary if needed
            if len(np.unique(y_test)) == 2:
                y_test_binary = (y_test > 0).astype(int) if not all(isinstance(val, (int, bool, np.integer, np.bool_)) for val in y_test) else y_test
                y_pred_binary = (y_pred > 0).astype(int) if not all(isinstance(val, (int, bool, np.integer, np.bool_)) for val in y_pred) else y_pred
            else:
                y_test_binary = y_test
                y_pred_binary = y_pred
            
            metrics = {
                'accuracy': accuracy_score(y_test_binary, y_pred_binary),
                'precision': precision_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0),
                'recall': recall_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0),
                'directional_accuracy': accuracy_score(y_test_binary, y_pred_binary)
            }
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'directional_accuracy': np.mean((y_test > 0) == (y_pred > 0)) if len(y_test) > 0 else 0.5
            }
        
        print(f"Model evaluation completed. Accuracy: {metrics.get('accuracy', metrics.get('directional_accuracy', 0)):.3f}")
        return metrics
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {
            'accuracy': 0.5,
            'directional_accuracy': 0.5
        }


def analyze_directional_impact_ml(combined_data, directional_impact, results_dirs=None):
    """
    Analyze directional impact using machine learning models.
    Focus on correlation analysis instead of XGBoost.
    
    Args:
        combined_data: DataFrame with combined BTC and altcoin data
        directional_impact: DataFrame with directional impact analysis
        results_dirs: Dictionary with result directories
        
    Returns:
        Dictionary with analysis results
    """
    print("Analyzing directional impact using correlation-based methods...")
    
    try:
        if combined_data.empty or directional_impact.empty:
            print("No data available for ML analysis")
            return {}
        
        # Create correlation-based analysis instead of ML
        correlation_results = {}
        
        # Analyze correlation between BTC signals and altcoin responses
        if 'btc_direction' in combined_data.columns and 'price_change' in combined_data.columns:
            correlation = combined_data['btc_direction'].corr(combined_data['price_change'])
            correlation_results['btc_altcoin_correlation'] = correlation
            
        print(f"Correlation analysis completed. BTC-Altcoin correlation: {correlation_results.get('btc_altcoin_correlation', 0):.3f}")
        
        return {
            'correlation_results': correlation_results,
            'method': 'correlation_analysis',
            'status': 'completed'
        }
        
    except Exception as e:
        print(f"Error in directional impact analysis: {e}")
        return {
            'error': str(e),
            'status': 'failed'
        }


def ensure_dir_exists(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_lag_features(data, target_col, lag_range=(1, 10)):
    """
    Create lag features for time series analysis.
    
    Args:
        data: DataFrame with time series data
        target_col: Column to create lags for
        lag_range: Tuple of (min_lag, max_lag)
        
    Returns:
        DataFrame with lag features added
    """
    df = data.copy()
    
    for lag in range(lag_range[0], lag_range[1] + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df.dropna()


def perform_correlation_analysis(data, target_col='price_change', feature_cols=None):
    """
    Perform correlation analysis instead of complex ML models.
    
    Args:
        data: DataFrame with features and target
        target_col: Target column for correlation
        feature_cols: List of feature columns
        
    Returns:
        Dictionary with correlation results
    """
    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != target_col and data[col].dtype in ['float64', 'int64']]
    
    correlations = {}
    for col in feature_cols:
        if col in data.columns and target_col in data.columns:
            corr = data[col].corr(data[target_col])
            if not np.isnan(corr):
                correlations[col] = corr
    
    return correlations


def plot_feature_importance(model, feature_names, output_dir=None):
    """
    Placeholder function that creates a simple correlation chart instead of XGBoost feature importance.
    
    Args:
        model: Model object (unused)
        feature_names: List of feature names
        output_dir: Directory to save output chart
        
    Returns:
        Empty list (no feature importance values)
    """
    print("XGBoost feature importance has been disabled. Using correlation-based analysis instead.")
    
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "XGBoost feature importance has been disabled.\nUsing correlation analysis instead.",
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save placeholder chart
        output_file = os.path.join(output_dir, 'feature_importance_placeholder.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Created placeholder feature importance chart at {output_file}")
    
    # Return empty feature importance list for compatibility
    return []

def analyze_directional_impact_ml(combined_data, directional_impact, results_dirs=None):
    """
    Enhanced directional impact analysis with optional ML.
    Uses ML when available, falls back to price action analysis.
    """
    print("\nAnalyzing optimal lag relationships...")
    
    # Setup directories and find altcoin name
    if results_dirs and 'charts' in results_dirs:
        charts_dir = results_dirs['charts']
    else:
        charts_dir = "results/charts"
        
    os.makedirs(charts_dir, exist_ok=True)
    
    alt_name = None
    for col in combined_data.columns:
        if col.endswith('_returns') and not col.startswith('btc'):
            alt_name = col.split('_')[0].upper()
            break
    
    if not alt_name:
        alt_name = "ALTCOIN"
    
    all_scenarios = [
        'btc_strong_up', 'btc_medium_up', 'btc_small_up',
        'btc_small_down', 'btc_medium_down', 'btc_strong_down'
    ]
    
    results = {}
    
    # Use direct correlation analysis instead of ML models
    print("Using direct correlation analysis for BTC-altcoin relationship detection")
    
    for scenario in all_scenarios:
        if scenario not in directional_impact:
            print(f"Scenario {scenario} not found in directional impact data")
            continue
            
        print(f"Analyzing lag impact for {scenario}...")
        scenario_data = directional_impact[scenario]
        instances = scenario_data.get('instances', 0)
        
        if instances < 10:
            print(f"  Too few instances ({instances}) for reliable analysis")
            continue
        
        lag_returns = {}
        lag_win_rates = {}
        
        # Use direct correlation analysis from existing data
        for lag, lag_data in scenario_data.get('lags', {}).items():
            lag_returns[lag] = lag_data.get('mean_return', 0)
            lag_win_rates[lag] = lag_data.get('win_rate', 0)
        
        if not lag_returns:
            print(f"  No lag data available for {scenario}")
            continue
            
        optimal_lag = max(lag_returns.items(), key=lambda x: x[1] * lag_win_rates.get(x[0], 0))[0]
        
        results[scenario] = {
            optimal_lag: {
                'mean_return': lag_returns.get(optimal_lag, 0),
                'win_rate': lag_win_rates.get(optimal_lag, 0),
                'accuracy': lag_win_rates.get(optimal_lag, 0),
                'features': ['btc_returns', f'btc_momentum_{optimal_lag}', 'btc_volatility_15'],
                'model': None
            }
        }
        
        print(f"  Optimal lag: {optimal_lag} minutes")
        print(f"  Mean return: {lag_returns.get(optimal_lag, 0):.4f}%")
        print(f"  Win rate: {lag_win_rates.get(optimal_lag, 0)*100:.1f}%")
    
    plt.figure(figsize=(12, 8))
    
    colors = {
        'btc_strong_up': 'darkgreen',
        'btc_medium_up': 'forestgreen', 
        'btc_small_up': 'lightgreen',
        'btc_small_down': 'lightcoral',
        'btc_medium_down': 'indianred',
        'btc_strong_down': 'darkred'
    }
    
    for scenario in results:
        if not results[scenario]:
            continue
            
        lags = list(directional_impact[scenario]['lags'].keys())
        returns = [directional_impact[scenario]['lags'][lag]['mean_return'] for lag in lags]
        
        plt.plot(lags, returns, 'o-', color=colors.get(scenario, 'blue'),
                 label=f"{scenario.replace('btc_', '').replace('_', ' ').title()}")
    
    plt.title(f'BTC Price Movement Impact on {alt_name} Returns by Lag')
    plt.xlabel('Minutes After BTC Movement')
    plt.ylabel('Mean Return (%)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/price_action_lag_impact.png")
    plt.close()
    
    return results