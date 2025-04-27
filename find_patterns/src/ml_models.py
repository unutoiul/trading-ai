import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that's thread-safe

"""Machine learning models for crypto pattern analysis and strategy optimization."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

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

def train_xgboost_model(X_train, y_train, params=None):
    """
    Placeholder function that returns a dummy model object.
    XGBoost functionality has been removed.
    
    Args:
        X_train: Training features (unused)
        y_train: Training targets (unused)
        params: XGBoost parameters (unused)
        
    Returns:
        Dummy model object
    """
    print("XGBoost training has been disabled. Using correlation-based analysis instead.")
    
    # Return a dummy model object with the minimal required attributes
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X))
        
        def get_booster(self):
            class DummyBooster:
                def get_score(self, importance_type=None):
                    return {}
            return DummyBooster()
    
    return DummyModel()

def evaluate_model(model, X_test, y_test):
    """
    Placeholder function that returns dummy metrics.
    XGBoost evaluation has been removed.
    
    Args:
        model: Model object (unused)
        X_test: Test features (unused)
        y_test: Test targets (unused)
        
    Returns:
        Dictionary with dummy metrics
    """
    print("XGBoost evaluation has been disabled. Using correlation-based analysis instead.")
    
    # Return dummy metrics
    return {
        'mse': 0.0,
        'rmse': 0.0,
        'mae': 0.0,
        'r2': 0.0,
        'directional_accuracy': 0.5
    }

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

def analyze_directional_impact_ml(combined_data, directional_impact):
    """Use ML to find optimal conditions for BTC directional moves to impact altcoin prices."""
    print("\nUsing ML to analyze optimal conditions for directional impact...")
    
    # Import needed libraries
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import os
    
    # Make charts directory if it doesn't exist
    os.makedirs("results/charts", exist_ok=True)
    
    # Include all six movement categories
    all_scenarios = [
        'btc_up_strong', 'btc_up_medium', 'btc_up_small',
        'btc_down_small', 'btc_down_medium', 'btc_down_strong'
    ]
    results = {}
    
    # Main analysis loop for each scenario
    for scenario in all_scenarios:
        if scenario not in directional_impact:
            print(f"Scenario {scenario} not found in directional impact data")
            continue
            
        print(f"Analyzing ML factors for {scenario}...")
        scenario_data = directional_impact[scenario]
        instances = scenario_data['instances']
        
        # Create a dataset of all instances where this scenario occurred
        mask = combined_data.index[combined_data[scenario] == True][:instances]
        scenario_df = combined_data.loc[mask].copy()
        
        if len(scenario_df) < 50:  # Need minimum samples for ML
            print(f"Not enough samples for {scenario}: {len(scenario_df)}")
            continue
            
        # Find altcoin return column
        altcoin_col = None
        for col in scenario_df.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                altcoin_col = col
                break
                
        if not altcoin_col:
            print(f"No altcoin returns column found for {scenario}")
            continue
        
        alt_name = altcoin_col.split('_')[0].upper()
        print(f"  Using {alt_name} returns column: {altcoin_col}")
            
        # Calculate forward returns at each minute from 1 to 15
        for lag in range(1, 16):
            scenario_df[f'forward_return_{lag}min'] = scenario_df[altcoin_col].shift(-lag)
        
        # Create target variables (strong move in expected direction)
        threshold = 0.0005  # 0.05% move is considered significant
        
        if 'up' in scenario:
            for lag in range(1, 16):
                scenario_df[f'target_{lag}min'] = scenario_df[f'forward_return_{lag}min'] > threshold
        else:
            for lag in range(1, 16):
                scenario_df[f'target_{lag}min'] = scenario_df[f'forward_return_{lag}min'] < -threshold
        
        # Select features for ML - be more selective with categories
        features = []
        
        # Technical indicators by category
        feature_categories = {
            'momentum': ['rsi', 'momentum', 'macd'],
            'volume': ['volume'],
            'volatility': ['volatility', 'atr'],
            'trend': ['sma', 'ema', 'trend'],
            'price_action': ['close', 'open', 'high', 'low']
        }
        
        category_features = {}
        for category, indicators in feature_categories.items():
            category_cols = []
            for col in scenario_df.columns:
                if any(ind in col.lower() for ind in indicators):
                    if 'forward' not in col and 'target' not in col:
                        category_cols.append(col)
                        features.append(col)
            category_features[category] = category_cols
            print(f"  {category.capitalize()}: {len(category_cols)} features")
                        
        # Ensure we have some features
        if not features:
            print(f"No valid features found for {scenario}")
            continue
            
        print(f"  Using {len(features)} total features for modeling")
        
        # Train ML models for each minute
        model_results = {}
        minute_accuracies = []
        
        for lag in range(1, 16):  # Every minute from 1 to 15
            target_col = f'target_{lag}min'
            
            # Prepare data
            X = scenario_df[features].fillna(0)
            y = scenario_df[target_col].fillna(False)
            
            if len(X) < 100:
                print(f"  Not enough samples for {lag}min lag: {len(X)}")
                continue
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model - use more trees for better accuracy
            model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            minute_accuracies.append(accuracy)
            
            # Extract feature importance
            feature_importance = list(zip(features, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate category importance
            category_importance = {}
            for category, category_cols in category_features.items():
                if category_cols:
                    importance_sum = sum(model.feature_importances_[features.index(col)] 
                                        for col in category_cols if col in features)
                    category_importance[category] = importance_sum
            
            # Store results
            model_results[lag] = {
                'accuracy': accuracy,
                'feature_importance': feature_importance[:10],
                'category_importance': category_importance,
                'model': model
            }
            
            # Print results
            print(f"  {lag}min prediction accuracy: {accuracy*100:.1f}%")
            print(f"  Top features: {', '.join([f[0] for f in feature_importance[:3]])}")
            
        # Create accuracy vs time plot for this scenario
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(minute_accuracies) + 1), [acc * 100 for acc in minute_accuracies], 
                marker='o', linewidth=2)
        plt.title(f'{scenario.replace("_", " ").title()} Impact on {alt_name} - Prediction Accuracy by Minute')
        plt.xlabel('Minutes After BTC Movement')
        plt.ylabel('Prediction Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.ylim(40, 100)
        
        # Add reference line at 50%
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, 
                   label='Random Guess (50%)')
        
        # Find best minute
        if minute_accuracies:
            best_minute = range(1, len(minute_accuracies) + 1)[minute_accuracies.index(max(minute_accuracies))]
            plt.axvline(x=best_minute, color='g', linestyle='--', alpha=0.5,
                      label=f'Best Minute ({best_minute})')
                      
            # Add annotation for best minute
            plt.annotate(f'Peak: {max(minute_accuracies)*100:.1f}%',
                        xy=(best_minute, max(minute_accuracies)*100),
                        xytext=(best_minute+0.5, max(minute_accuracies)*100),
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
            
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/charts/{scenario}_minute_accuracy.png")
        plt.close()
        
        # Create feature category importance chart for best minute
        if minute_accuracies:
            best_minute = range(1, len(minute_accuracies) + 1)[minute_accuracies.index(max(minute_accuracies))]
            best_result = model_results[best_minute]
            
            plt.figure(figsize=(10, 6))
            categories = list(best_result['category_importance'].keys())
            importances = list(best_result['category_importance'].values())
            
            # Sort by importance
            sorted_indices = np.argsort(importances)
            categories = [categories[i] for i in sorted_indices]
            importances = [importances[i] for i in sorted_indices]
            
            plt.barh(categories, importances)
            plt.title(f'Feature Category Importance for {scenario.replace("_", " ").title()} (Minute {best_minute})')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(f"results/charts/{scenario}_category_importance.png")
            plt.close()
        
        results[scenario] = model_results
    
    # Create summary chart comparing all scenarios
    plt.figure(figsize=(12, 8))
    
    colors = {
        'btc_up_strong': 'darkgreen',
        'btc_up_medium': 'forestgreen', 
        'btc_up_small': 'lightgreen',
        'btc_down_small': 'lightcoral',
        'btc_down_medium': 'indianred',
        'btc_down_strong': 'darkred'
    }
    
    for scenario in results:
        # Get accuracies for each minute
        minutes = sorted(results[scenario].keys())
        accuracies = [results[scenario][m]['accuracy'] * 100 for m in minutes]
        
        # Plot with appropriate color
        plt.plot(minutes, accuracies, label=scenario.replace('_', ' ').title(),
                color=colors.get(scenario, 'blue'), marker='o', linewidth=2)
    
    plt.title(f'BTC Movement Impact on {alt_name} - Prediction Accuracy by Minute')
    plt.xlabel('Minutes After BTC Movement')
    plt.ylabel('Prediction Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/charts/all_scenarios_accuracy.png")
    plt.close()
    
    return results