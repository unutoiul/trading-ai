import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that's thread-safe

"""Machine learning models for crypto price prediction."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import shap  

def prepare_features_targets(data, target_col='doge_returns', lag_periods=range(1, 21), 
                           feature_cols=None):
    """
    Prepare features and target variables for machine learning.
    
    Args:
        data: DataFrame with combined BTC and DOGE data
        target_col: Column to predict (e.g., 'doge_returns')
        lag_periods: Range of lag periods to include
        feature_cols: List of columns to use as features
        
    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_names: List of feature names
    """
    # If no feature columns specified, use numeric columns except pattern columns
    if feature_cols is None:
        feature_cols = [col for col in data.select_dtypes(include=['number']).columns 
                       if not ('strong_' in col or 'volatility_' in col or 
                              'steady_' in col or 'rsi_' in col or 'macd_' in col or
                              'stoch_' in col or target_col == col)]
    
    # Create lagged features
    feature_data = data[feature_cols].copy()
    
    # Add lagged BTC features
    for lag in lag_periods:
        if lag > 0:
            for col in ['btc_returns', 'btc_momentum_15m', 'rsi_btc']:
                if col in data.columns:
                    feature_data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    # Prepare target variable
    target_data = data[target_col]
    
    # Drop rows with NaN values
    feature_data = feature_data.dropna()
    target_data = target_data.loc[feature_data.index]
    
    return feature_data, target_data, feature_data.columns.tolist()

def train_xgboost_model(X_train, y_train, params=None):
    """
    Train an XGBoost regression model for crypto returns prediction.
    
    Args:
        X_train: Training features
        y_train: Training targets
        params: XGBoost parameters
        
    Returns:
        Trained XGBoost model
    """
    # Default XGBoost parameters
    if params is None:
        params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate directional accuracy (up/down prediction)
    directional_accuracy = np.mean((y_test > 0) == (y_pred > 0))
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'predictions': y_pred
    }

def plot_feature_importance(model, feature_names, output_dir):
    """
    Plot feature importance for XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[-15:]  # Show top 15 features
    
    # Plot feature importance
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Saved feature importance plot to {output_file}")
    
    # Return sorted feature importance for reporting
    return [(feature_names[i], importance[i]) for i in indices]

def plot_shap_values(model, X_test, feature_names, output_dir):
    """Generate SHAP values to explain feature contributions."""
    try:
        # Create explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_importance.png'))
        plt.close()
        
        return True
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return False