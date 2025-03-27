import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ta  # Technical Analysis library
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
print("Loading data...")
btc_data = pd.read_csv('exports/BTC_USDT_24-February-2025_to_26-March-2025.csv')
doge_data = pd.read_csv('exports/DOGE_USDT_24-February-2025_to_26-March-2025.csv')

# Convert timestamp to datetime
btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
doge_data['timestamp'] = pd.to_datetime(doge_data['timestamp'])

# Set timestamp as index
btc_data.set_index('timestamp', inplace=True)
doge_data.set_index('timestamp', inplace=True)

# Merge the data on the index
print("Merging datasets...")
combined_data = pd.merge(
    btc_data, 
    doge_data, 
    left_index=True, 
    right_index=True, 
    suffixes=('_btc', '_doge')
)

# Calculate returns
combined_data['btc_returns'] = combined_data['close_btc'].pct_change()
combined_data['doge_returns'] = combined_data['close_doge'].pct_change()

# Drop NaN values created by pct_change
combined_data.dropna(inplace=True)

print("Generating technical indicators...")
# Feature Engineering - Add technical indicators for both assets
for suffix in ['_btc', '_doge']:
    # RSI
    combined_data[f'rsi{suffix}'] = ta.momentum.RSIIndicator(
        close=combined_data[f'close{suffix}'], window=14
    ).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=combined_data[f'close{suffix}'])
    combined_data[f'macd{suffix}'] = macd.macd()
    combined_data[f'macd_signal{suffix}'] = macd.macd_signal()
    combined_data[f'macd_diff{suffix}'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=combined_data[f'close{suffix}'])
    combined_data[f'bb_width{suffix}'] = bollinger.bollinger_pband()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=combined_data[f'high{suffix}'],
        low=combined_data[f'low{suffix}'],
        close=combined_data[f'close{suffix}']
    )
    combined_data[f'stoch_k{suffix}'] = stoch.stoch()
    combined_data[f'stoch_d{suffix}'] = stoch.stoch_signal()
    
    # Rate of Change
    combined_data[f'roc{suffix}'] = ta.momentum.ROCIndicator(
        close=combined_data[f'close{suffix}'], window=10
    ).roc()

# Cross-asset features
# Create momentum features at different timeframes
for window in [5, 15, 30, 60]:
    # BTC momentum
    combined_data[f'btc_momentum_{window}m'] = combined_data['btc_returns'].rolling(window).sum()
    # DOGE momentum
    combined_data[f'doge_momentum_{window}m'] = combined_data['doge_returns'].rolling(window).sum()
    # Correlation between BTC and DOGE
    combined_data[f'corr_{window}m'] = combined_data['btc_returns'].rolling(window).corr(combined_data['doge_returns'])

# BTC leading indicators (lagged features)
for lag in [1, 3, 5, 10, 15]:
    combined_data[f'btc_returns_lag_{lag}'] = combined_data['btc_returns'].shift(lag)
    combined_data[f'btc_rsi_lag_{lag}'] = combined_data['rsi_btc'].shift(lag)

# Custom momentum indicators
combined_data['btc_up_momentum'] = (
    (combined_data['btc_returns'] > 0) & 
    (combined_data['btc_momentum_15m'] > 0) & 
    (combined_data['rsi_btc'] > 50)
)

# Target: DOGE returns after BTC momentum
combined_data['target_doge_future_return'] = combined_data['doge_returns'].shift(-1)  # Next period DOGE returns

# Drop rows with NaN values
combined_data.dropna(inplace=True)

print("Analyzing lag relationship...")
# Find optimal lag between BTC and DOGE
lag_corrs = []
for lag in range(0, 20):  # Test lags from 0 to 20 minutes
    # Shift DOGE returns back by 'lag' periods (BTC leads DOGE)
    lag_corr = pearsonr(
        combined_data['btc_returns'], 
        combined_data['doge_returns'].shift(-lag)
    )[0]
    lag_corrs.append((lag, lag_corr))

lag_df = pd.DataFrame(lag_corrs, columns=['lag_minutes', 'correlation'])
optimal_lag = lag_df.loc[lag_df['correlation'].abs().idxmax()]
print(f"Optimal lag: {optimal_lag['lag_minutes']} minutes (correlation: {optimal_lag['correlation']:.4f})")

# Add the optimally lagged DOGE returns as a feature
combined_data['doge_optimal_lagged'] = combined_data['doge_returns'].shift(-int(optimal_lag['lag_minutes']))
combined_data.dropna(inplace=True)

print("Preparing data for modeling...")
# Prepare data for modeling
X = combined_data.drop(['target_doge_future_return', 'doge_optimal_lagged', 
                       'open_btc', 'high_btc', 'low_btc', 'volume_btc', 
                       'open_doge', 'high_doge', 'low_doge', 'volume_doge'], axis=1)
y = combined_data['target_doge_future_return']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (keeping time order)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

print("Training XGBoost model...")
# Train XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_preds = xgb_model.predict(X_test)

# Calculate error metrics
rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print(f"XGBoost RMSE: {rmse:.6f}")

# Feature importance from XGBoost
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Compare different timeframes
timeframe_analysis = pd.DataFrame({
    'timeframe': ['5m', '15m', '30m', '60m'],
    'btc_momentum_corr': [
        combined_data[f'btc_momentum_5m'].corr(combined_data['doge_returns']),
        combined_data[f'btc_momentum_15m'].corr(combined_data['doge_returns']),
        combined_data[f'btc_momentum_30m'].corr(combined_data['doge_returns']),
        combined_data[f'btc_momentum_60m'].corr(combined_data['doge_returns'])
    ],
    'btc_momentum_next_corr': [
        combined_data[f'btc_momentum_5m'].corr(combined_data['target_doge_future_return']),
        combined_data[f'btc_momentum_15m'].corr(combined_data['target_doge_future_return']),
        combined_data[f'btc_momentum_30m'].corr(combined_data['target_doge_future_return']),
        combined_data[f'btc_momentum_60m'].corr(combined_data['target_doge_future_return'])
    ]
})

print("\nTimeframe Analysis:")
print(timeframe_analysis)

# Calculate DOGE returns during BTC uptrends vs. overall
doge_returns_on_btc_up = combined_data.loc[combined_data['btc_up_momentum'], 'doge_returns']
doge_returns_general = combined_data['doge_returns']

print(f"\nAverage DOGE returns when BTC has upward momentum: {doge_returns_on_btc_up.mean():.6f}")
print(f"Average DOGE returns overall: {doge_returns_general.mean():.6f}")
print(f"Ratio: {doge_returns_on_btc_up.mean() / doge_returns_general.mean():.2f}x")

# Generate trading signals
print("\nGenerating trading signals...")
combined_data['predicted_doge_return'] = np.nan
combined_data.iloc[len(combined_data)-len(xgb_preds):, combined_data.columns.get_loc('predicted_doge_return')] = xgb_preds

# Create trading signals
combined_data['signal'] = 0
threshold = combined_data['doge_returns'].std() * 0.2  # Dynamic threshold based on volatility

# Buy signal: BTC has upward momentum and model predicts positive DOGE returns
combined_data.loc[
    (combined_data['btc_up_momentum']) & 
    (combined_data['predicted_doge_return'] > threshold), 
    'signal'
] = 1

# Sell/avoid signal
combined_data.loc[
    (~combined_data['btc_up_momentum']) | 
    (combined_data['predicted_doge_return'] < -threshold), 
    'signal'
] = -1

# Calculate strategy returns
combined_data['strategy_returns'] = combined_data['signal'].shift(1) * combined_data['doge_returns']
combined_data.dropna(subset=['strategy_returns'], inplace=True)

print(f"Strategy cumulative returns: {combined_data['strategy_returns'].sum():.4f}")
print(f"Buy-and-hold returns: {combined_data['doge_returns'].sum():.4f}")

# Create visualization
print("Creating visualization...")
fig = make_subplots(rows=4, cols=1, 
                   shared_xaxes=True,
                   vertical_spacing=0.05,
                   subplot_titles=('BTC-DOGE Prices (Normalized)', 'BTC-DOGE Momentum', 'Correlation', 'Trading Signals'))

# Normalize prices for comparison
btc_norm = combined_data['close_btc'] / combined_data['close_btc'].iloc[0]
doge_norm = combined_data['close_doge'] / combined_data['close_doge'].iloc[0]

# Plot 1: Normalized prices
fig.add_trace(go.Scatter(x=combined_data.index, y=btc_norm, name='BTC (norm)'), row=1, col=1)
fig.add_trace(go.Scatter(x=combined_data.index, y=doge_norm, name='DOGE (norm)'), row=1, col=1)

# Plot 2: Momentum
fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['btc_momentum_15m'], name='BTC 15m Momentum'), row=2, col=1)
fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['doge_momentum_15m'], name='DOGE 15m Momentum'), row=2, col=1)

# Plot 3: Correlation
fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['corr_15m'], name='15m Rolling Correlation'), row=3, col=1)

# Plot 4: Trading signals
fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['signal'], mode='markers', 
                        marker=dict(size=10, color=combined_data['signal'], 
                                   colorscale='RdYlGn', showscale=True),
                        name='Trading Signal'), row=4, col=1)

# Update layout
fig.update_layout(height=1200, title_text="BTC-DOGE Momentum Analysis and Trading Signals")
fig.write_html('btc_doge_momentum_analysis.html')
print("Analysis complete! Visualization saved to btc_doge_momentum_analysis.html")

# Generate a summary report
with open('momentum_trading_strategy.txt', 'w') as f:
    f.write("BTC-DOGE MOMENTUM TRADING STRATEGY\n")
    f.write("==================================\n\n")
    f.write(f"Optimal lag between BTC and DOGE: {optimal_lag['lag_minutes']} minutes\n")
    f.write(f"Optimal timeframe for momentum: {timeframe_analysis.iloc[timeframe_analysis['btc_momentum_next_corr'].abs().idxmax()]['timeframe']}\n\n")
    f.write("TOP INDICATORS BY IMPORTANCE:\n")
    for idx, row in feature_importance.head(10).iterrows():
        f.write(f"- {row['feature']}: {row['importance']:.4f}\n")
    f.write("\nTRADING STRATEGY:\n")
    f.write("1. Monitor BTC for upward momentum (positive returns, RSI > 50, positive 15m momentum)\n")
    f.write(f"2. When BTC shows upward momentum, expect DOGE to follow in approximately {optimal_lag['lag_minutes']} minutes\n")
    f.write("3. Use the model to predict DOGE's expected return\n")
    f.write("4. Take a long position in DOGE when the model predicts returns above threshold\n")
    f.write("5. Exit or avoid DOGE when BTC momentum turns negative or predicted returns fall below threshold\n\n")
    f.write(f"Strategy performance: {combined_data['strategy_returns'].sum():.4f} vs Buy-and-hold: {combined_data['doge_returns'].sum():.4f}\n")
print("Trading strategy saved to momentum_trading_strategy.txt")