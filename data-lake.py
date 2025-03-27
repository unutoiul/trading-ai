import datetime
import matplotlib.pyplot as plt
import pandas as pd
# from prophet import Prophet  # Commented out Prophet import
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import time
import ta  # Technical Analysis library

# Initialize exchange
exchange = ccxt.binance()

def fetch_historical_data(symbol, timeframe, start_date, end_date):
    all_ohlcv = []
    current_date = start_date
    
    while current_date < end_date:
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                exchange.parse8601(current_date.isoformat()),
                limit=1000  # Maximum records per request
            )
            
            if len(ohlcv) > 0:
                all_ohlcv.extend(ohlcv)
                # Update current_date to the last timestamp received
                current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000) + timedelta(minutes=1)
            else:
                break
                
            print(f"Fetched data until {current_date}")
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(2)  # Wait longer on error
            
    return all_ohlcv

# Set date range for 5 years
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print(f"Fetching data from {start_date} to {end_date}")

# Fetch historical data
symbol = 'BTC/USDT'
timeframe = '1m'  # Changed to daily timeframe
ohlcv = fetch_historical_data(symbol, timeframe, start_date, end_date)

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

print(f"Fetched {len(df)} records")

# Calculate mid price
df['mid_price'] = (df['high'] + df['low']) / 2

"""
# Commented out Prophet-related code
# Prepare data for Prophet
prophet_df = pd.DataFrame()
prophet_df['ds'] = df['timestamp']
prophet_df['y'] = df['mid_price']

# Create and fit the Prophet model
model = Prophet(
    changepoint_prior_scale=0.05,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)
model.fit(prophet_df)

# Create future dates for prediction
future_dates = model.make_future_dataframe(periods=30)
forecast = model.predict(future_dates)
"""

# Create exports directory if it doesn't exist
export_dir = 'exports'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# Generate timestamp for unique filenames
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Export raw data
df.to_csv(f'{export_dir}/raw_data_{timestamp}.csv', index=False)
# df.to_excel(f'{export_dir}/raw_data_{timestamp}.xlsx', index=False)
df.to_json(f'{export_dir}/raw_data_{timestamp}.json', orient='records')

# After creating the initial DataFrame, add technical indicators
def add_technical_indicators(df):
    # Add rolling windows
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
    
    # Add RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    
    # Add MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    
    # Add Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    # Add On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    return df

# After fetching the data and creating the DataFrame:
print("Adding technical indicators...")
df = add_technical_indicators(df)

# Export the enhanced dataset
export_dir = 'exports'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df.to_csv(f'{export_dir}/enhanced_data_{timestamp}.csv', index=False)

print("\nDataset now includes:")
print("1. Basic OHLCV data:")
print("   - Open, High, Low, Close prices")
print("   - Volume")
print("\n2. Technical Indicators:")
print("   - Simple Moving Averages (20, 50, 200 days)")
print("   - Relative Strength Index (RSI)")
print("   - Moving Average Convergence Divergence (MACD)")
print("   - Bollinger Bands")
print("   - Average True Range (ATR)")
print("   - On-Balance Volume (OBV)")

print(f"\nFiles exported to {export_dir}/ directory:")
print(f"- raw_data_{timestamp}.csv")
print(f"- raw_data_{timestamp}.xlsx")
print(f"- raw_data_{timestamp}.json")
print(f"- enhanced_data_{timestamp}.csv")