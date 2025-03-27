import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import time

# Create an exports directory if it doesn't exist
if not os.path.exists('exports'):
    os.makedirs('exports')

# Initialize the Binance exchange
print("Initializing Binance API...")
exchange = ccxt.binance({
    'enableRateLimit': True,  # Enable built-in rate limiting
})

# Define the symbols
symbols = ['DOGE/USDT', 'BTC/USDT']

# Define the timeframe (1m is the smallest available)
timeframe = '1m'  # 1 minute timeframe (closest to tick-by-tick)

# Define the date range (one month)
end_date = datetime.now(timezone.utc)  # Use timezone-aware datetime
start_date = end_date - timedelta(days=30)

print(f"Fetching data from {start_date} to {end_date}")

# Fetch and save data
for symbol in symbols:
    print(f"Fetching {symbol} data...")
    try:
        # Fetch data in single request (last 1000 candles)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
        
        if not ohlcv:
            print(f"No data fetched for {symbol}.")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Print a sample of the data to check
        for index, row in df.iterrows():
            print(f"Date: {datetime.fromtimestamp(row['timestamp'].timestamp(), tz=timezone.utc)}, Open: {row['open']}, High: {row['high']}, Low: {row['low']}, Close: {row['close']}, Volume: {row['volume']}")
            if index >= 5:  # Print only the first 5 rows
                break
        
        # Generate filename and save
        symbol_filename = symbol.replace('/', '_')
        start_str = start_date.strftime('%d-%B-%Y')  # Day-MonthName-Year
        end_str = end_date.strftime('%d-%B-%Y')
        filename = f"exports/{symbol_filename}_{start_str}_to_{end_str}.csv"
        df.to_csv(filename, index=False)
        
        print(f"Saved {len(df)} records for {symbol} to {filename}")
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
    
    # Wait between symbols
    time.sleep(2)

print("Done!")