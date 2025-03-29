import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import time

def available_pairs():
    """
    Get available trading pairs from Binance.
    
    Returns:
        List of USDT trading pairs, sorted with major coins first
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    markets = exchange.load_markets()
    
    # Extract USDT pairs
    usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
    
    # Sort pairs with BTC first, then other major coins, then alphabetically
    def sort_key(pair):
        if pair == 'BTC/USDT':
            return '0'
        elif pair == 'ETH/USDT':
            return '1'
        elif pair == 'DOGE/USDT':
            return '2'
        else:
            return pair
    
    usdt_pairs.sort(key=sort_key)
    return usdt_pairs

def fetch_data(pairs, start_date, end_date, timeframe='1m'):
    """
    Fetch historical data for selected trading pairs.
    
    Args:
        pairs: List of trading pairs (e.g., ['BTC/USDT', 'DOGE/USDT'])
        start_date: Start datetime (datetime object)
        end_date: End datetime (datetime object)
        timeframe: Candle timeframe (default: '1m')
        
    Returns:
        Dictionary with results for each pair
    """
    # Initialize exchange
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Track results
    results = []
    
    for i, symbol in enumerate(pairs):
        print(f"Fetching {symbol} data...")
        
        # Convert dates to milliseconds
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)
        
        all_candles = []
        current_since = since
        
        try:
            # Fetch data in batches
            while current_since < until:
                print(f"  Fetching batch starting at {datetime.fromtimestamp(current_since/1000, tz=timezone.utc)}")
                
                # The 1000 here is the API's maximum
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                
                if not candles:
                    print("  No more data available")
                    break
                    
                # Add this batch to our collection
                all_candles.extend(candles)
                
                # Move to next batch (start after the last timestamp we received)
                current_since = candles[-1][0] + 1
                
                # Stop if we've reached the end date
                if current_since >= until:
                    break
                    
                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)
            
            if not all_candles:
                print(f"No data fetched for {symbol}.")
                results.append({
                    'symbol': symbol,
                    'error': 'No data available'
                })
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])
            df = df.sort_values('timestamp')
            
            # Print sample
            print(f"Sample of fetched data (first 5 of {len(df)} records):")
            for index, row in df.head(5).iterrows():
                print(f"Date: {row['timestamp']}, Open: {row['open']}, Close: {row['close']}")
            
            # Save to file
            symbol_filename = symbol.replace('/', '_')
            start_str = start_date.strftime('%d-%B-%Y')
            end_str = end_date.strftime('%d-%B-%Y')
            filename = f"data/{symbol_filename}_{start_str}_to_{end_str}.csv"
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            df.to_csv(filename, index=False)
            
            print(f"Saved {len(df)} records for {symbol} to {filename}")
            
            results.append({
                'symbol': symbol,
                'records': len(df),
                'filename': filename,
                'display_name': f"{symbol} ({len(df)} records)"
            })
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'error': str(e)
            })
        
        # Wait between symbols
        time.sleep(2)
    
    return results

# Main script execution - this will run when you run data_fetch.py directly
if __name__ == "__main__":
    # Define symbols and timeframe
    symbols = ['DOGE/USDT', 'BTC/USDT']
    timeframe = '1m'  # 1 minute timeframe
    
    # Define date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=10)
    
    print(f"Fetching data from {start_date} to {end_date}")
    
    # Use the fetch_data function
    fetch_data(symbols, start_date, end_date, timeframe)
    
    print("Done!")