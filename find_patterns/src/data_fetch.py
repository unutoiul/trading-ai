import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import time
import argparse

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
        # Define priority coins - can easily add more
        priority_coins = {
            'BTC/USDT': '0',
            'ETH/USDT': '1',
            'DOGE/USDT': '2',
            'SOL/USDT': '3',
            'XRP/USDT': '4',
            'ADA/USDT': '5',
        }
        
        # Return priority number if it's a priority coin, otherwise just return the name
        return priority_coins.get(pair, pair)
    
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
                print(f" Fetching batch starting at ➡️ {datetime.fromtimestamp(current_since/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                
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
            
            # Include timeframe in the filename
            filename = f"data/{symbol_filename}_{timeframe}_{start_str}_to_{end_str}.csv"
            
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

def fetch_btc_and_altcoin(altcoin, days=10, timeframe='1m'):
    """
    Fetch both BTC and the specified altcoin data
    
    Args:
        altcoin: The altcoin to fetch (e.g., 'DOGE')
        days: Number of days of history to fetch
        timeframe: Candle timeframe
        
    Returns:
        Dictionary with results
    """
    # Ensure altcoin has the correct format
    if not altcoin.endswith('/USDT'):
        altcoin = f"{altcoin}/USDT"
    
    symbols = ['BTC/USDT', altcoin]
    
    # Define date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    print(f"Fetching {', '.join(symbols)} data from {start_date} to {end_date}")
    
    # Use the fetch_data function
    return fetch_data(symbols, start_date, end_date, timeframe)

# Main script execution
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Fetch cryptocurrency data.')
    parser.add_argument('--altcoin', type=str, default='DOGE', 
                        help='Altcoin to analyze against BTC (e.g., DOGE, ETH, SOL)')
    parser.add_argument('--days', type=int, default=10,
                        help='Number of days of history to fetch')
    parser.add_argument('--timeframe', type=str, default='1m',
                        help='Timeframe for candles (e.g., 1m, 5m, 15m, 1h)')
    parser.add_argument('--list', action='store_true',
                        help='List available trading pairs')
    
    args = parser.parse_args()
    
    # List available pairs if requested
    if args.list:
        print("Available trading pairs:")
        pairs = available_pairs()
        for pair in pairs[:50]:  # Show first 50 to avoid overwhelming output
            print(f"  {pair}")
        print(f"...and {len(pairs) - 50} more")
        exit()
    
    # Fetch data for BTC and the specified altcoin
    results = fetch_btc_and_altcoin(args.altcoin, args.days, args.timeframe)
    
    print("\nFetch results:")
    for result in results:
        if 'error' in result:
            print(f"❌ {result['symbol']}: {result['error']}")
        else:
            print(f"✅ {result['symbol']}: {result['records']} records saved to {result['filename']}")
    
    print("\nDone!")