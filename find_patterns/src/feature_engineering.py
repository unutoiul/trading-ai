"""Technical indicator and feature generation for crypto analysis."""

import pandas as pd
import numpy as np
import ta
from finta import TA

def generate_technical_indicators(df, suffix, asset_name=None):
    """
    Add technical indicators to the dataframe.
    
    Args:
        df: DataFrame with price data
        suffix: String suffix for column names (e.g., '_btc')
        asset_name: Asset name for logging (optional)
        
    Returns:
        DataFrame with added technical indicators
    """
    asset_display = asset_name if asset_name else suffix.strip('_')
    print(f"Generating technical indicators for {asset_display}...")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # RSI
    df[f'rsi{suffix}'] = ta.momentum.RSIIndicator(
        close=df[f'close{suffix}'], window=14
    ).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df[f'close{suffix}'])
    df[f'macd{suffix}'] = macd.macd()
    df[f'macd_signal{suffix}'] = macd.macd_signal()
    df[f'macd_diff{suffix}'] = macd.macd_diff()
    
    # Stochastic Oscillator - Using ta instead of FinTA
    stoch = ta.momentum.StochasticOscillator(
        high=df[f'high{suffix}'],
        low=df[f'low{suffix}'],
        close=df[f'close{suffix}'],
        window=14,
        smooth_window=3
    )
    df[f'stoch_k{suffix}'] = stoch.stoch()
    df[f'stoch_d{suffix}'] = stoch.stoch_signal()
    
    # Rate of Change
    df[f'roc{suffix}'] = ta.momentum.ROCIndicator(
        close=df[f'close{suffix}'], window=9
    ).roc()
    
    return df

def create_momentum_features(df, asset_prefix):
    """
    Create momentum and volatility features at different timeframes.
    
    Args:
        df: DataFrame with returns columns
        asset_prefix: String prefix for the asset (e.g., 'btc')
        
    Returns:
        DataFrame with momentum features
    """
    df = df.copy()
    
    # Create momentum features at different timeframes
    for window in [5, 15, 30, 60]:
        df[f'{asset_prefix}_momentum_{window}m'] = df[f'{asset_prefix}_returns'].rolling(window).sum()
        df[f'{asset_prefix}_volatility_{window}m'] = df[f'{asset_prefix}_returns'].rolling(window).std()
    
    return df

def calculate_momentum_indicators(df, btc_prefix='btc', alt_prefix=None):
    """
    Calculate momentum indicators for both BTC and altcoin using FinTA.
    """
    from finta import TA
    
    # Find altcoin prefix if not provided
    if not alt_prefix:
        for col in df.columns:
            if col.endswith('_returns') and not col.startswith(btc_prefix):
                alt_prefix = col.split('_')[0]
                break
    
    if not alt_prefix:
        print("WARNING: Could not identify altcoin columns")
        return df
    
    print(f"Calculating momentum indicators for {btc_prefix} and {alt_prefix}...")
    
    # Create OHLCV dataframes for FinTA
    btc_ohlc = pd.DataFrame({
        'open': df[f'{btc_prefix}_open'],
        'high': df[f'{btc_prefix}_high'],
        'low': df[f'{btc_prefix}_low'],
        'close': df[f'{btc_prefix}_close'],
        'volume': df[f'volume_{btc_prefix}'] if f'volume_{btc_prefix}' in df.columns else df['volume']
    })
    
    alt_ohlc = pd.DataFrame({
        'open': df[f'{alt_prefix}_open'],
        'high': df[f'{alt_prefix}_high'],
        'low': df[f'{alt_prefix}_low'],
        'close': df[f'{alt_prefix}_close'],
        'volume': df[f'volume_{alt_prefix}'] if f'volume_{alt_prefix}' in df.columns else df['volume']
    })
    
    # Calculate BTC momentum indicators with FinTA
    df[f'rsi_{btc_prefix}'] = TA.RSI(btc_ohlc, 14)
    df[f'{btc_prefix}_cci'] = TA.CCI(btc_ohlc)
    df[f'{btc_prefix}_mfi'] = TA.MFI(btc_ohlc)
    
    # Calculate BTC momentum over different timeframes
    for period in [5, 15, 30, 60]:
        df[f'{btc_prefix}_momentum_{period}m'] = TA.MOM(btc_ohlc, period)
        
    # Calculate BTC trend indicators
    macd = TA.MACD(btc_ohlc)
    df[f'{btc_prefix}_macd'] = macd['MACD']
    df[f'{btc_prefix}_macd_signal'] = macd['SIGNAL']
    df[f'{btc_prefix}_macd_hist'] = macd['MACD'] - macd['SIGNAL']
    
    # Calculate BTC volatility
    for period in [15, 30, 60]:
        df[f'{btc_prefix}_volatility_{period}m'] = df[f'{btc_prefix}_close'].pct_change().rolling(period).std() * 100
    
    df[f'{btc_prefix}_atr'] = TA.ATR(btc_ohlc, 14)
    
    # Calculate altcoin momentum indicators with FinTA
    df[f'rsi_{alt_prefix}'] = TA.RSI(alt_ohlc, 14)
    df[f'{alt_prefix}_cci'] = TA.CCI(alt_ohlc)
    df[f'{alt_prefix}_mfi'] = TA.MFI(alt_ohlc)
    
    # Calculate altcoin momentum over different timeframes
    for period in [5, 15, 30, 60]:
        df[f'{alt_prefix}_momentum_{period}m'] = TA.MOM(alt_ohlc, period)
    
    # Calculate altcoin volatility
    for period in [15, 30, 60]:
        df[f'{alt_prefix}_volatility_{period}m'] = df[f'{alt_prefix}_close'].pct_change().rolling(period).std() * 100
    
    # Calculate relative indicators
    if all(indicator in df.columns for indicator in [f'{btc_prefix}_momentum_15m', f'{alt_prefix}_momentum_15m']):
        df[f'{alt_prefix}_vs_{btc_prefix}_momentum_15m'] = df[f'{alt_prefix}_momentum_15m'] - df[f'{btc_prefix}_momentum_15m']
    
    if all(indicator in df.columns for indicator in [f'{btc_prefix}_volatility_15m', f'{alt_prefix}_volatility_15m']):
        df[f'{alt_prefix}_vs_{btc_prefix}_volatility_15m'] = (
            df[f'{alt_prefix}_volatility_15m'] / df[f'{btc_prefix}_volatility_15m']
        ).replace([np.inf, -np.inf], np.nan).fillna(1)
    
    return df

def preprocess_data(btc_data, alt_data, alt_prefix=None):
    """
    Preprocess BTC and altcoin data and merge into a single dataframe.
    """
    print("Preprocessing data...")
    
    # Detect altcoin prefix if not provided
    if alt_prefix is None:
        # Try to detect from column names
        if 'symbol' in alt_data.columns:
            alt_prefix = alt_data['symbol'].iloc[0].lower()
        else:
            # Try to detect from filename patterns
            alt_prefix = 'alt'
            # Extract from column names if possible
            for col in alt_data.columns:
                if col.startswith('close_') and col != 'close_btc':
                    alt_prefix = col.split('_')[1]
                    break
    
    print(f"Processing altcoin data with prefix: {alt_prefix}")
    
    # Convert timestamp to datetime
    btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
    alt_data['timestamp'] = pd.to_datetime(alt_data['timestamp'])
    
    # Set timestamp as index
    btc_data.set_index('timestamp', inplace=True)
    alt_data.set_index('timestamp', inplace=True)
    
    # Merge the data on the index
    print("Merging datasets...")
    combined_data = pd.merge(
        btc_data, 
        alt_data, 
        left_index=True, 
        right_index=True, 
        suffixes=('_btc', f'_{alt_prefix}')
    )

    
    # Calculate returns
    combined_data['btc_returns'] = combined_data['close_btc'].pct_change().replace([np.inf, -np.inf], np.nan)
    combined_data[f'{alt_prefix}_returns'] = combined_data[f'close_{alt_prefix}'].pct_change().replace([np.inf, -np.inf], np.nan)
    
    # Prepare OHLCV dataframes
    btc_ohlc = pd.DataFrame({
        'open': combined_data['open_btc'],
        'high': combined_data['high_btc'],
        'low': combined_data['low_btc'],
        'close': combined_data['close_btc'],
        'volume': combined_data['volume_btc']
    })
    
    alt_ohlc = pd.DataFrame({
        'open': combined_data[f'open_{alt_prefix}'],
        'high': combined_data[f'high_{alt_prefix}'],
        'low': combined_data[f'low_{alt_prefix}'],
        'close': combined_data[f'close_{alt_prefix}'],
        'volume': combined_data[f'volume_{alt_prefix}']
    })
    
    # Calculate BTC indicators
    print("Calculating BTC indicators...")
    
    # RSI - use ta library instead of FinTA for more reliability
    combined_data['rsi_btc'] = ta.momentum.RSIIndicator(
        close=combined_data['close_btc'], window=14
    ).rsi().replace([np.inf, -np.inf], np.nan)
    
    # Momentum
    for period in [15, 30, 60]:
        # Calculate momentum using simple price difference for reliability
        combined_data[f'btc_momentum_{period}m'] = (
            combined_data['close_btc'] - combined_data['close_btc'].shift(period)
        ).replace([np.inf, -np.inf], np.nan)
    
    # Volatility - using simple std calculation
    for period in [15, 30, 60]:
        combined_data[f'btc_volatility_{period}m'] = (
            combined_data['close_btc'].pct_change().rolling(period).std() * 100
        ).replace([np.inf, -np.inf], np.nan)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        close=combined_data['close_btc'], window=20, window_dev=2
    )
    combined_data['btc_bb_upper'] = bb.bollinger_hband()
    combined_data['btc_bb_middle'] = bb.bollinger_mavg()
    combined_data['btc_bb_lower'] = bb.bollinger_lband()
    
    # MACD
    macd = ta.trend.MACD(close=combined_data['close_btc'])
    combined_data['btc_macd'] = macd.macd()
    combined_data['btc_macd_signal'] = macd.macd_signal()
    combined_data['btc_macd_histogram'] = macd.macd_diff()
    
    # Stochastic
    stoch_btc = ta.momentum.StochasticOscillator(
        high=combined_data['high_btc'],
        low=combined_data['low_btc'],
        close=combined_data['close_btc'],
        window=14,
        smooth_window=3
    )
    combined_data['btc_stoch_k'] = stoch_btc.stoch()
    combined_data['btc_stoch_d'] = stoch_btc.stoch_signal()
    
    # ATR
    combined_data['btc_atr'] = ta.volatility.AverageTrueRange(
        high=combined_data['high_btc'],
        low=combined_data['low_btc'],
        close=combined_data['close_btc'],
        window=14
    ).average_true_range()
    
    # Calculate altcoin indicators
    print(f"Calculating {alt_prefix.upper()} indicators...")
    
    # RSI
    combined_data[f'rsi_{alt_prefix}'] = ta.momentum.RSIIndicator(
        close=combined_data[f'close_{alt_prefix}'], window=14
    ).rsi().replace([np.inf, -np.inf], np.nan)
    
    # Momentum
    for period in [15, 30, 60]:
        combined_data[f'{alt_prefix}_momentum_{period}m'] = (
            combined_data[f'close_{alt_prefix}'] - combined_data[f'close_{alt_prefix}'].shift(period)
        ).replace([np.inf, -np.inf], np.nan)
    
    # Volatility
    for period in [15, 30, 60]:
        combined_data[f'{alt_prefix}_volatility_{period}m'] = (
            combined_data[f'close_{alt_prefix}'].pct_change().rolling(period).std() * 100
        ).replace([np.inf, -np.inf], np.nan)
    
    # MACD
    macd_alt = ta.trend.MACD(close=combined_data[f'close_{alt_prefix}'])
    combined_data[f'{alt_prefix}_macd'] = macd_alt.macd()
    combined_data[f'{alt_prefix}_macd_signal'] = macd_alt.macd_signal()
    combined_data[f'{alt_prefix}_macd_histogram'] = macd_alt.macd_diff()
    
    # Stochastic
    stoch_alt = ta.momentum.StochasticOscillator(
        high=combined_data[f'high_{alt_prefix}'],
        low=combined_data[f'low_{alt_prefix}'],
        close=combined_data[f'close_{alt_prefix}'],
        window=14,
        smooth_window=3
    )
    combined_data[f'{alt_prefix}_stoch_k'] = stoch_alt.stoch()
    combined_data[f'{alt_prefix}_stoch_d'] = stoch_alt.stoch_signal()
    
    # Add correlation indicators
    corr_window = 30
    if len(combined_data) >= corr_window:
        # Safe correlation calculation
        combined_data[f'btc_{alt_prefix}_correlation'] = combined_data['btc_returns'].rolling(
            window=corr_window).corr(combined_data[f'{alt_prefix}_returns'])
    
    # Handle NaN values
  
    key_cols = ['btc_momentum_15m', 'btc_returns', f'rsi_{alt_prefix}', 'btc_macd']
    available_cols = [col for col in key_cols if col in combined_data.columns]

    if available_cols:
        print("NaN counts in key columns:", combined_data[available_cols].isnull().sum())

    # Fill NaNs with appropriate values
    # For momentum and returns, fill with 0
    momentum_cols = [col for col in combined_data.columns if 'momentum' in col or 'returns' in col]
    combined_data[momentum_cols] = combined_data[momentum_cols].fillna(0)

    # For RSI, fill with 50 (neutral)
    rsi_cols = [col for col in combined_data.columns if 'rsi' in col]
    combined_data[rsi_cols] = combined_data[rsi_cols].fillna(50)

    # For stochastic oscillator, fill with neutral values
    stoch_cols = [col for col in combined_data.columns if 'stoch_' in col]
    combined_data[stoch_cols] = combined_data[stoch_cols].fillna(50)

    # For other indicators, use forward fill then backward fill
    combined_data = combined_data.ffill().bfill()

    # Final safety check - replace any remaining NaNs with 0
    combined_data = combined_data.fillna(0)

    print(f"DataFrame shape: {combined_data.shape}")
    
    return combined_data

def preprocess_chunk(chunk, asset_prefix=None):
    """
    Preprocess a single chunk of data (for use with large datasets).
    
    Args:
        chunk: DataFrame chunk to process
        asset_prefix: Optional asset prefix (e.g., 'btc', 'eth')
        
    Returns:
        Processed DataFrame chunk
    """
    # If timestamp is a column, convert it to datetime and set as index
    if 'timestamp' in chunk.columns:
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        chunk.set_index('timestamp', inplace=True)
    
    # Identify asset type based on column names
    if asset_prefix is None:
        if 'close_btc' in chunk.columns:
            asset_prefix = 'btc'
            suffix = '_btc'
        elif any(col.startswith('close_') for col in chunk.columns):
            # Find the first close_X column
            for col in chunk.columns:
                if col.startswith('close_') and col != 'close_btc':
                    asset_prefix = col.split('_')[1]
                    suffix = f'_{asset_prefix}'
                    break
        else:
            # Default suffix
            asset_prefix = 'unknown'
            suffix = ''
    else:
        suffix = f'_{asset_prefix}'
    
    # Calculate returns
    return_col = f"{asset_prefix}_returns"
    chunk[return_col] = chunk[f'close{suffix}'].pct_change()
    
    # Add technical indicators
    chunk = generate_technical_indicators(chunk, suffix, asset_prefix)
    
    # Create momentum features
    chunk = create_momentum_features(chunk, asset_prefix)
    
    # Fill NaN values
    momentum_cols = [col for col in chunk.columns if 'momentum' in col or 'returns' in col]
    chunk[momentum_cols] = chunk[momentum_cols].fillna(0)
    
    rsi_cols = [col for col in chunk.columns if 'rsi' in col]
    chunk[rsi_cols] = chunk[rsi_cols].fillna(50)
    
    # For other indicators, use forward fill then backward fill
    chunk = chunk.ffill().bfill()
    
    return chunk

def add_enhanced_momentum_features(df, prefix='btc', periods=[5, 14, 21]):
    """Add advanced momentum indicators for a given asset"""
    
    # Store original column names for reference
    orig_cols = df.columns.tolist()
    close = df[f'{prefix}_close']
    high = df[f'{prefix}_high']
    low = df[f'{prefix}_low']
    volume = df[f'{prefix}_volume'] if f'{prefix}_volume' in df.columns else None
    
    # RSI with multiple periods
    for period in periods:
        df[f'{prefix}_rsi_{period}'] = ta.momentum.rsi(close, window=period)
    
    # Stochastic Oscillator (high sensitivity to price reversals)
    for period in [7, 14, 21]:
        df[f'{prefix}_stoch_k_{period}'] = ta.momentum.stoch(high, low, close, window=period)
        df[f'{prefix}_stoch_d_{period}'] = ta.momentum.stoch_signal(high, low, close, window=period)
    
    # Rate of Change (pure momentum)
    for period in [3, 9, 14]:
        df[f'{prefix}_roc_{period}'] = ta.momentum.roc(close, window=period)
    
    # Williams %R (overbought/oversold)
    for period in [7, 14]:
        df[f'{prefix}_williams_{period}'] = ta.momentum.williams_r(high, low, close, lbp=period)
    
    # Commodity Channel Index (trend strength)
    for period in [14, 20]:
        df[f'{prefix}_cci_{period}'] = ta.trend.cci(high, low, close, window=period)
    
    # Awesome Oscillator (market momentum)
    df[f'{prefix}_ao'] = ta.momentum.awesome_oscillator(high, low)
    
    # Ultimate Oscillator (multi-timeframe momentum)
    df[f'{prefix}_uo'] = ta.momentum.ultimate_oscillator(high, low, close)
    
    # Relative Momentum Index (smoother version of RSI)
    df[f'{prefix}_rmi'] = (close / close.shift(10) - 1) * 100
    
    # Money Flow Index (volume-weighted RSI)
    if volume is not None:
        df[f'{prefix}_mfi'] = ta.volume.money_flow_index(high, low, close, volume)
    
    # Triple EMA oscillator (smooth momentum)
    df[f'{prefix}_trix'] = ta.trend.trix(close, window=14)
    
    # Add momentum signal changes
    for col in [c for c in df.columns if 'rsi' in c or 'stoch' in c or 'cci' in c]:
        df[f'{col}_change'] = df[col] - df[col].shift(1)
    
    # Add this to the end of each feature function to catch any infinity values:
    for col in [c for c in df.columns if c not in orig_cols]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Return names of newly added columns
    return [col for col in df.columns if col not in orig_cols]

def add_price_action_features(df, prefix='btc'):
    """Add sophisticated price action features"""
    
    orig_cols = df.columns.tolist()
    close = df[f'{prefix}_close']
    open = df[f'{prefix}_open']
    high = df[f'{prefix}_high']
    low = df[f'{prefix}_low']
    
    # Candle patterns
    df[f'{prefix}_body_size'] = abs(close - open)
    df[f'{prefix}_upper_wick'] = high - np.maximum(close, open)
    df[f'{prefix}_lower_wick'] = np.minimum(close, open) - low
    df[f'{prefix}_candle_range'] = high - low
    
    # Candle relative metrics
    df[f'{prefix}_body_percent'] = df[f'{prefix}_body_size'] / df[f'{prefix}_candle_range']
    
    # Engulfing patterns
    df[f'{prefix}_bullish_engulfing'] = (
        (open.shift(1) > close.shift(1)) &  # Previous candle is bearish
        (close > open) &                    # Current candle is bullish
        (open <= close.shift(1)) &          # Current open below prev close
        (close > open.shift(1))             # Current close above prev open
    ).astype(int)
    
    df[f'{prefix}_bearish_engulfing'] = (
        (close.shift(1) > open.shift(1)) &  # Previous candle is bullish
        (open > close) &                    # Current candle is bearish
        (open >= close.shift(1)) &          # Current open above prev close
        (close < open.shift(1))             # Current close below prev open
    ).astype(int)
    
    # Doji detection
    body_threshold = df[f'{prefix}_candle_range'] * 0.1  # 10% of range
    df[f'{prefix}_doji'] = (df[f'{prefix}_body_size'] <= body_threshold).astype(int)
    
    # Pin bars (high reversal potential)
    df[f'{prefix}_bullish_pin'] = (
        (df[f'{prefix}_lower_wick'] >= df[f'{prefix}_body_size'] * 2) & 
        (df[f'{prefix}_lower_wick'] >= df[f'{prefix}_upper_wick'] * 2)
    ).astype(int)
    
    df[f'{prefix}_bearish_pin'] = (
        (df[f'{prefix}_upper_wick'] >= df[f'{prefix}_body_size'] * 2) & 
        (df[f'{prefix}_upper_wick'] >= df[f'{prefix}_lower_wick'] * 2)
    ).astype(int)
    
    # Support/Resistance detection
    for period in [5, 10, 20]:
        # Rolling highest high and lowest low
        df[f'{prefix}_resistance_{period}'] = high.rolling(period).max()
        df[f'{prefix}_support_{period}'] = low.rolling(period).min()
        
        # Distance to resistance and support
        df[f'{prefix}_res_distance_{period}'] = (df[f'{prefix}_resistance_{period}'] - close) / close
        df[f'{prefix}_sup_distance_{period}'] = (close - df[f'{prefix}_support_{period}'] / close)
        
    # Consolidation detection
    for period in [5, 10, 20]:
        df[f'{prefix}_range_ratio_{period}'] = (
            (high.rolling(period).max() - low.rolling(period).min()) / 
            low.rolling(period).min()
        )
    
    # Breakout detection
    for period in [10, 20]:
        prev_high = high.shift(1).rolling(period).max()
        prev_low = low.shift(1).rolling(period).min()
        
        df[f'{prefix}_breakout_up_{period}'] = (high > prev_high).astype(int)
        df[f'{prefix}_breakout_down_{period}'] = (low < prev_low).astype(int)
    
    # Higher highs and lower lows (trend strength)
    df[f'{prefix}_higher_high'] = (high > high.shift(1)).astype(int)
    df[f'{prefix}_lower_low'] = (low < low.shift(1)).astype(int)
    
    for period in [5, 10]:
        df[f'{prefix}_consec_higher_highs'] = df[f'{prefix}_higher_high'].rolling(period).sum()
        df[f'{prefix}_consec_lower_lows'] = df[f'{prefix}_lower_low'].rolling(period).sum()
    
    # Add this to the end of each feature function to catch any infinity values:
    for col in [c for c in df.columns if c not in orig_cols]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Return names of newly added columns
    return [col for col in df.columns if col not in orig_cols]

def add_relationship_features(df, btc_prefix='btc', alt_prefix='doge'):
    """Add features that capture BTC-altcoin relationships"""
    
    orig_cols = df.columns.tolist()
    
    # Price ratios
    df['price_ratio'] = df[f'{alt_prefix}_close'] / df[f'{btc_prefix}_close']
    
    # Rolling correlations
    for period in [10, 30, 60]:
        df[f'btc_alt_corr_{period}'] = df[f'{btc_prefix}_returns'].rolling(period).corr(
            df[f'{alt_prefix}_returns']
        )
    
    # Beta (sensitivity of altcoin to BTC movements)
    for period in [30, 60, 100]:
        cov = df[f'{alt_prefix}_returns'].rolling(period).cov(df[f'{btc_prefix}_returns'])
        var = df[f'{btc_prefix}_returns'].rolling(period).var()
        df[f'beta_{period}'] = cov / var
    
    # Relative strength
    for period in [14, 30]:
        df[f'relative_strength_{period}'] = (
            df[f'{alt_prefix}_close'] / df[f'{alt_prefix}_close'].shift(period)
        ) / (
            df[f'{btc_prefix}_close'] / df[f'{btc_prefix}_close'].shift(period)
        )
    
    # Volume relative strength 
    if f'{alt_prefix}_volume' in df.columns and f'{btc_prefix}_volume' in df.columns:
        for period in [5, 14]:
            df[f'vol_relative_strength_{period}'] = (
                df[f'{alt_prefix}_volume'].rolling(period).mean() / 
                df[f'{alt_prefix}_volume'].shift(period).rolling(period).mean()
            ) / (
                df[f'{btc_prefix}_volume'].rolling(period).mean() / 
                df[f'{btc_prefix}_volume'].shift(period).rolling(period).mean()
            )
    
    # Volatility ratio
    for period in [10, 20]:
        btc_vol = df[f'{btc_prefix}_returns'].rolling(period).std() * np.sqrt(period)
        alt_vol = df[f'{alt_prefix}_returns'].rolling(period).std() * np.sqrt(period)
        df[f'volatility_ratio_{period}'] = np.divide(
            alt_vol, btc_vol, 
            out=np.ones_like(alt_vol), 
            where=btc_vol != 0
        )
    
    # Momentum divergence
    if f'{btc_prefix}_rsi_14' in df.columns and f'{alt_prefix}_rsi_14' in df.columns:
        df['rsi_divergence'] = df[f'{alt_prefix}_rsi_14'] - df[f'{btc_prefix}_rsi_14']
    
    # Lead-lag effectiveness
    for lag in [1, 2, 3, 5]:
        # How well does BTC predict altcoin direction
        df[f'btc_prediction_accuracy_{lag}'] = (
            (df[f'{btc_prefix}_returns'] > 0) == 
            (df[f'{alt_prefix}_returns'].shift(-lag) > 0)
        ).rolling(50).mean()
    
    # Add this to the end of each feature function to catch any infinity values:
    for col in [c for c in df.columns if c not in orig_cols]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Return names of newly added columns
    return [col for col in df.columns if col not in orig_cols]

def add_multi_timeframe_features(df, btc_prefix='btc', timeframes=[5, 15, 60]):
    """Add features from multiple timeframes"""
    
    orig_cols = df.columns.tolist()
    
    for timeframe in timeframes:
        try:
            # Resample to higher timeframe
            cols_to_resample = {}
            
            # Dynamically build resample columns based on what's available
            for col_type in ['open', 'high', 'low', 'close', 'volume']:
                col_name = f'{btc_prefix}_{col_type}'
                if col_name in df.columns:
                    agg_method = 'sum' if col_type == 'volume' else 'last' if col_type == 'close' else 'first' if col_type == 'open' else 'max' if col_type == 'high' else 'min'
                    cols_to_resample[col_name] = agg_method
            
            # Only proceed if we have essential columns
            if f'{btc_prefix}_close' in cols_to_resample:
                # Resample data
                resampled = df.resample(f'{timeframe}min').agg(cols_to_resample).dropna()
                
                # Calculate indicators on higher timeframe if we have enough data
                if len(resampled) > 50:
                    # RSI
                    resampled[f'{btc_prefix}_rsi_14_{timeframe}min'] = ta.momentum.RSIIndicator(
                        close=resampled[f'{btc_prefix}_close'], window=14
                    ).rsi()
                    
                    # SMA for trend
                    resampled[f'{btc_prefix}_sma_4_{timeframe}min'] = ta.trend.SMAIndicator(
                        close=resampled[f'{btc_prefix}_close'], window=4
                    ).sma_indicator()
                    
                    # Trend direction
                    resampled[f'{btc_prefix}_trend_{timeframe}min'] = (
                        resampled[f'{btc_prefix}_close'] > resampled[f'{btc_prefix}_sma_4_{timeframe}min']
                    ).astype(int)
                    
                    # Forward fill to original timeframe
                    for col in resampled.columns:
                        if col not in cols_to_resample:
                            # Only map new indicators back to original dataframe
                            df[col] = df.index.map(resampled[col].reindex(df.index).ffill())
        
        except Exception as e:
            print(f"Error creating {timeframe}min timeframe features: {e}")
            continue
    
    # Add this to the end of each feature function to catch any infinity values:
    for col in [c for c in df.columns if c not in orig_cols]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Return names of newly added columns
    return [col for col in df.columns if col not in orig_cols]

def add_composite_features(df, btc_prefix='btc', alt_prefix='doge'):
    """Create composite features that combine multiple indicators"""
    
    orig_cols = df.columns.tolist()
    
    # Make sure we have SMA indicators that composite features depend on
    for prefix in [btc_prefix, alt_prefix]:
        if f'{prefix}_sma_50' not in df.columns:
            if f'{prefix}_close' in df.columns:
                df[f'{prefix}_sma_50'] = ta.trend.sma_indicator(df[f'{prefix}_close'], window=50)
        
        if f'{prefix}_sma_200' not in df.columns:
            if f'{prefix}_close' in df.columns:
                df[f'{prefix}_sma_200'] = ta.trend.sma_indicator(df[f'{prefix}_close'], window=200)
    
    # Price-momentum divergence
    if f'{btc_prefix}_rsi_14' in df.columns:
        # BTC price making higher highs but RSI making lower highs
        df[f'{btc_prefix}_bearish_div'] = (
            (df[f'{btc_prefix}_close'] > df[f'{btc_prefix}_close'].shift(1)) & 
            (df[f'{btc_prefix}_rsi_14'] < df[f'{btc_prefix}_rsi_14'].shift(1))
        ).astype(int)
        
        # BTC price making lower lows but RSI making higher lows
        df[f'{btc_prefix}_bullish_div'] = (
            (df[f'{btc_prefix}_close'] < df[f'{btc_prefix}_close'].shift(1)) & 
            (df[f'{btc_prefix}_rsi_14'] > df[f'{btc_prefix}_rsi_14'].shift(1))
        ).astype(int)
    
    # Multiple timeframe momentum alignment
    if all(f'{btc_prefix}_rsi_{p}' in df.columns for p in [5, 14, 21]):
        # Bullish alignment: faster RSI > medium RSI > slow RSI
        df[f'{btc_prefix}_bullish_alignment'] = (
            (df[f'{btc_prefix}_rsi_5'] > df[f'{btc_prefix}_rsi_14']) & 
            (df[f'{btc_prefix}_rsi_14'] > df[f'{btc_prefix}_rsi_21'])
        ).astype(int)
        
        # Bearish alignment: faster RSI < medium RSI < slow RSI
        df[f'{btc_prefix}_bearish_alignment'] = (
            (df[f'{btc_prefix}_rsi_5'] < df[f'{btc_prefix}_rsi_14']) & 
            (df[f'{btc_prefix}_rsi_14'] < df[f'{btc_prefix}_rsi_21'])
        ).astype(int)
    
    # Market regime features
    if f'{btc_prefix}_volatility_20' in df.columns:
        # High volatility environment
        df[f'high_volatility_regime'] = (
            df[f'{btc_prefix}_volatility_20'] > 
            df[f'{btc_prefix}_volatility_20'].rolling(100).mean() * 1.5
        ).astype(int)
    
    # Trend strength composite
    if all(x in df.columns for x in [f'{btc_prefix}_close', f'{btc_prefix}_sma_50', f'{btc_prefix}_sma_200']):
        df[f'{btc_prefix}_trend_strength'] = (
            (df[f'{btc_prefix}_close'] - df[f'{btc_prefix}_sma_50']) / df[f'{btc_prefix}_sma_50'] +
            (df[f'{btc_prefix}_sma_50'] - df[f'{btc_prefix}_sma_200']) / df[f'{btc_prefix}_sma_200']
        )
    
    # Momentum regime shifts
    if f'{btc_prefix}_rsi_14' in df.columns:
        # Oversold to rising momentum
        df[f'{btc_prefix}_oversold_recovery'] = (
            (df[f'{btc_prefix}_rsi_14'].shift(1) < 30) &
            (df[f'{btc_prefix}_rsi_14'] > df[f'{btc_prefix}_rsi_14'].shift(1))
        ).astype(int)
        
        # Overbought to falling momentum
        df[f'{btc_prefix}_overbought_reversal'] = (
            (df[f'{btc_prefix}_rsi_14'].shift(1) > 70) &
            (df[f'{btc_prefix}_rsi_14'] < df[f'{btc_prefix}_rsi_14'].shift(1))
        ).astype(int)
    
    # Convert boolean features to integers for better ML compatibility
    bool_cols = df.select_dtypes(include='bool').columns
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # Add this to the end of each feature function to catch any infinity values:
    for col in [c for c in df.columns if c not in orig_cols]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Return names of newly added columns
    return [col for col in df.columns if col not in orig_cols]