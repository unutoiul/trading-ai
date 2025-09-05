"""
Volatility Breakout Detection System
Detects meaningful breakouts and drops over multi-timeframe periods (1-10 minutes)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class VolatilityBreakoutDetector:
    """Detect volatility breakouts and drops over multiple timeframes."""
    
    def __init__(self, data: pd.DataFrame, asset_prefix: str = 'btc'):
        """
        Initialize the breakout detector.
        
        Args:
            data: DataFrame with OHLCV data
            asset_prefix: Prefix for the asset columns (e.g., 'btc', 'doge')
        """
        self.data = data.copy()
        self.asset_prefix = asset_prefix
        self.close_col = f'{asset_prefix}_close'
        self.high_col = f'{asset_prefix}_high'
        self.low_col = f'{asset_prefix}_low'
        self.volume_col = f'{asset_prefix}_volume'
        
        # Verify required columns exist
        required_cols = [self.close_col, self.high_col, self.low_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def calculate_multi_timeframe_features(self, timeframes: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) -> pd.DataFrame:
        """
        Calculate volatility and price movement features across multiple timeframes (1-10 minutes).
        
        Args:
            timeframes: List of timeframes in minutes to analyze (default: 1-10 minutes)
            
        Returns:
            DataFrame with multi-timeframe features added
        """
        df = self.data.copy()
        
        for tf in timeframes:
            # Rolling price changes over timeframe
            df[f'{self.asset_prefix}_return_{tf}m'] = df[self.close_col].pct_change(tf)
            
            # Rolling high-low volatility over timeframe
            df[f'{self.asset_prefix}_hl_volatility_{tf}m'] = (
                df[self.high_col].rolling(tf).max() - df[self.low_col].rolling(tf).min()
            ) / df[self.close_col].rolling(tf).mean()
            
            # Rolling standard deviation (true volatility)
            df[f'{self.asset_prefix}_volatility_{tf}m'] = df[self.close_col].pct_change().rolling(tf).std()
            
            # Price range expansion
            df[f'{self.asset_prefix}_range_{tf}m'] = (
                df[self.high_col].rolling(tf).max() - df[self.low_col].rolling(tf).min()
            ) / df[self.close_col]
            
            # Volume-weighted price movement
            if self.volume_col in df.columns:
                price_volume = df[self.close_col].pct_change() * df[self.volume_col]
                df[f'{self.asset_prefix}_vol_weighted_move_{tf}m'] = price_volume.rolling(tf).mean()
        
        return df
    
    def detect_volatility_breakouts(self, 
                                  lookback_period: int = 50, 
                                  breakout_multiplier: float = 2.0,
                                  timeframes: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) -> Dict[str, pd.Series]:
        """
        Detect volatility breakouts when current volatility exceeds historical averages.
        
        Args:
            lookback_period: Number of periods to look back for average volatility
            breakout_multiplier: Multiplier for determining breakout threshold
            timeframes: Timeframes to analyze
            
        Returns:
            Dictionary of boolean Series indicating breakout conditions
        """
        df = self.calculate_multi_timeframe_features(timeframes)
        breakout_conditions = {}
        
        for tf in timeframes:
            vol_col = f'{self.asset_prefix}_volatility_{tf}m'
            hl_vol_col = f'{self.asset_prefix}_hl_volatility_{tf}m'
            
            # Volatility breakout (current vol > historical average * multiplier)
            vol_avg = df[vol_col].rolling(lookback_period, min_periods=10).mean()
            breakout_conditions[f'{self.asset_prefix}_vol_breakout_{tf}m'] = (
                df[vol_col] > vol_avg * breakout_multiplier
            )
            
            # High-Low range breakout
            hl_avg = df[hl_vol_col].rolling(lookback_period, min_periods=10).mean()
            breakout_conditions[f'{self.asset_prefix}_hl_breakout_{tf}m'] = (
                df[hl_vol_col] > hl_avg * breakout_multiplier
            )
        
        return breakout_conditions
    
    def detect_price_breakouts(self, 
                             lookback_period: int = 20,
                             breakout_threshold: float = 0.02,
                             timeframes: List[int] = [1, 2, 3, 5, 10]) -> Dict[str, pd.Series]:
        """
        Detect price breakouts when price moves significantly beyond recent ranges.
        
        Args:
            lookback_period: Period for calculating support/resistance levels
            breakout_threshold: Minimum percentage move to qualify as breakout
            timeframes: Timeframes to analyze
            
        Returns:
            Dictionary of boolean Series indicating breakout conditions
        """
        df = self.calculate_multi_timeframe_features(timeframes)
        breakout_conditions = {}
        
        for tf in timeframes:
            return_col = f'{self.asset_prefix}_return_{tf}m'
            
            # Calculate rolling support and resistance levels
            rolling_high = df[self.close_col].rolling(lookback_period).max()
            rolling_low = df[self.close_col].rolling(lookback_period).min()
            rolling_range = rolling_high - rolling_low
            
            # Upward breakout: price moves above recent high + threshold
            breakout_conditions[f'{self.asset_prefix}_price_breakout_up_{tf}m'] = (
                (df[self.close_col] > rolling_high) & 
                (df[return_col] > breakout_threshold)
            )
            
            # Downward breakout: price moves below recent low - threshold  
            breakout_conditions[f'{self.asset_prefix}_price_breakout_down_{tf}m'] = (
                (df[self.close_col] < rolling_low) & 
                (df[return_col] < -breakout_threshold)
            )
            
            # Strong upward momentum (without necessarily breaking resistance)
            breakout_conditions[f'{self.asset_prefix}_strong_up_{tf}m'] = (
                df[return_col] > breakout_threshold
            )
            
            # Medium upward momentum (half of strong threshold)
            medium_threshold = breakout_threshold * 0.5
            breakout_conditions[f'{self.asset_prefix}_medium_up_{tf}m'] = (
                (df[return_col] > medium_threshold) & (df[return_col] <= breakout_threshold)
            )
            
            # Small upward momentum (quarter of strong threshold)
            small_threshold = breakout_threshold * 0.25
            breakout_conditions[f'{self.asset_prefix}_small_up_{tf}m'] = (
                (df[return_col] > small_threshold) & (df[return_col] <= medium_threshold)
            )
            
            # Strong downward momentum (dropping pattern)
            breakout_conditions[f'{self.asset_prefix}_strong_down_{tf}m'] = (
                df[return_col] < -breakout_threshold
            )
            
            # Medium downward momentum
            breakout_conditions[f'{self.asset_prefix}_medium_down_{tf}m'] = (
                (df[return_col] < -medium_threshold) & (df[return_col] >= -breakout_threshold)
            )
            
            # Small downward momentum
            breakout_conditions[f'{self.asset_prefix}_small_down_{tf}m'] = (
                (df[return_col] < -small_threshold) & (df[return_col] >= -medium_threshold)
            )
        
        return breakout_conditions
    
    def detect_sustained_moves(self, 
                             consecutive_periods: int = 3,
                             min_move_threshold: float = 0.005,
                             timeframes: List[int] = [1, 2, 3, 5]) -> Dict[str, pd.Series]:
        """
        Detect sustained price movements over consecutive periods.
        
        Args:
            consecutive_periods: Number of consecutive periods required
            min_move_threshold: Minimum move per period to qualify
            timeframes: Timeframes to analyze
            
        Returns:
            Dictionary of boolean Series indicating sustained movement conditions
        """
        df = self.calculate_multi_timeframe_features(timeframes)
        sustained_conditions = {}
        
        for tf in timeframes:
            return_col = f'{self.asset_prefix}_return_{tf}m'
            
            # Sustained upward movement
            up_moves = df[return_col] > min_move_threshold
            sustained_conditions[f'{self.asset_prefix}_sustained_up_{tf}m'] = (
                up_moves.rolling(consecutive_periods).sum() == consecutive_periods
            )
            
            # Sustained downward movement (dropping pattern)
            down_moves = df[return_col] < -min_move_threshold
            sustained_conditions[f'{self.asset_prefix}_sustained_down_{tf}m'] = (
                down_moves.rolling(consecutive_periods).sum() == consecutive_periods
            )
        
        return sustained_conditions
    
    def detect_volume_breakouts(self, 
                              volume_multiplier: float = 2.0,
                              lookback_period: int = 20,
                              timeframes: List[int] = [1, 2, 3, 5, 10]) -> Dict[str, pd.Series]:
        """
        Detect volume breakouts accompanying price movements.
        
        Args:
            volume_multiplier: Volume must be this multiple of average
            lookback_period: Period for calculating average volume
            timeframes: Timeframes to analyze
            
        Returns:
            Dictionary of boolean Series indicating volume breakout conditions
        """
        if self.volume_col not in self.data.columns:
            return {}
            
        df = self.calculate_multi_timeframe_features(timeframes)
        volume_conditions = {}
        
        # Average volume over lookback period
        avg_volume = df[self.volume_col].rolling(lookback_period, min_periods=5).mean()
        
        for tf in timeframes:
            return_col = f'{self.asset_prefix}_return_{tf}m'
            vol_weighted_col = f'{self.asset_prefix}_vol_weighted_move_{tf}m'
            
            # High volume + upward breakout
            high_volume = df[self.volume_col] > avg_volume * volume_multiplier
            volume_conditions[f'{self.asset_prefix}_volume_breakout_up_{tf}m'] = (
                high_volume & (df[return_col] > 0.01)  # 1% move with high volume
            )
            
            # High volume + downward breakout (dropping with volume)
            volume_conditions[f'{self.asset_prefix}_volume_breakout_down_{tf}m'] = (
                high_volume & (df[return_col] < -0.01)  # -1% move with high volume
            )
        
        return volume_conditions
    
    def generate_all_conditions(self, 
                              timeframes: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              **kwargs) -> pd.DataFrame:
        """
        Generate all breakout and pattern conditions for 1-10 minute timeframes.
        
        Args:
            timeframes: Timeframes to analyze (default: 1-10 minutes)
            **kwargs: Additional parameters for detection methods
            
        Returns:
            DataFrame with original data plus all condition columns
        """
        df = self.calculate_multi_timeframe_features(timeframes)
        
        # Get all condition types
        vol_breakouts = self.detect_volatility_breakouts(timeframes=timeframes, **kwargs)
        price_breakouts = self.detect_price_breakouts(timeframes=timeframes, **kwargs)
        sustained_moves = self.detect_sustained_moves(timeframes=timeframes, **kwargs)
        volume_breakouts = self.detect_volume_breakouts(timeframes=timeframes, **kwargs)
        
        # Add all conditions to dataframe
        all_conditions = {**vol_breakouts, **price_breakouts, **sustained_moves, **volume_breakouts}
        
        for condition_name, condition_series in all_conditions.items():
            df[condition_name] = condition_series.fillna(False)
        
        return df
    
    def get_condition_summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get a summary of all detected conditions.
        
        Args:
            df: DataFrame with conditions (if None, generates all conditions)
            
        Returns:
            DataFrame with condition statistics
        """
        if df is None:
            df = self.generate_all_conditions()
        
        # Find all condition columns
        condition_cols = [col for col in df.columns if 
                         any(pattern in col for pattern in ['_breakout_', '_strong_', '_sustained_'])]
        
        summary_data = []
        for col in condition_cols:
            if col in df.columns:
                true_count = df[col].sum()
                total_count = df[col].count()
                percentage = (true_count / total_count * 100) if total_count > 0 else 0
                
                summary_data.append({
                    'condition': col,
                    'occurrences': true_count,
                    'total_periods': total_count,
                    'percentage': percentage,
                    'timeframe': col.split('_')[-1] if '_' in col else 'unknown'
                })
        
        return pd.DataFrame(summary_data).sort_values('percentage', ascending=False)


def create_breakout_detector_for_asset(data: pd.DataFrame, asset_name: str) -> VolatilityBreakoutDetector:
    """
    Convenience function to create a breakout detector for a specific asset.
    
    Args:
        data: DataFrame with market data
        asset_name: Name of the asset (e.g., 'btc', 'doge')
        
    Returns:
        Configured VolatilityBreakoutDetector instance
    """
    return VolatilityBreakoutDetector(data, asset_name)


# Example usage and testing
if __name__ == "__main__":
    # This would be used with actual market data
    print("Volatility Breakout Detector - Ready for use with market data")
    print("Usage:")
    print("1. detector = VolatilityBreakoutDetector(data, 'btc')")
    print("2. conditions_df = detector.generate_all_conditions()")
    print("3. summary = detector.get_condition_summary(conditions_df)")
