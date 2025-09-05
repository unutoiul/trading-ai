"""
Multi-timeframe volatility breakout filters for cryptocurrency pattern analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .volatility_breakout_detector import VolatilityBreakoutDetector

class ConditionsFilter:
    """Class for filtering cryptocurrency data based on multi-timeframe breakout conditions."""
    
    def __init__(self, combined_data, alt_prefix):
        """Initialize with combined market data and altcoin prefix."""
        self.data = combined_data.copy()
        self.filtered_data = combined_data.copy()
        self.active_filters = {}
        self.alt_prefix = alt_prefix.lower()
        self.breakout_detectors = {}
        print(f"Using altcoin: {self.alt_prefix.upper()}")
        
        # Initialize breakout detectors
        self._setup_breakout_detectors()
    
    def _setup_breakout_detectors(self):
        """Setup volatility breakout detectors for available assets."""
        # Setup BTC detector
        btc_cols = ['btc_close', 'close_btc']
        btc_close_col = None
        for col in btc_cols:
            if col in self.data.columns:
                btc_close_col = col
                break
        
        if btc_close_col:
            btc_data = self.data.copy()
            if btc_close_col == 'close_btc':
                btc_data = btc_data.rename(columns={
                    'close_btc': 'btc_close',
                    'high_btc': 'btc_high',
                    'low_btc': 'btc_low',
                    'volume_btc': 'btc_volume'
                })
            
            try:
                self.breakout_detectors['btc'] = VolatilityBreakoutDetector(btc_data, 'btc')
            except Exception as e:
                print(f"Warning: Could not create BTC breakout detector: {e}")
        
        # Setup altcoin detector
        alt_cols = [f'{self.alt_prefix}_close', f'close_{self.alt_prefix}']
        alt_close_col = None
        for col in alt_cols:
            if col in self.data.columns:
                alt_close_col = col
                break
        
        if alt_close_col:
            alt_data = self.data.copy()
            if alt_close_col == f'close_{self.alt_prefix}':
                alt_data = alt_data.rename(columns={
                    f'close_{self.alt_prefix}': f'{self.alt_prefix}_close',
                    f'high_{self.alt_prefix}': f'{self.alt_prefix}_high',
                    f'low_{self.alt_prefix}': f'{self.alt_prefix}_low',
                    f'volume_{self.alt_prefix}': f'{self.alt_prefix}_volume'
                })
            
            try:
                self.breakout_detectors[self.alt_prefix] = VolatilityBreakoutDetector(alt_data, self.alt_prefix)
            except Exception as e:
                print(f"Warning: Could not create {self.alt_prefix.upper()} breakout detector: {e}")
    
    def define_standard_conditions(self, volatility_threshold=0.0015):
        """
        Define multi-timeframe breakout conditions using volatility breakout detectors.
        
        Args:
            volatility_threshold: Base threshold for breakout detection
            
        Returns:
            List of condition column names
        """
        timeframes = [1, 2, 3, 5, 10]  # Multi-timeframe analysis
        condition_columns = []
        
        # Generate conditions for each available asset
        for asset_name, detector in self.breakout_detectors.items():
            try:
                # Generate all breakout conditions
                conditions_df = detector.generate_all_conditions(
                    timeframes=timeframes
                )
                
                # Add all new condition columns to our data
                for col in conditions_df.columns:
                    if any(pattern in col for pattern in ['_breakout_', '_strong_', '_sustained_']):
                        self.data[col] = conditions_df[col].fillna(False)
                        condition_columns.append(col)
                        
                # Add feature columns (returns, volatility) for analysis
                for col in conditions_df.columns:
                    if any(pattern in col for pattern in ['_return_', '_volatility_', '_range_']):
                        if col not in self.data.columns:
                            self.data[col] = conditions_df[col]
                
            except Exception as e:
                print(f"Warning: Could not generate conditions for {asset_name.upper()}: {e}")
                # Fallback to legacy conditions if breakout detector fails
                self._add_legacy_conditions(asset_name, volatility_threshold)
        
        # Add cross-asset conditions if multiple assets available
        if len(self.breakout_detectors) > 1:
            self._add_cross_asset_conditions(timeframes)
        
        # Add summary conditions for easier filtering
        self._add_summary_conditions()
        
        # Update condition columns list
        all_conditions = [col for col in self.data.columns if 
                         any(pattern in col for pattern in ['_breakout_', '_strong_', '_sustained_', '_up', '_down'])]
        
        print(f"Generated {len(all_conditions)} breakout conditions across {len(timeframes)} timeframes")
        return all_conditions
    
    def _add_legacy_conditions(self, asset_name, volatility_threshold):
        """Add legacy conditions if breakout detector fails."""
        returns_col = f'{asset_name}_returns'
        if returns_col in self.data.columns:
            self.data[f'{asset_name}_strong_up'] = (self.data[returns_col] > volatility_threshold)
            self.data[f'{asset_name}_medium_up'] = (
                (self.data[returns_col] > volatility_threshold/2) & 
                (self.data[returns_col] <= volatility_threshold)
            )
            self.data[f'{asset_name}_small_up'] = (
                (self.data[returns_col] > 0) & 
                (self.data[returns_col] <= volatility_threshold/2)
            )
            self.data[f'{asset_name}_small_down'] = (
                (self.data[returns_col] < 0) & 
                (self.data[returns_col] >= -volatility_threshold/2)
            )
            self.data[f'{asset_name}_medium_down'] = (
                (self.data[returns_col] < -volatility_threshold/2) & 
                (self.data[returns_col] >= -volatility_threshold)
            )
            self.data[f'{asset_name}_strong_down'] = (self.data[returns_col] < -volatility_threshold)
    
    def _add_cross_asset_conditions(self, timeframes):
        """Add conditions involving multiple assets."""
        try:
            assets = list(self.breakout_detectors.keys())
            if len(assets) >= 2:
                asset1, asset2 = assets[0], assets[1]
                
                for tf in timeframes:
                    # Same direction breakouts
                    up1 = f'{asset1}_strong_up_{tf}m'
                    up2 = f'{asset2}_strong_up_{tf}m'
                    down1 = f'{asset1}_strong_down_{tf}m'
                    down2 = f'{asset2}_strong_down_{tf}m'
                    
                    if all(col in self.data.columns for col in [up1, up2, down1, down2]):
                        self.data[f'both_breakout_up_{tf}m'] = self.data[up1] & self.data[up2]
                        self.data[f'both_breakout_down_{tf}m'] = self.data[down1] & self.data[down2]
                        self.data[f'opposite_breakouts_{tf}m'] = (
                            (self.data[up1] & self.data[down2]) |
                            (self.data[down1] & self.data[up2])
                        )
        except Exception as e:
            print(f"Warning: Could not create cross-asset conditions: {e}")
    
    def _add_summary_conditions(self):
        """Add summary conditions across timeframes."""
        try:
            for asset_name in self.breakout_detectors.keys():
                # Any upward breakout across timeframes
                up_cols = [col for col in self.data.columns if 
                          asset_name in col and 'up' in col and ('breakout' in col or 'strong' in col)]
                if up_cols:
                    self.data[f'{asset_name}_any_breakout_up'] = (
                        self.data[up_cols].any(axis=1)
                    )
                
                # Any downward breakout across timeframes
                down_cols = [col for col in self.data.columns if 
                           asset_name in col and 'down' in col and ('breakout' in col or 'strong' in col)]
                if down_cols:
                    self.data[f'{asset_name}_any_breakout_down'] = (
                        self.data[down_cols].any(axis=1)
                    )
                
                # High volatility periods
                vol_cols = [col for col in self.data.columns if 
                          asset_name in col and 'volatility' in col and 'breakout' in col]
                if vol_cols:
                    self.data[f'{asset_name}_high_volatility_period'] = (
                        self.data[vol_cols].any(axis=1)
                    )
                    
                # Sustained movements (3+ consecutive periods)
                sustained_up_cols = [col for col in self.data.columns if 
                                   asset_name in col and 'sustained_up' in col]
                if sustained_up_cols:
                    self.data[f'{asset_name}_any_sustained_up'] = (
                        self.data[sustained_up_cols].any(axis=1)
                    )
                
                sustained_down_cols = [col for col in self.data.columns if 
                                     asset_name in col and 'sustained_down' in col]
                if sustained_down_cols:
                    self.data[f'{asset_name}_any_sustained_down'] = (
                        self.data[sustained_down_cols].any(axis=1)
                    )
                    
        except Exception as e:
            print(f"Warning: Could not create summary conditions: {e}")
    
    def apply_filters(self, conditions=None):
        """
        Apply selected condition filters to data.
        
        Args:
            conditions: List of condition names to filter by
            
        Returns:
            Filtered DataFrame
        """
        if not conditions:
            # Reset to original data if no filters
            self.filtered_data = self.data.copy()
            self.active_filters = {}
            print("Filters reset. Using full dataset.")
            return self.filtered_data
        
        if isinstance(conditions, str):
            conditions = [conditions]  # Convert single condition to list
            
        filter_mask = None
        
        for condition in conditions:
            if condition in self.data.columns:
                if filter_mask is None:
                    filter_mask = self.data[condition]
                else:
                    filter_mask = filter_mask | self.data[condition]  # Always use 'any' logic
            else:
                print(f"Warning: Condition '{condition}' not found in data")
        
        if filter_mask is not None:
            self.filtered_data = self.data[filter_mask]
            self.active_filters = {
                'conditions': conditions,
                'row_count': len(self.filtered_data),
                'percentage': len(self.filtered_data) / len(self.data) * 100
            }
            print(f"Applied {len(conditions)} filters. Retained {self.active_filters['row_count']} rows ({self.active_filters['percentage']:.2f}% of total).")
        else:
            # If no valid filters, keep all data
            self.filtered_data = self.data.copy()
            self.active_filters = {}
            print("No valid filters provided. Using full dataset.")
        
        return self.filtered_data
    
    def get_breakout_summary(self):
        """Get summary statistics of breakout conditions."""
        breakout_cols = [col for col in self.data.columns if 
                        any(pattern in col for pattern in ['_breakout_', '_strong_', '_sustained_'])]
        
        if not breakout_cols:
            print("No breakout conditions found. Run define_standard_conditions() first.")
            return None
        
        summary_data = []
        for col in breakout_cols:
            true_count = self.data[col].sum()
            total_count = self.data[col].count()
            percentage = (true_count / total_count * 100) if total_count > 0 else 0
            
            # Extract timeframe and asset from column name
            parts = col.split('_')
            timeframe = 'N/A'
            asset = 'unknown'
            direction = 'unknown'
            
            if 'btc' in col:
                asset = 'BTC'
            elif self.alt_prefix in col:
                asset = self.alt_prefix.upper()
            
            if 'up' in col:
                direction = 'UP'
            elif 'down' in col:
                direction = 'DOWN'
            
            for part in parts:
                if part.endswith('m'):
                    timeframe = part
                    break
            
            summary_data.append({
                'condition': col,
                'asset': asset,
                'direction': direction,
                'timeframe': timeframe,
                'occurrences': true_count,
                'percentage': percentage
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('percentage', ascending=False)
        return summary_df
    
    def get_filtered_data(self):
        """Return the current filtered data."""
        return self.filtered_data


# Legacy function for backward compatibility
def define_standard_conditions(data, volatility_threshold=0.0015):
    """
    Legacy function for backward compatibility.
    Use ConditionsFilter class for new implementations.
    """
    print("Warning: Using legacy function. Consider using ConditionsFilter class for better features.")
    
    conditions = {}
    
    # Basic volatility conditions
    if 'btc_returns' in data.columns:
        conditions['btc_positive_momentum'] = data['btc_returns'] > volatility_threshold
        conditions['btc_negative_momentum'] = data['btc_returns'] < -volatility_threshold
    
    if 'doge_returns' in data.columns:
        conditions['doge_positive_momentum'] = data['doge_returns'] > volatility_threshold
        conditions['doge_negative_momentum'] = data['doge_returns'] < -volatility_threshold
    
    return conditions
