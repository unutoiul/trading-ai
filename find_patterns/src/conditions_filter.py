"""Interactive condition filters for cryptocurrency pattern analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

class ConditionsFilter:
    """Class for filtering cryptocurrency data based on market conditions."""
    
    def __init__(self, combined_data):
        """Initialize with combined market data."""
        self.data = combined_data.copy()
        self.filtered_data = combined_data.copy()
        self.active_filters = {}
        
        # Find altcoin name from data columns
        self.alt_prefix = self._detect_altcoin_name()
        print(f"Detected altcoin: {self.alt_prefix.upper()}")
        
    def _detect_altcoin_name(self):
        """Detect which altcoin is being analyzed from data columns."""
        for col in self.data.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                return col.split('_')[0]
        
        # If not found in columns, check if stored as attribute
        if 'altcoin_name' in self.data.columns and not self.data['altcoin_name'].isna().all():
            return self.data['altcoin_name'].iloc[0]
            
        return "alt"  # Default fallback
    
    def define_standard_conditions(self, volatility_threshold=0.0015):
        """Define standard market conditions based on price movements."""
        # BTC conditions
        self.data['btc_strong_up'] = (self.data['btc_returns'] > volatility_threshold)
        self.data['btc_medium_up'] = (self.data['btc_returns'] > volatility_threshold/2) & (self.data['btc_returns'] <= volatility_threshold)
        self.data['btc_small_up'] = (self.data['btc_returns'] > 0) & (self.data['btc_returns'] <= volatility_threshold/2)
        self.data['btc_small_down'] = (self.data['btc_returns'] < 0) & (self.data['btc_returns'] >= -volatility_threshold/2)
        self.data['btc_medium_down'] = (self.data['btc_returns'] < -volatility_threshold/2) & (self.data['btc_returns'] >= -volatility_threshold)
        self.data['btc_strong_down'] = (self.data['btc_returns'] < -volatility_threshold)
        self.data['btc_sideways'] = (abs(self.data['btc_returns']) < volatility_threshold/4)
        
        # Altcoin conditions
        self.data[f'{self.alt_prefix}_strong_up'] = (self.data[f'{self.alt_prefix}_returns'] > volatility_threshold)
        self.data[f'{self.alt_prefix}_medium_up'] = (self.data[f'{self.alt_prefix}_returns'] > volatility_threshold/2) & (self.data[f'{self.alt_prefix}_returns'] <= volatility_threshold)
        self.data[f'{self.alt_prefix}_small_up'] = (self.data[f'{self.alt_prefix}_returns'] > 0) & (self.data[f'{self.alt_prefix}_returns'] <= volatility_threshold/2)
        self.data[f'{self.alt_prefix}_small_down'] = (self.data[f'{self.alt_prefix}_returns'] < 0) & (self.data[f'{self.alt_prefix}_returns'] >= -volatility_threshold/2)
        self.data[f'{self.alt_prefix}_medium_down'] = (self.data[f'{self.alt_prefix}_returns'] < -volatility_threshold/2) & (self.data[f'{self.alt_prefix}_returns'] >= -volatility_threshold)
        self.data[f'{self.alt_prefix}_strong_down'] = (self.data[f'{self.alt_prefix}_returns'] < -volatility_threshold)
        self.data[f'{self.alt_prefix}_sideways'] = (abs(self.data[f'{self.alt_prefix}_returns']) < volatility_threshold/4)
        
        # Also define volatility conditions
        self.data['btc_high_volatility'] = (self.data['btc_volatility_15'] > self.data['btc_volatility_15'].rolling(100, min_periods=20).mean() * 1.5)
        self.data['btc_low_volatility'] = (self.data['btc_volatility_15'] < self.data['btc_volatility_15'].rolling(100, min_periods=20).mean() * 0.5)
        
        return self.data.filter(regex='^(btc|{})_(strong|medium|small|sideways|high_volatility|low_volatility)'.format(self.alt_prefix)).columns.tolist()
    
    def apply_filters(self, conditions=None, condition_type='any'):
        """
        Apply selected condition filters to data.
        
        Args:
            conditions: List of condition names to filter by
            condition_type: 'any' or 'all' - whether any or all conditions must be met
            
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
                    if condition_type == 'any':
                        filter_mask = filter_mask | self.data[condition]
                    else:  # 'all'
                        filter_mask = filter_mask & self.data[condition]
            else:
                print(f"Warning: Condition '{condition}' not found in data")
        
        if filter_mask is not None:
            self.filtered_data = self.data[filter_mask]
            self.active_filters = {
                'conditions': conditions,
                'condition_type': condition_type,
                'row_count': len(self.filtered_data),
                'percentage': len(self.filtered_data) / len(self.data) * 100
            }
            print(f"Applied {len(conditions)} filters ({condition_type}). Retained {self.active_filters['row_count']} rows ({self.active_filters['percentage']:.2f}% of total).")
        else:
            # If no valid filters, keep all data
            self.filtered_data = self.data.copy()
            self.active_filters = {}
            print("No valid filters provided. Using full dataset.")
        
        return self.filtered_data
    
    def create_interactive_widget(self, jupyter=True):
        """
        Create interactive filter widget for Jupyter notebook.
        Only works in Jupyter/IPython environment.
        """
        if not jupyter:
            print("Interactive widgets only available in Jupyter notebooks.")
            return None
            
        try:
            import ipywidgets as widgets
            from IPython.display import display
            
            # Get all condition columns
            condition_columns = [col for col in self.data.columns if 
                                col.endswith(('_up', '_down', '_sideways', '_volatility'))]
            
            # Create checkboxes for each condition
            checkboxes = {col: widgets.Checkbox(value=False, description=col, disabled=False) for col in condition_columns}
            
            # Radio button for ANY/ALL
            condition_type = widgets.RadioButtons(
                options=['any', 'all'],
                value='any',
                description='Match:',
                disabled=False
            )
            
            # Button to apply filters
            apply_button = widgets.Button(description='Apply Filters')
            reset_button = widgets.Button(description='Reset Filters')
            
            # Output area for results
            output = widgets.Output()
            
            # Layout for checkboxes
            btc_up_box = widgets.VBox([checkboxes[col] for col in condition_columns if 'btc' in col and '_up' in col])
            btc_down_box = widgets.VBox([checkboxes[col] for col in condition_columns if 'btc' in col and '_down' in col])
            btc_other_box = widgets.VBox([checkboxes[col] for col in condition_columns if 'btc' in col and not ('_up' in col or '_down' in col)])
            
            alt_up_box = widgets.VBox([checkboxes[col] for col in condition_columns if self.alt_prefix in col and '_up' in col])
            alt_down_box = widgets.VBox([checkboxes[col] for col in condition_columns if self.alt_prefix in col and '_down' in col])
            alt_other_box = widgets.VBox([checkboxes[col] for col in condition_columns if self.alt_prefix in col and not ('_up' in col or '_down' in col)])
            
            # Organize into tabs
            btc_tab = widgets.Tab(children=[btc_up_box, btc_down_box, btc_other_box])
            btc_tab.set_title(0, 'BTC Up')
            btc_tab.set_title(1, 'BTC Down')
            btc_tab.set_title(2, 'BTC Other')
            
            alt_tab = widgets.Tab(children=[alt_up_box, alt_down_box, alt_other_box])
            alt_tab.set_title(0, f'{self.alt_prefix.upper()} Up')
            alt_tab.set_title(1, f'{self.alt_prefix.upper()} Down')
            alt_tab.set_title(2, f'{self.alt_prefix.upper()} Other')
            
            # Main tab layout
            main_tab = widgets.Tab(children=[btc_tab, alt_tab])
            main_tab.set_title(0, 'BTC Conditions')
            main_tab.set_title(1, f'{self.alt_prefix.upper()} Conditions')
            
            # Define button actions
            def on_apply_button_clicked(b):
                with output:
                    output.clear_output()
                    selected_conditions = [col for col, checkbox in checkboxes.items() if checkbox.value]
                    self.apply_filters(selected_conditions, condition_type.value)
                    print(f"Applied {len(selected_conditions)} filters. Showing {len(self.filtered_data)} data points.")
                    
                    # Display summary of filtered data
                    if len(self.filtered_data) > 0:
                        print("\nAltcoin returns summary in filtered data:")
                        returns_col = f"{self.alt_prefix}_returns"
                        print(f"Mean return: {self.filtered_data[returns_col].mean()*100:.4f}%")
                        print(f"Win rate: {(self.filtered_data[returns_col] > 0).mean()*100:.2f}%")
                        
                        # Create histogram of returns
                        plt.figure(figsize=(8, 5))
                        sns.histplot(self.filtered_data[returns_col]*100, kde=True)
                        plt.axvline(x=0, color='r', linestyle='--')
                        plt.title(f"{self.alt_prefix.upper()} Returns Distribution (Filtered Data)")
                        plt.xlabel("Returns (%)")
                        plt.tight_layout()
                        plt.show()
            
            def on_reset_button_clicked(b):
                with output:
                    output.clear_output()
                    # Reset all checkboxes
                    for checkbox in checkboxes.values():
                        checkbox.value = False
                    # Reset data filter
                    self.filtered_data = self.data.copy()
                    self.active_filters = {}
                    print("Filters reset. Using full dataset.")
            
            # Connect buttons to actions
            apply_button.on_click(on_apply_button_clicked)
            reset_button.on_click(on_reset_button_clicked)
            
            # Put everything together
            button_box = widgets.HBox([apply_button, reset_button, condition_type])
            
            # Return the full widget for display
            return widgets.VBox([
                main_tab,
                button_box,
                output
            ])
            
        except ImportError:
            print("ipywidgets not available. Please install with: pip install ipywidgets")
            return None
        
    def plot_filtered_data_stats(self, lags=[1, 3, 5, 10, 15]):
        """Plot statistics for filtered data at different lags."""
        if len(self.filtered_data) < 10:
            print("Not enough data points to analyze after filtering.")
            return
            
        returns_col = f"{self.alt_prefix}_returns"
        
        # Calculate lag statistics
        lag_stats = {}
        for lag in lags:
            # Forward returns at each lag
            self.filtered_data[f'forward_return_{lag}'] = self.filtered_data[returns_col].shift(-lag)
            
            # Calculate statistics
            lag_stats[lag] = {
                'mean_return': self.filtered_data[f'forward_return_{lag}'].mean() * 100,
                'win_rate': (self.filtered_data[f'forward_return_{lag}'] > 0).mean() * 100,
                'sample_size': self.filtered_data[f'forward_return_{lag}'].count()
            }
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean returns at each lag
        ax1.plot([lag for lag in lag_stats], [lag_stats[lag]['mean_return'] for lag in lag_stats], 'o-', linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Mean Return at Different Lags')
        ax1.set_xlabel('Lag (periods)')
        ax1.set_ylabel('Mean Return (%)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Win rate at each lag
        ax2.plot([lag for lag in lag_stats], [lag_stats[lag]['win_rate'] for lag in lag_stats], 'o-', linewidth=2)
        ax2.axhline(y=50, color='r', linestyle='--')
        ax2.set_title('Win Rate at Different Lags')
        ax2.set_xlabel('Lag (periods)')
        ax2.set_ylabel('Win Rate (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\nLag Statistics:")
        for lag, stats in lag_stats.items():
            print(f"Lag {lag}: Mean Return = {stats['mean_return']:.4f}%, Win Rate = {stats['win_rate']:.2f}%, Sample Size = {stats['sample_size']}")
        
        return lag_stats
        
    def get_filtered_data(self):
        """Return the current filtered data."""
        return self.filtered_data