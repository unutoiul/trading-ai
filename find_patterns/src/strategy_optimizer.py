"""Optimize trading strategy parameters using advanced ML techniques."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add seaborn import
import seaborn as sns

# Add vectorbt import
import vectorbt as vbt



class StrategyOptimizer:
    """Class to optimize trading strategy parameters using machine learning."""
    
    def __init__(self, data, pattern_stats):
        """Initialize with market data and pattern statistics."""
        self.data = data
        self.pattern_stats = pattern_stats
        self.results = None
        self.best_params = None
        self.performance = None
        self.backtest_results = None
        
        # Default parameter space
        self.param_space = {
            'entry_threshold': [0.0001, 0.0002, 0.0005, 0.001, 0.002],
            'stop_loss_pct': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0],
            'take_profit_pct': [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
            'max_holding_time': [5, 10, 15, 30, 60, 120, 240, 360, 720],
            'pattern_lag': [1, 2, 3, 5, 10, 15, 20],
            'use_pattern': list(pattern_stats.keys()) if pattern_stats else [],
            'position_size_pct': [1, 2, 5, 10, 15, 20, 25, 50, 75, 100]
        }
        
        # Find strongest patterns
        self.strongest_patterns = self._identify_strongest_patterns()
        
    def _identify_strongest_patterns(self):
        """Identify the strongest patterns based on pattern statistics."""
        pattern_strength = {}
        
        for pattern, stats in self.pattern_stats.items():
            if isinstance(stats, dict) and 'avg_return' in stats and 'win_rate' in stats:
                # Calculate a composite score: magnitude of avg_return * win_rate * sqrt(instances)
                score = abs(stats['avg_return']) * stats['win_rate'] * np.sqrt(stats['instances'])
                pattern_strength[pattern] = score
        
        # Sort patterns by score and return top 3
        sorted_patterns = sorted(pattern_strength.items(), key=lambda x: abs(x[1]), reverse=True)
        return [p[0] for p in sorted_patterns[:3]]
    
    def backtest_strategy(self, params):
        """
        Backtest a trading strategy using VectorBT.
        
        Args:
            params: Dictionary of strategy parameters
            
        Returns:
            dict: Performance metrics
        """
        try:
            # Extract parameters
            entry_threshold = params['entry_threshold']
            stop_loss_pct = params['stop_loss_pct'] / 100
            take_profit_pct = params['take_profit_pct'] / 100
            max_holding_time = params['max_holding_time']
            pattern_lag = params['pattern_lag']
            use_pattern = params['use_pattern']
            position_size_pct = params['position_size_pct'] / 100
            
            # Copy data to avoid modifying original
            backtest_data = self.data.copy()
            
            # Find the right column names using a more robust approach for any altcoin
            close_column = None
            btc_column = None
            altcoin_column = None
            
            # Check column names to identify the altcoin price column
            for col in backtest_data.columns:
                col_lower = col.lower()
                
                # Skip if not a price column
                if 'close' not in col_lower:
                    continue
                    
                # Identify BTC column
                if 'btc' in col_lower:
                    btc_column = col
                    continue
                
                # Identify alt column (anything that's not BTC)
                # Either it has explicit alt names or it's just 'close'
                if any(alt in col_lower for alt in ['doge', 'alt', 'eth', 'coin']):
                    altcoin_column = col
                    break
            
            # If we didn't find an explicit altcoin column but found 'close', use that
            if altcoin_column is None and 'close' in backtest_data.columns:
                altcoin_column = 'close'
                
            # Use the identified altcoin column
            close_column = altcoin_column
            
            if close_column is None:
                print(f"Error: Could not find altcoin price column in data. Available columns: {backtest_data.columns.tolist()}")
                return self._get_default_metrics()
            
            
            # Get price series
            price = backtest_data[close_column]
            
            print(f"Using price column: {close_column} - {price}")
            
            # Create entry signal based on the pattern and lag
            if use_pattern in backtest_data.columns:
                # Create entry signal
                entries = backtest_data[use_pattern].shift(pattern_lag).astype(bool)
                
                # Filter by entry threshold if needed
                if entry_threshold > 0:
                    # Find returns column or calculate it
                    returns_column = None
                    for col in backtest_data.columns:
                        if 'return' in col.lower() and ('doge' in col.lower() or 'alt' in col.lower()):
                            returns_column = col
                            break
                    
                    if returns_column is None:
                        backtest_data['alt_returns'] = backtest_data[close_column].pct_change()
                        returns_column = 'alt_returns'
                    
                    # Apply threshold filter
                    entries = entries & (backtest_data[returns_column] > entry_threshold)
            else:
                print(f"Pattern {use_pattern} not found in data columns. Available columns: {backtest_data.columns.tolist()}")
                return self._get_default_metrics()
            
            # Get price series and convert to numpy for efficiency
            price_np = price.to_numpy()
            entries_np = entries.to_numpy()
            
            # Create custom exit signals incorporating stop-loss, take-profit and max holding time
            exits = np.zeros_like(entries_np)
            
            # Detect entry points
            entry_indices = np.where(entries_np)[0]
            
            # For each entry, calculate where the exit would happen
            for entry_idx in entry_indices:
                if entry_idx >= len(price_np) - 1:
                    continue  # Skip if entry is at the last bar
                
                # Initialize tracking for this trade
                exit_idx = None
                entry_price = price_np[entry_idx]
                
                # Check each subsequent bar for exit conditions
                for i in range(entry_idx + 1, min(len(price_np), entry_idx + max_holding_time + 1)):
                    current_price = price_np[i]
                    
                    # Calculate return
                    current_return = (current_price / entry_price) - 1
                    
                    # Check stop-loss
                    if current_return <= -stop_loss_pct:
                        exit_idx = i
                        break
                    
                    # Check take-profit
                    if current_return >= take_profit_pct:
                        exit_idx = i
                        break
                    
                    # Check max holding time
                    if i - entry_idx >= max_holding_time:
                        exit_idx = i
                        break
                
                # Set exit signal
                if exit_idx is not None:
                    exits[exit_idx] = True
            
            # Create a new portfolio with the custom exits
            pf = vbt.Portfolio.from_signals(
                close=price,
                entries=entries,
                exits=exits,
                init_cash=100000,  # Starting with $100k
                size_type='value',
                size=position_size_pct,  # Position size as percentage of portfolio
                fees=0.001,  # 0.1% trading fee
                freq='1m'  # Assuming 1-minute data
            )
            
            # Get performance metrics
            # Use a completely different, robust approach that works with all VectorBT versions
            
            # Store backtest results for visualization (safely)
            try:
                self.backtest_results = {
                    'trades': pf.trades.records_readable if hasattr(pf.trades, 'records_readable') else pf.trades.records,
                    'equity_curve': pf.value()
                }
            except Exception as e:
                print(f"Warning: Could not store backtest results: {e}")
                self.backtest_results = {
                    'trades': pd.DataFrame(),
                    'equity_curve': pd.Series([1.0])
                }
                
            # Get total trades safely
            try:
                total_trades = len(pf.trades)
            except:
                total_trades = 0
                
            # Calculate win rate safely
            try:
                win_rate = pf.trades.win_rate
            except:
                try:
                    # Alternative method
                    returns = pf.trades.returns
                    win_rate = (returns > 0).mean() if len(returns) > 0 else 0
                except:
                    win_rate = 0
            
            # Calculate profit factor safely
            try:
                # Different approaches for different VectorBT versions
                profit_factor = 0
                
                # Try using records directly (safest approach)
                records = pf.trades.records
                
                # Check if we have the necessary data in records
                if 'return' in records.columns or 'pnl' in records.columns:
                    use_col = 'pnl' if 'pnl' in records.columns else 'return'
                    
                    # Get winning and losing trade values
                    wins = records[records[use_col] > 0][use_col].sum()
                    losses = abs(records[records[use_col] <= 0][use_col].sum())
                    
                    # Calculate profit factor
                    profit_factor = wins / losses if losses > 0 else (1.0 if wins > 0 else 0.0)
                else:
                    # Default if we can't calculate
                    profit_factor = 1.0
            except Exception as e:
                print(f"Warning: Error calculating profit factor: {e}")
                profit_factor = 1.0
            
            # Get returns and calculate metrics safely
            try:
                # Total return
                total_return = float(pf.total_return())
                total_return_pct = total_return * 100
            except:
                total_return = 0
                total_return_pct = 0
                
            # Maximum drawdown
            try:
                max_dd = float(pf.max_drawdown())
                max_drawdown = max_dd * 100 if max_dd is not None else 0
            except:
                max_drawdown = 0
                
            # Sharpe ratio calculation
            try:
                returns = pf.returns()
                if len(returns) > 0 and returns.std() > 0:
                    # Annualized Sharpe Ratio (assuming 1-minute data)
                    sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252*1440))
                else:
                    sharpe_ratio = 0
            except:
                sharpe_ratio = 0
                
            # Average win/loss safely
            try:
                if hasattr(records, 'groupby'):
                    # Group by whether return/pnl is positive
                    use_col = 'pnl' if 'pnl' in records.columns else 'return'
                    grouped = records.groupby(records[use_col] > 0)
                    
                    # Get averages
                    avg_win = float(grouped.get_group(True)[use_col].mean()) if True in grouped.groups else 0
                    avg_loss = float(grouped.get_group(False)[use_col].mean()) if False in grouped.groups else 0
                else:
                    avg_win = 0
                    avg_loss = 0
            except:
                avg_win = 0
                avg_loss = 0
            
            # Return metrics in our standard format
            metrics = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
            return metrics
            
        except Exception as e:
            print(f"VectorBT backtest error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_default_metrics()
    
    def _get_default_metrics(self):
        """Return default metrics for failed backtests."""
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'total_return': 0,
            'total_return_pct': 0,
            'max_drawdown': 100,
            'sharpe_ratio': -10
        }
    
    def ml_optimization(self):
        """
        Use machine learning to optimize strategy parameters.
        
        Returns:
            dict: Best parameters and metrics
        """
        # Prepare for optimization
        param_combinations = []
        metrics_list = []
        
        # Focus on the strongest patterns for efficiency
        use_patterns = self.strongest_patterns
        print(f"Optimizing for the top patterns: {use_patterns}")
        
        # Define more focused parameter grid based on typical altcoin characteristics
        sl_tp_grid = []
        
        # Add typical SL/TP combinations that professionals use for altcoin trading
        # Generally using wider stops and larger targets for volatile assets
        for sl in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]:
            for tp in [2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0]:
                # Only include reasonable risk:reward ratios (1:1 or better)
                if tp >= sl:
                    sl_tp_grid.append((sl, tp))
        
        total_combinations = len(use_patterns) * len(sl_tp_grid) * len(self.param_space['pattern_lag'])
        print(f"Testing {total_combinations} parameter combinations...")
        
        # Run optimization grid search
        counter = 0
        best_sharpe = -float('inf')
        best_params = None
        best_metrics = None
        
        for pattern in use_patterns:
            for lag in self.param_space['pattern_lag']:
                for sl, tp in sl_tp_grid:
                    # Progress update
                    counter += 1
                    if counter % 50 == 0:
                        print(f"Progress: {counter}/{total_combinations} combinations tested")
                    
                    # Define parameter set
                    params = {
                        'entry_threshold': 0.0001,  # Small positive threshold
                        'stop_loss_pct': sl,
                        'take_profit_pct': tp,
                        'max_holding_time': 240,  # 4 hours default
                        'pattern_lag': lag,
                        'use_pattern': pattern,
                        'position_size_pct': 10  # Use 10% position size by default
                    }
                    
                    # Run backtest
                    metrics = self.backtest_strategy(params)
                    
                    # Store results
                    param_combinations.append(params)
                    metrics_list.append(metrics)
                    
                    # Update best if better
                    # We'll use a combined score of Sharpe ratio and total return, 
                    # with more weight on Sharpe ratio
                    combined_score = (metrics['sharpe_ratio'] * 0.7) + (metrics['total_return_pct'] / 100 * 0.3)
                    if combined_score > best_sharpe and metrics['total_trades'] >= 30:
                        best_sharpe = combined_score
                        best_params = params.copy()
                        best_metrics = metrics
                        print(f"New best found: Sharpe={metrics['sharpe_ratio']:.2f}, Return={metrics['total_return_pct']:.2f}%, Trades={metrics['total_trades']}")
        
        # Store results
        self.results = pd.DataFrame({
            **{k: [p[k] for p in param_combinations] for k in param_combinations[0].keys()},
            **{k: [m[k] for m in metrics_list] for k in metrics_list[0].keys()}
        })
        
        # Run final backtest with best parameters and save it
        if best_params:
            final_metrics = self.backtest_strategy(best_params)
            self.best_params = best_params
            self.performance = final_metrics
            
            print("\nBest Parameters Found:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
                
            print("\nPerformance Metrics:")
            for k, v in final_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print("No valid parameter combination found")
        
        return {
            'best_params': best_params,
            'best_metrics': best_metrics
        }
    
    def visualize_results(self):
        """Generate and save visualization of the backtest results."""
        if self.results is None or self.backtest_results is None:
            print("No backtest results to visualize")
            return
        
        # Create plots directory if needed
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # 1. Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.backtest_results['equity_curve'])
        plt.title('Strategy Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity (starting = 1.0)')
        plt.grid(True)
        plt.savefig('equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot stop-loss vs take-profit heatmap
        if len(self.results) > 0:
            # Get unique values
            sl_values = self.results['stop_loss_pct'].unique()
            tp_values = self.results['take_profit_pct'].unique()
            
            if len(sl_values) > 1 and len(tp_values) > 1:
                # Create pivot table for heatmap
                heatmap_data = self.results.pivot_table(
                    index='stop_loss_pct', 
                    columns='take_profit_pct', 
                    values='sharpe_ratio',
                    aggfunc='mean'
                )
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f')
                plt.title('Sharpe Ratio by Stop-Loss and Take-Profit (%)')
                plt.tight_layout()
                plt.savefig('sl_tp_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Plot trades by day of week
        if 'trades' in self.backtest_results and len(self.backtest_results['trades']) > 0:
            trades_df = self.backtest_results['trades']
            
            # Add day of week
            trades_df['day_of_week'] = trades_df['entry_time'].dt.day_name()
            
            # Group by day and calculate win rate
            day_stats = trades_df.groupby('day_of_week').agg(
                win_rate=('return', lambda x: (x > 0).mean()),
                count=('return', 'count')
            ).reset_index()
            
            # Order by standard days of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_stats['day_of_week'] = pd.Categorical(day_stats['day_of_week'], categories=day_order, ordered=True)
            day_stats = day_stats.sort_values('day_of_week')
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(day_stats['day_of_week'], day_stats['win_rate'], alpha=0.7)
            
            # Add count labels
            for bar, count in zip(bars, day_stats['count']):
                plt.text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.01, 
                    f'n={count}', 
                    ha='center'
                )
            
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
            plt.title('Win Rate by Day of Week')
            plt.ylabel('Win Rate')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('day_win_rate.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Plot parameter importance using a Random Forest model
        if len(self.results) >= 10:  # Need enough samples for ML
            try:
                # Prepare features and target
                features = [
                    'stop_loss_pct', 'take_profit_pct', 'pattern_lag', 
                    'max_holding_time', 'position_size_pct'
                ]
                
                # Create dummy variables for categorical features
                X = pd.get_dummies(self.results[features], drop_first=True)
                y = self.results['sharpe_ratio']
                
                # Train a Random Forest to evaluate feature importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Get feature importance
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Plot
                plt.figure(figsize=(12, 6))
                sns.barplot(x='importance', y='feature', data=importance.head(10))
                plt.title('Parameter Importance for Strategy Performance')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig('parameter_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error creating parameter importance plot: {e}")
    
    def get_optimal_strategy_summary(self):
        """Generate a text summary of the optimal strategy."""
        if self.best_params is None or self.performance is None:
            return "No optimization results available."
        
        # Get pattern stats for the best pattern
        pattern = self.best_params['use_pattern']
        lag = self.best_params['pattern_lag']
        
        best_pattern_info = ""
        if pattern in self.pattern_stats:
            stats = self.pattern_stats[pattern]
            if isinstance(stats, dict):
                best_pattern_info = f"""
Pattern "{pattern}" details:
- Number of instances: {stats.get('instances', 'N/A')}
- Correlation: {stats.get('correlation', 'N/A'):.4f}
- Average return: {stats.get('avg_return', 'N/A'):.6f}
- Win rate: {stats.get('win_rate', 'N/A'):.2%}
"""
        
        # Format strategy summary
        summary = f"""
========== OPTIMAL STRATEGY SUMMARY ==========

STRATEGY PARAMETERS:
- Pattern to monitor: {pattern}
- Signal lag (minutes): {lag}
- Entry threshold: {self.best_params['entry_threshold']:.6f}
- Stop Loss: {self.best_params['stop_loss_pct']:.2f}%
- Take Profit: {self.best_params['take_profit_pct']:.2f}%
- Max holding time: {self.best_params['max_holding_time']} minutes
- Position size: {self.best_params['position_size_pct']}% of capital

{best_pattern_info}

PERFORMANCE METRICS:
- Total trades: {self.performance['total_trades']}
- Win rate: {self.performance['win_rate']:.2%}
- Profit factor: {self.performance['profit_factor']:.2f}
- Average win: {self.performance['avg_win']:.4f}
- Average loss: {self.performance['avg_loss']:.4f}
- Sharpe ratio: {self.performance['sharpe_ratio']:.2f}
- Total return: {self.performance['total_return_pct']:.2f}%
- Maximum drawdown: {self.performance['max_drawdown']:.2f}%

STRATEGY IMPLEMENTATION:
1. Monitor Bitcoin for the "{pattern}" pattern
2. When pattern is detected, wait {lag} minutes
3. Enter DOGE long position with {self.best_params['position_size_pct']}% of capital
4. Set stop-loss at {self.best_params['stop_loss_pct']:.2f}% below entry price
5. Set take-profit at {self.best_params['take_profit_pct']:.2f}% above entry price
6. Exit the position if max holding time of {self.best_params['max_holding_time']} minutes is reached
"""
        
        return summary

    def generate_html_report(self, output_file):
        """Generate a detailed HTML report of strategy optimization results."""
        try:
            if not hasattr(self, 'best_params') or not hasattr(self, 'performance') or self.best_params is None:
                with open(output_file, 'w') as f:
                    f.write("<html><body><h1>No optimization results available.</h1></body></html>")
                print(f"Created empty report (no results) at {output_file}")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Generate HTML content
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Strategy Optimization Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-box {{ border: 1px solid #ddd; border-radius: 5px; padding: 10px; text-align: center; }}
        .metric-label {{ font-size: 0.8em; color: #666; }}
        .metric-value {{ font-weight: bold; font-size: 1.2em; margin-top: 5px; }}
        .chart-container {{ margin: 25px 0; }}
        .chart-container img {{ max-width: 100%; border: 1px solid #eee; border-radius: 5px; }}
        .parameters-section {{ margin: 25px 0; }}
        .strategy-section {{ margin: 25px 0; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }}
        .code-block {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; max-height: 400px; overflow-y: auto; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .back-link {{ margin-bottom: 20px; }}
        .download-btn {{ background-color: #4CAF50; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; display: inline-block; }}
        .download-btn:hover {{ background-color: #45a049; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>StrategyOptimizer Results</h1>
        <p>Machine learning optimized trading parameters based on BTC-Altcoin pattern analysis.</p>
        
        <h2>Strategy Performance</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{self.performance.get('win_rate', 0)*100:.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{self.performance.get('profit_factor', 0):.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{self.performance.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{self.performance.get('total_return_pct', 0):.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Drawdown</div>
                <div class="metric-value">{self.performance.get('max_drawdown', 0):.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{self.performance.get('total_trades', 0)}</div>
            </div>
        </div>
        
        <h2>Optimal Strategy Parameters</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Pattern</div>
                <div class="metric-value">{self.best_params.get('use_pattern', 'N/A')}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Lag (min)</div>
                <div class="metric-value">{self.best_params.get('pattern_lag', 0)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Stop Loss</div>
                <div class="metric-value">{self.best_params.get('stop_loss_pct', 0):.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Take Profit</div>
                <div class="metric-value">{self.best_params.get('take_profit_pct', 0):.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Position Size</div>
                <div class="metric-value">{self.best_params.get('position_size_pct', 0):.1f}%</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Strategy Performance Charts</h2>
            <h3>Equity Curve</h3>
            <img src="../charts/equity_curve.png" alt="Equity Curve" />
            
            <h3>Stop-Loss/Take-Profit Optimization</h3>
            <img src="../charts/sl_tp_heatmap.png" alt="SL/TP Optimization" />
            
            <h3>Performance by Day</h3>
            <img src="../charts/day_win_rate.png" alt="Performance by Day" />
            
            <h3>Parameter Importance</h3>
            <img src="../charts/parameter_importance.png" alt="Parameter Importance" />
        </div>
        
        <div class="strategy-section">
            <h2>Strategy Implementation Guide</h2>
            <p>This strategy monitors Bitcoin for the <strong>{self.best_params.get('use_pattern', 'N/A')}</strong> pattern and trades the altcoin accordingly with the following approach:</p>
            
            <ol>
                <li>Monitor Bitcoin for the <strong>{self.best_params.get('use_pattern', 'N/A')}</strong> pattern</li>
                <li>When pattern is detected, wait <strong>{self.best_params.get('pattern_lag', 0)} minutes</strong></li>
                <li>Enter a long position with <strong>{self.best_params.get('position_size_pct', 0):.1f}%</strong> of capital</li>
                <li>Set stop loss at <strong>{self.best_params.get('stop_loss_pct', 0):.1f}%</strong> below entry</li>
                <li>Set take profit at <strong>{self.best_params.get('take_profit_pct', 0):.1f}%</strong> above entry</li>
                <li>Exit after <strong>{self.best_params.get('max_holding_time', 0)}</strong> minutes if neither target is reached</li>
            </ol>
            
            <h3>Complete Strategy Summary</h3>
            <div class="code-block">{self.get_optimal_strategy_summary()}</div>
        </div>
    </div>
</body>
</html>"""
            
            # Write HTML content to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
                
            print(f"Successfully generated strategy HTML report: {output_file}")
            
        except Exception as e:
            print(f"Error generating strategy HTML report: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create a minimal error report
            try:
                with open(output_file, 'w') as f:
                    f.write(f"""<html>
<body>
<h1>Error Generating Strategy Report</h1>
<p>An error occurred while generating the strategy optimization report:</p>
<pre>{str(e)}</pre>
</body>
</html>""")
            except:
                pass
