"""Simplified trading strategy optimization."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vectorbt as vbt
import seaborn as sns
from datetime import datetime
import random

class StrategyOptimizer:
    """Class to optimize trading strategy parameters."""
    
    def __init__(self, data, pattern_stats):
        """Initialize with market data and pattern statistics."""
        self.data = data
        self.pattern_stats = pattern_stats
        self.results = []
        self.best_params = None
        self.performance_metrics = None
        self.trade_list = []
        self.equity_curve = None
        self.daily_returns = []
        self.daily_dates = []
        
        # Find altcoin name from data columns
        self.altcoin_name = self._detect_altcoin_name()
        
        # More robust pattern detection
        patterns_list = []
        
        # Try to get patterns from pattern_stats
        if isinstance(pattern_stats, dict) and pattern_stats:
            patterns_list = list(pattern_stats.keys())
            print(f"Found {len(patterns_list)} patterns in pattern_stats")
        
        # If no patterns found, scan data columns
        if not patterns_list:
            patterns_list = [col for col in data.columns if 'pattern' in col.lower()]
            if patterns_list:
                print(f"Found {len(patterns_list)} pattern columns in data")
        
        # If still no patterns, add default indicator columns
        if not patterns_list:
            print("No patterns found. Using basic indicators if available.")
            for col in data.columns:
                if any(indicator in col.lower() for indicator in ['rsi', 'macd', 'bb_', 'stoch']):
                    patterns_list.append(col)
        
        # Validate patterns exist in data
        valid_patterns = [p for p in patterns_list if p in data.columns]
        print(f"Using {len(valid_patterns)} valid patterns for optimization")
        
        # Simplified parameter space
        self.param_space = {
            'use_pattern': valid_patterns if valid_patterns else ['btc_returns'],  # Fallback
            'pattern_lag': [1, 3, 5],
            'stop_loss_pct': [2.0, 3.0, 5.0],
            'take_profit_pct': [4.0, 6.0, 10.0],
            'max_holding_time': [5, 15, 30, 60],
            'position_size_pct': [5, 10]
        }
        
    def _detect_altcoin_name(self):
        """Detect which altcoin is being analyzed from data columns."""
        for col in self.data.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                return col.split('_')[0].upper()
        return "Altcoin"
        
    def backtest_strategy(self, params):
        """Simple backtest implementation."""
        try:
            # Get price data
            alt_prefix = self.altcoin_name.lower()
            print(f"Looking for price column with prefix: {alt_prefix}")
            
            # Find the price column (more robust method)
            close_column = None
            for col in self.data.columns:
                if 'close' in col.lower() and alt_prefix in col.lower():
                    close_column = col
                    break
            
            # Fallback to any close column
            if not close_column:
                for col in self.data.columns:
                    if 'close' in col.lower():
                        close_column = col
                        break
            
            if not close_column:
                print("ERROR: Could not find price column for backtesting")
                return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0}
            
            print(f"Using price column: {close_column}")
            
            # Get price and create signal
            price = self.data[close_column]
            
            # Handle case if use_pattern column doesn't exist
            if params['use_pattern'] not in self.data.columns:
                print(f"ERROR: Pattern {params['use_pattern']} not found in data")
                return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0}
            
            # Create entries and exits
            entries = self.data[params['use_pattern']].shift(params['pattern_lag']).fillna(False)
            exits = pd.Series(False, index=self.data.index)
            
            # Check if we have any entry signals
            if not entries.any():
                print(f"WARNING: No entry signals found for pattern {params['use_pattern']}")
                return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0}
                
            # Create portfolio
            pf = vbt.Portfolio.from_signals(
                close=price,
                entries=entries,
                exits=exits,
                init_cash=10000,
                fees=0.001,
                size=params['position_size_pct']/100,
                size_type='percent',
                sl_stop=params['stop_loss_pct']/100,
                tp_stop=params['take_profit_pct']/100
            )
            
            # Store equity curve
            self.equity_curve = pf.value()
            
            # Extract basic metrics directly (avoid complex trade extraction)
            metrics = {
                'total_trades': pf.stats()['total_trades'] if 'total_trades' in pf.stats() else 0,
                'win_rate': float(pf.stats()['win_rate']) if 'win_rate' in pf.stats() else 0,
                'total_return_pct': float(pf.stats()['total_return']) * 100 if 'total_return' in pf.stats() else 0,
                'max_drawdown': float(pf.stats()['max_drawdown']) * 100 if 'max_drawdown' in pf.stats() else 0,
                'sharpe_ratio': float(pf.stats()['sharpe_ratio']) if 'sharpe_ratio' in pf.stats() else 0
            }
            
            # Convert NumPy types to native Python types
            return self._convert_numpy_types(metrics)
            
        except Exception as e:
            print(f"Backtest error: {e}")
            return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0}
            
    def ml_optimization(self):
        """Simplified optimization process."""
        print(f"Optimizing strategy parameters for {self.altcoin_name}...")
        
        # Verify we have patterns to test
        if not self.param_space['use_pattern']:
            print("ERROR: No valid patterns for optimization!")
            return []
        
        # Print number of patterns being used
        print(f"Testing {len(self.param_space['use_pattern'])} patterns")
        
        # Create parameter combinations
        param_grid = []
        for pattern in self.param_space['use_pattern']:
            for lag in self.param_space['pattern_lag']:
                for sl in self.param_space['stop_loss_pct']:
                    for tp in self.param_space['take_profit_pct']:
                        if tp > sl:  # Filter for logical combinations
                            for position_size in self.param_space['position_size_pct']:
                                param_grid.append({
                                    'use_pattern': pattern,
                                    'pattern_lag': lag,
                                    'stop_loss_pct': sl,
                                    'take_profit_pct': tp,
                                    'max_holding_time': 60,  # Simplified
                                    'position_size_pct': position_size
                                })
        
        # Check if we have any combinations
        if not param_grid:
            print("ERROR: No parameter combinations generated!")
            return []
            
        print(f"Generated {len(param_grid)} parameter combinations")
        
        # Limit combinations to 100
        if len(param_grid) > 100:
            print(f"Limiting parameter combinations from {len(param_grid)} to 100")
            random.shuffle(param_grid)
            param_grid = param_grid[:100]
        
        # Test each combination
        best_return = -float('inf')
        for i, params in enumerate(param_grid):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(param_grid)} combinations evaluated")
                
            metrics = self.backtest_strategy(params)
            result = {'params': params, 'metrics': metrics}
            self.results.append(result)
            
            # Track best parameters by return
            if metrics['total_return_pct'] > best_return and metrics['total_trades'] > 5:
                best_return = metrics['total_return_pct']
                self.best_params = params
                self.performance_metrics = metrics
        
        # Check if we found any good results
        if not self.best_params:
            print("No profitable strategies found!")
            # Use the least worst strategy if available
            if self.results:
                best_result = max(self.results, key=lambda x: x['metrics']['total_return_pct'])
                self.best_params = best_result['params']
                self.performance_metrics = best_result['metrics']
        
        print(f"Optimization completed. Evaluated {len(self.results)} combinations.")
        return self.results
        
    def generate_strategy_charts(self, charts_dir):
        """Generate basic strategy charts."""
        os.makedirs(charts_dir, exist_ok=True)
        
        # Equity curve
        plt.figure(figsize=(10, 6))
        if self.equity_curve is not None:
            plt.plot(self.equity_curve)
            plt.title('Strategy Equity Curve')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No equity curve data available', 
                    horizontalalignment='center', verticalalignment='center')
        plt.savefig(os.path.join(charts_dir, 'equity_curve.png'))
        plt.close()
        
        # Create other placeholder charts
        for chart_name in ['sl_tp_optimization', 'daily_performance', 'parameter_importance']:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'No {chart_name} data available', 
                   horizontalalignment='center', verticalalignment='center')
            plt.savefig(os.path.join(charts_dir, f'{chart_name}.png'))
            plt.close()
        
        return True
        
    def generate_html_report(self, output_file, charts_dir):
        """Generate a basic HTML report."""
        try:
            # Generate charts first
            self.generate_strategy_charts(charts_dir)
            
            # Create directory for output file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Generate simple HTML report
            if self.best_params and self.performance_metrics:
                html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Strategy Results for {self.altcoin_name}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Strategy Results for {self.altcoin_name}</h1>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <h3>Best Parameters</h3>
                <ul class="list-group">
                    <li class="list-group-item">Pattern: {self.best_params.get('use_pattern', 'N/A')}</li>
                    <li class="list-group-item">Lag: {self.best_params.get('pattern_lag', 0)} minutes</li>
                    <li class="list-group-item">Stop Loss: {self.best_params.get('stop_loss_pct', 0)}%</li>
                    <li class="list-group-item">Take Profit: {self.best_params.get('take_profit_pct', 0)}%</li>
                </ul>
            </div>
            
            <div class="col-md-6">
                <h3>Performance</h3>
                <ul class="list-group">
                    <li class="list-group-item">Total Return: {self.performance_metrics.get('total_return_pct', 0):.2f}%</li>
                    <li class="list-group-item">Win Rate: {self.performance_metrics.get('win_rate', 0)*100:.1f}%</li>
                    <li class="list-group-item">Total Trades: {self.performance_metrics.get('total_trades', 0)}</li>
                </ul>
            </div>
        </div>
        
        <div class="mt-4">
            <h3>Equity Curve</h3>
            <img src="../charts/equity_curve.png" class="img-fluid" alt="Equity Curve">
        </div>
        
        <div class="mt-4">
            <h4>Holding Time Analysis</h4>"""

                # Check if we have holding time data
                if 'all_holding_time_results' in self.results:
                    # Create a table of holding time results
                    html += """
        <div class="table-responsive">
            <table class="table table-sm table-hover">
                <thead>
                    <tr>
                        <th>Holding Period</th>
                        <th>Return %</th>
                        <th>Win Rate</th>
                        <th>Trades</th>
                        <th>Sharpe</th>
                    </tr>
                </thead>
                <tbody>"""
    
                    # Sort by return
                    sorted_times = sorted(
                        self.results['all_holding_time_results'].items(),
                        key=lambda x: x[1]['total_return_pct'],
                        reverse=True
                    )
    
                    for hold_time, ht_result in sorted_times:
                        is_optimal = hold_time == self.results.get('holding_time')
                        html += f"""
                <tr class="{'table-success' if is_optimal else ''}">
                    <td>{"★ " if is_optimal else ""}{hold_time} min</td>
                    <td>{ht_result['total_return_pct']:.2f}%</td>
                    <td>{ht_result['win_rate']*100:.1f}%</td>
                    <td>{ht_result['total_trades']}</td>
                    <td>{ht_result.get('sharpe_ratio', 0):.2f}</td>
                </tr>"""
    
                    html += """
                </tbody>
            </table>
        </div>"""
        
                    # Also add a chart visualizing holding time performance
                    # Create it on-the-fly
                    try:
                        plt.figure(figsize=(10, 6))
        
                        # Get data for the chart
                        times = []
                        returns = []
                        win_rates = []
        
                        for time, result in sorted(self.results['all_holding_time_results'].items()):
                            times.append(time)
                            returns.append(result['total_return_pct'])
                            win_rates.append(result['win_rate'] * 100)
        
                        # Plot the returns
                        plt.plot(times, returns, 'b-o', label='Return %')
                        plt.grid(True, alpha=0.3)
        
                        # Mark optimal holding time
                        optimal = self.results.get('holding_time')
                        if optimal in times:
                            idx = times.index(optimal)
                            plt.plot([optimal], [returns[idx]], 'ro', markersize=10, label='Optimal')
        
                        plt.xlabel('Holding Time (minutes)')
                        plt.ylabel('Return %')
                        plt.title(f'Performance by Holding Time: {self.altcoin_name}')
                        plt.legend()
        
                        # Save the chart
                        chart_path = os.path.join(charts_dir, f"holding_time.png")
                        plt.savefig(chart_path)
                        plt.close()
        
                        # Add it to the HTML report
                        html += f"""
        <div class="mt-3">
            <img src="../charts/holding_time.png" class="img-fluid" alt="Holding Time Analysis">
        </div>"""
                    except Exception as e:
                        print(f"Error generating holding time chart: {e}")
                        html += f"""
        <div class="alert alert-warning mt-2">
            Error generating holding time chart: {e}
        </div>"""
                else:
                    html += """
        <div class="alert alert-warning">
            No holding time analysis data available
        </div>"""
                
                html += """
    </div>
</body>
</html>"""
            else:
                html = """<!DOCTYPE html>
<html>
<head>
    <title>Strategy Optimization Failed</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Strategy Optimization Failed</h1>
        <div class="alert alert-danger">
            No optimization results available. Check logs for errors.
        </div>
    </div>
</body>
</html>"""
                
            with open(output_file, 'w') as f:
                f.write(html)
                
        except Exception as e:
            print(f"Error generating HTML report: {e}")

    def _convert_numpy_types(self, obj):
        """Convert NumPy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
