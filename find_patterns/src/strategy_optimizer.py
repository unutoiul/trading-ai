"""Price action based trading strategy optimization."""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random

class StrategyOptimizer:
    """Class to optimize trading strategy parameters based on price action signals."""
    
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
        
        # More robust pattern detection for price action
        patterns_list = []
        
        # Try to get patterns from pattern_stats
        if isinstance(pattern_stats, dict) and pattern_stats:
            patterns_list = list(pattern_stats.keys())
            print(f"Found {len(patterns_list)} patterns in pattern_stats")
        
        # If no patterns found, scan data columns
        if not patterns_list:
            # Look for price action columns
            price_action_patterns = [
                col for col in data.columns if any(term in col.lower() for term in 
                ['strong_up', 'medium_up', 'small_up', 'strong_down', 'medium_down', 'small_down'])
            ]
            if price_action_patterns:
                patterns_list = price_action_patterns
                print(f"Found {len(patterns_list)} price action patterns in data columns")
        
        # Validate patterns exist in data
        valid_patterns = [p for p in patterns_list if p in data.columns]
        print(f"Using {len(valid_patterns)} valid patterns for optimization")
        
        # Enhanced parameter space with trailing stop and hold time variations
        self.param_space = {
            'use_pattern': valid_patterns if valid_patterns else ['btc_returns'],  # Fallback
            'pattern_lag': [1, 2, 3, 5, 10, 15],  # More granular lag options
            'stop_loss_pct': [1.5, 2.0, 3.0, 5.0],  # More options including tighter stops
            'take_profit_pct': [3.0, 4.0, 6.0, 10.0],
            'trailing_stop_pct': [1.0, 1.5, 2.0, 3.0, 5.0],  # More granular trailing stops
            'max_holding_time': [5, 15, 30, 60, 120, 240],  # More holding time options
            'position_size_pct': [5, 10, 20]  # Added larger position size
        }
        
        # Track various performance metrics
        self.metrics_by_pattern = {}
        self.metrics_by_parameter = {}
        
    def _detect_altcoin_name(self):
        """Detect which altcoin is being analyzed from data columns."""
        for col in self.data.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                return col.split('_')[0].upper()
        
        # Try alternative detection methods
        if 'altcoin_name' in self.data.columns and not self.data['altcoin_name'].isna().all():
            altcoin = self.data['altcoin_name'].iloc[0]
            if isinstance(altcoin, str):
                return altcoin.upper()
        
        return "ALTCOIN"
        
    def backtest_strategy(self, params):
        """Enhanced backtest implementation with trailing stop and realistic trading simulation."""
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
                    if 'close' in col.lower() and not 'btc' in col.lower():
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
            
            # Create entries and exits based on pattern signals
            entries = self.data[params['use_pattern']].shift(params['pattern_lag']).fillna(False)
            exits = pd.Series(False, index=self.data.index)
            
            # Check if we have any entry signals
            if not entries.any():
                print(f"WARNING: No entry signals found for pattern {params['use_pattern']}")
                return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0}
            
            # Run backtest using realistic trading simulation
            initial_capital = 10000
            capital = initial_capital
            position = 0
            entry_price = 0
            trades = []
            equity_curve = []
            
            # For tracking trailing stop
            highest_price = 0
            trailing_stop_price = 0
            
            # Track max holding time
            entry_time = None
            max_holding_time = params['max_holding_time']
            
            for i in range(len(self.data)):
                current_price = price.iloc[i]
                current_time = price.index[i]
                equity_curve.append(capital + position * current_price)
                
                # Check trailing stop if in position
                if position > 0:
                    # Update highest price seen since entry
                    highest_price = max(highest_price, current_price)
                    # Recalculate trailing stop price
                    trailing_stop_level = params['trailing_stop_pct'] / 100
                    trailing_stop_price = highest_price * (1 - trailing_stop_level)
                    
                    # Check holding time
                    time_held = 0
                    if entry_time is not None and hasattr(current_time, 'timestamp') and hasattr(entry_time, 'timestamp'):
                        time_held = (current_time - entry_time).total_seconds() / 60  # in minutes
                    
                    # Determine if we should exit based on trailing stop, stop-loss, take-profit, or max holding time
                    stop_loss_price = entry_price * (1 - params['stop_loss_pct']/100)
                    take_profit_price = entry_price * (1 + params['take_profit_pct']/100)
                    
                    exit_reason = None
                    
                    if current_price <= trailing_stop_price:
                        exit_reason = "trailing_stop"
                    elif current_price <= stop_loss_price:
                        exit_reason = "stop_loss"
                    elif current_price >= take_profit_price:
                        exit_reason = "take_profit"
                    elif max_holding_time > 0 and time_held >= max_holding_time:
                        exit_reason = "max_holding_time"
                    
                    if exit_reason:
                        profit_pct = (current_price / entry_price - 1) * 100
                        trade_info = {
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'profit_pct': profit_pct,
                            'exit_reason': exit_reason,
                            'pattern': params['use_pattern']
                        }
                        trades.append(trade_info)
                        
                        # Update capital
                        capital = capital + position * current_price
                        position = 0
                        entry_price = 0
                        entry_time = None
                
                # Check for new entry signals if not in position
                if position == 0 and entries.iloc[i]:
                    # Calculate position size based on parameter
                    position_value = capital * (params['position_size_pct'] / 100)
                    position = position_value / current_price
                    capital -= position_value
                    entry_price = current_price
                    entry_time = current_time
                    highest_price = current_price
                    trailing_stop_price = current_price * (1 - params['trailing_stop_pct']/100)
            
            # Close any open position at the end
            if position > 0:
                current_price = price.iloc[-1]
                profit_pct = (current_price / entry_price - 1) * 100
                trade_info = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': price.index[-1],
                    'exit_price': current_price,
                    'profit_pct': profit_pct,
                    'exit_reason': 'end_of_data',
                    'pattern': params['use_pattern']
                }
                trades.append(trade_info)
                
                # Update final capital
                capital = capital + position * current_price
                position = 0
            
            # Calculate performance metrics
            final_equity = capital
            total_return_pct = (final_equity / initial_capital - 1) * 100
            win_trades = [t for t in trades if t['profit_pct'] > 0]
            win_rate = len(win_trades) / len(trades) if trades else 0
            
            # Calculate average profit per trade and max drawdown
            profits = [t['profit_pct'] for t in trades]
            avg_profit = sum(profits) / len(profits) if profits else 0
            
            # Calculate drawdown
            peak = initial_capital
            drawdowns = []
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown_pct = (peak - equity) / peak * 100
                drawdowns.append(drawdown_pct)
            
            max_drawdown = max(drawdowns) if drawdowns else 0
            
            # Store equity curve
            self.equity_curve = pd.Series(equity_curve, index=price.index)
            self.trade_list = trades
            
            # Calculate additional metrics (Sharpe Ratio approximation)
            if len(equity_curve) > 1:
                daily_returns = [(equity_curve[i] / equity_curve[i-1] - 1) for i in range(1, len(equity_curve))]
                avg_daily_return = sum(daily_returns) / len(daily_returns)
                std_daily_return = np.std(daily_returns) if len(daily_returns) > 1 else 0.01
                sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Compile metrics
            metrics = {
                'total_trades': len(trades),
                'win_rate': win_rate,
                'total_return_pct': total_return_pct,
                'avg_profit_per_trade': avg_profit,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'final_equity': final_equity,
                'profit_factor': sum([t['profit_pct'] for t in win_trades]) / abs(sum([t['profit_pct'] for t in trades if t['profit_pct'] < 0])) if sum([t['profit_pct'] for t in trades if t['profit_pct'] < 0]) != 0 else float('inf')
            }
            
            return metrics
            
        except Exception as e:
            print(f"Backtest error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0}
            
    def ml_optimization(self, price_action_focus=True, optimize_count=100):
        """Enhanced optimization process with price action focus."""
        print(f"Optimizing strategy parameters for {self.altcoin_name}...")
        
        # Verify we have patterns to test
        if not self.param_space['use_pattern']:
            print("ERROR: No valid patterns for optimization!")
            return []
        
        # If price action focus is enabled, prioritize price action patterns
        if price_action_focus:
            price_action_patterns = [
                p for p in self.param_space['use_pattern'] 
                if any(term in p for term in ['strong', 'medium', 'small', 'up', 'down'])
            ]
            if price_action_patterns:
                print(f"Focusing on {len(price_action_patterns)} price action patterns")
                self.param_space['use_pattern'] = price_action_patterns
        
        # Print number of patterns being used
        print(f"Testing {len(self.param_space['use_pattern'])} patterns")
        
        # Create parameter combinations
        param_grid = []
        for pattern in self.param_space['use_pattern']:
            for lag in self.param_space['pattern_lag']:
                for sl in self.param_space['stop_loss_pct']:
                    for tp in self.param_space['take_profit_pct']:
                        for ts in self.param_space['trailing_stop_pct']:
                            for hold_time in self.param_space['max_holding_time']:
                                for pos_size in self.param_space['position_size_pct']:
                                    param_grid.append({
                                        'use_pattern': pattern,
                                        'pattern_lag': lag,
                                        'stop_loss_pct': sl,
                                        'take_profit_pct': tp,
                                        'trailing_stop_pct': ts,
                                        'max_holding_time': hold_time,
                                        'position_size_pct': pos_size
                                    })
        
        # Check if we have any combinations
        if not param_grid:
            print("ERROR: No parameter combinations generated!")
            return []
            
        print(f"Generated {len(param_grid)} parameter combinations")
        
        # Limit combinations to optimize_count for efficiency
        if len(param_grid) > optimize_count:
            print(f"Limiting parameter combinations from {len(param_grid)} to {optimize_count}")
            random.shuffle(param_grid)
            param_grid = param_grid[:optimize_count]
        
        # Track metrics by parameter value
        parameter_metrics = {
            'pattern_lag': {},
            'stop_loss_pct': {},
            'take_profit_pct': {},
            'trailing_stop_pct': {},
            'max_holding_time': {}
        }
        
        # Test each combination
        best_return = -float('inf')
        for i, params in enumerate(param_grid):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(param_grid)} combinations evaluated")
                
            metrics = self.backtest_strategy(params)
            result = {'params': params, 'metrics': metrics}
            self.results.append(result)
            
            # Track best parameters by return
            if metrics['total_return_pct'] > best_return and metrics['total_trades'] >= 5:
                best_return = metrics['total_return_pct']
                self.best_params = params
                self.performance_metrics = metrics
            
            # Track metrics by pattern
            pattern = params['use_pattern']
            if pattern not in self.metrics_by_pattern:
                self.metrics_by_pattern[pattern] = []
            self.metrics_by_pattern[pattern].append(metrics)
            
            # Track metrics by parameter value
            for param_name in parameter_metrics:
                param_value = params[param_name]
                if param_value not in parameter_metrics[param_name]:
                    parameter_metrics[param_name][param_value] = []
                parameter_metrics[param_name][param_value].append(metrics)
        
        # Calculate average metrics by parameter value
        self.metrics_by_parameter = {}
        for param_name, param_values in parameter_metrics.items():
            self.metrics_by_parameter[param_name] = {}
            for value, metrics_list in param_values.items():
                if metrics_list:
                    avg_metrics = {
                        'total_return_pct': sum(m['total_return_pct'] for m in metrics_list) / len(metrics_list),
                        'win_rate': sum(m['win_rate'] for m in metrics_list) / len(metrics_list),
                        'trade_count': sum(m['total_trades'] for m in metrics_list) / len(metrics_list),
                        'count': len(metrics_list)
                    }
                    self.metrics_by_parameter[param_name][value] = avg_metrics
        
        # Generate additional trailing stop analysis
        self._analyze_trailing_stop_impact()
        
        # Check if we found any good results
        if not self.best_params:
            print("No profitable strategies found!")
            # Use the least worst strategy if available
            if self.results:
                best_result = max(self.results, key=lambda x: x['metrics']['total_return_pct'])
                self.best_params = best_result['params']
                self.performance_metrics = best_result['metrics']
        
        print(f"Optimization completed. Evaluated {len(self.results)} combinations.")
        
        # Print best parameters
        if self.best_params:
            print("\nBest parameters found:")
            print(f"Pattern: {self.best_params['use_pattern']}")
            print(f"Lag: {self.best_params['pattern_lag']} periods")
            print(f"Stop Loss: {self.best_params['stop_loss_pct']}%")
            print(f"Take Profit: {self.best_params['take_profit_pct']}%")
            print(f"Trailing Stop: {self.best_params['trailing_stop_pct']}%")
            print(f"Max Hold Time: {self.best_params['max_holding_time']} periods")
            print(f"Position Size: {self.best_params['position_size_pct']}%")
            print("\nPerformance metrics:")
            print(f"Total Return: {self.performance_metrics['total_return_pct']:.2f}%")
            print(f"Win Rate: {self.performance_metrics['win_rate']*100:.1f}%")
            print(f"Total Trades: {self.performance_metrics['total_trades']}")
            print(f"Max Drawdown: {self.performance_metrics['max_drawdown']:.2f}%")
            
        return self.results
        
    def generate_strategy_charts(self, charts_dir):
        """Generate strategy performance charts."""
        os.makedirs(charts_dir, exist_ok=True)
        
        # Equity curve
        plt.figure(figsize=(10, 6))
        if self.equity_curve is not None and len(self.equity_curve) > 0:
            plt.plot(self.equity_curve, linewidth=2)
            plt.title('Strategy Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'No equity curve data available', 
                    horizontalalignment='center', verticalalignment='center')
        plt.savefig(os.path.join(charts_dir, 'equity_curve.png'))
        plt.close()
        
        # Parameter impact charts
        self._generate_parameter_impact_charts(charts_dir)
        
        # Trade distribution chart
        if self.trade_list:
            self._generate_trade_distribution_chart(charts_dir)
        
        # Trailing stop impact chart
        if hasattr(self, 'trailing_stop_analysis'):
            self._generate_trailing_stop_chart(charts_dir)
        
        return True
    
    def _generate_parameter_impact_charts(self, charts_dir):
        """Generate charts showing impact of different parameters."""
        # Only create if we have parameter metrics
        if not self.metrics_by_parameter:
            return
        
        # Create chart for each parameter
        for param_name, param_data in self.metrics_by_parameter.items():
            if not param_data:
                continue
                
            # Sort by parameter value
            param_values = sorted(param_data.keys())
            returns = [param_data[v]['total_return_pct'] for v in param_values]
            win_rates = [param_data[v]['win_rate'] * 100 for v in param_values]
            
            # Create chart
            plt.figure(figsize=(10, 6))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Plot returns
            ax1.bar(range(len(param_values)), returns, color='blue', alpha=0.7)
            ax1.set_ylabel('Average Return (%)')
            ax1.set_title(f'Impact of {param_name} on Performance')
            ax1.grid(True, alpha=0.3)
            
            # Plot win rates
            ax2.bar(range(len(param_values)), win_rates, color='green', alpha=0.7)
            ax2.set_xlabel(param_name)
            ax2.set_ylabel('Win Rate (%)')
            ax2.set_xticks(range(len(param_values)))
            ax2.set_xticklabels([str(v) for v in param_values])
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, f'{param_name}_impact.png'))
            plt.close()
    
    def _generate_trade_distribution_chart(self, charts_dir):
        """Generate chart showing distribution of trade profits."""
        if not self.trade_list:
            return
            
        # Extract profit percentages
        profits = [trade['profit_pct'] for trade in self.trade_list]
        
        plt.figure(figsize=(10, 6))
        plt.hist(profits, bins=20, alpha=0.7, color='blue')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Distribution of Trade Profits')
        plt.xlabel('Profit (%)')
        plt.ylabel('Number of Trades')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'trade_distribution.png'))
        plt.close()
    
    def _generate_trailing_stop_chart(self, charts_dir):
        """Generate chart showing impact of trailing stop percentages."""
        if not hasattr(self, 'trailing_stop_analysis'):
            return
            
        # Create visualization
        ts_values = sorted(self.trailing_stop_analysis.keys())
        returns = [self.trailing_stop_analysis[ts]['avg_return'] for ts in ts_values]
        win_rates = [self.trailing_stop_analysis[ts]['avg_win_rate'] * 100 for ts in ts_values]
        
        plt.figure(figsize=(10, 8))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot returns
        ax1.bar(range(len(ts_values)), returns, color='blue', alpha=0.7)
        ax1.set_ylabel('Average Return (%)')
        ax1.set_title('Trailing Stop Impact on Performance')
        ax1.grid(True, alpha=0.3)
        
        # Plot win rates
        ax2.bar(range(len(ts_values)), win_rates, color='green', alpha=0.7)
        ax2.set_xlabel('Trailing Stop (%)')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_xticks(range(len(ts_values)))
        ax2.set_xticklabels([str(ts) for ts in ts_values])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'trailing_stop_impact.png'))
        plt.close()
        
    def generate_html_report(self, output_file, charts_dir):
        """Generate HTML report with strategy performance details."""
        try:
            # Generate charts first
            self.generate_strategy_charts(charts_dir)
            
            # Create directory for output file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Start building HTML content
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Strategy Optimization Results - {self.altcoin_name}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-box {{ padding: 15px; margin: 5px; background-color: #f8f9fa; border-radius: 4px; text-align: center; }}
        .metric-title {{ font-size: 0.9rem; color: #6c757d; }}
        .metric-value {{ font-size: 1.2rem; font-weight: bold; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .chart-container {{ margin-bottom: 30px; }}
        .table-sm {{ font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Price Action Strategy Optimization for {self.altcoin_name}</h1>
        
        <div class="alert alert-info">
            <p>This report shows the results of optimizing trading strategies based on BTC price action signals.</p>
        </div>
"""
            
            # Add best strategy section
            if self.best_params and self.performance_metrics:
                return_class = "positive" if self.performance_metrics['total_return_pct'] > 0 else "negative"
                
                html_content += f"""
        <h2>Best Strategy</h2>
        <div class="row mb-4">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h3>{self.best_params['use_pattern']}</h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="metric-box">
                                    <div class="metric-title">Total Return</div>
                                    <div class="metric-value {return_class}">{self.performance_metrics['total_return_pct']:.2f}%</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-box">
                                    <div class="metric-title">Win Rate</div>
                                    <div class="metric-value">{self.performance_metrics['win_rate']*100:.1f}%</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-box">
                                    <div class="metric-title">Trades</div>
                                    <div class="metric-value">{self.performance_metrics['total_trades']}</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-box">
                                    <div class="metric-title">Max Drawdown</div>
                                    <div class="metric-value">{self.performance_metrics.get('max_drawdown', 0):.2f}%</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row mt-3">
                            <div class="col-md-12">
                                <h4>Optimal Parameters:</h4>
                                <ul>
                                    <li><strong>Pattern:</strong> {self.best_params['use_pattern']}</li>
                                    <li><strong>Lag:</strong> {self.best_params['pattern_lag']} periods</li>
                                    <li><strong>Stop Loss:</strong> {self.best_params['stop_loss_pct']}%</li>
                                    <li><strong>Take Profit:</strong> {self.best_params['take_profit_pct']}%</li>
                                    <li><strong>Trailing Stop:</strong> {self.best_params['trailing_stop_pct']}%</li>
                                    <li><strong>Max Hold Time:</strong> {self.best_params['max_holding_time']} periods</li>
                                    <li><strong>Position Size:</strong> {self.best_params['position_size_pct']}%</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h4>Additional Metrics</h4>
                    </div>
                    <div class="card-body">
                        <table class="table table-sm">
                            <tr>
                                <td>Avg Profit/Trade</td>
                                <td>{self.performance_metrics.get('avg_profit_per_trade', 0):.2f}%</td>
                            </tr>
                            <tr>
                                <td>Sharpe Ratio</td>
                                <td>{self.performance_metrics.get('sharpe_ratio', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>Profit Factor</td>
                                <td>{self.performance_metrics.get('profit_factor', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>Final Equity</td>
                                <td>${self.performance_metrics.get('final_equity', 10000):.2f}</td>
                            </tr>
                        </table>
                    </div>
                </div>
            </div>
        </div>
"""
            
            # Add equity curve chart
            html_content += f"""
        <div class="chart-container">
            <h2>Equity Curve</h2>
            <img src="../charts/equity_curve.png" class="img-fluid border rounded" alt="Equity Curve">
        </div>
"""
            
            # Add trade distribution chart if available
            if self.trade_list:
                html_content += f"""
        <div class="chart-container">
            <h2>Trade Distribution</h2>
            <img src="../charts/trade_distribution.png" class="img-fluid border rounded" alt="Trade Distribution">
        </div>
"""
            
            # Add parameter impact charts
            if self.metrics_by_parameter:
                html_content += """
        <h2>Parameter Impact Analysis</h2>
        <div class="row">
"""
                
                for param_name in self.metrics_by_parameter:
                    chart_path = f"../charts/{param_name}_impact.png"
                    # Check if chart file exists
                    chart_file = os.path.join(charts_dir, f"{param_name}_impact.png")
                    if os.path.exists(chart_file):
                        html_content += f"""
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header">
                        <h4>Impact of {param_name}</h4>
                    </div>
                    <div class="card-body">
                        <img src="{chart_path}" class="img-fluid border rounded" alt="{param_name} Impact">
                    </div>
                </div>
            </div>
"""
                
                html_content += """
        </div>
"""
            
            # Add trailing stop analysis if available
            if hasattr(self, 'trailing_stop_analysis'):
                html_content += f"""
        <div class="chart-container">
            <h2>Trailing Stop Analysis</h2>
            <img src="../charts/trailing_stop_impact.png" class="img-fluid border rounded" alt="Trailing Stop Impact">
            
            <div class="table-responsive mt-3">
                <table class="table table-striped table-sm">
                    <thead>
                        <tr>
                            <th>Trailing Stop %</th>
                            <th>Avg Return %</th>
                            <th>Win Rate %</th>
                            <th>Avg Trades</th>
                            <th>Combinations Tested</th>
                        </tr>
                    </thead>
                    <tbody>
"""
                
                for ts in sorted(self.trailing_stop_analysis.keys()):
                    metrics = self.trailing_stop_analysis[ts]
                    html_content += f"""
                        <tr>
                            <td>{ts}%</td>
                            <td>{metrics['avg_return']:.2f}%</td>
                            <td>{metrics['avg_win_rate']*100:.1f}%</td>
                            <td>{metrics['avg_trades']:.1f}</td>
                            <td>{metrics['count']}</td>
                        </tr>
"""
                
                html_content += """
                    </tbody>
                </table>
            </div>
        </div>
"""
            
            # Add top strategies table
            if self.results:
                # Sort results by return
                sorted_results = sorted(self.results, key=lambda x: x['metrics']['total_return_pct'], reverse=True)
                top_results = sorted_results[:10]  # Show top 10
                
                html_content += f"""
        <h2>Top 10 Strategies</h2>
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Pattern</th>
                        <th>Lag</th>
                        <th>Stop Loss</th>
                        <th>Take Profit</th>
                        <th>Trailing Stop</th>
                        <th>Hold Time</th>
                        <th>Return %</th>
                        <th>Win Rate</th>
                        <th>Trades</th>
                    </tr>
                </thead>
                <tbody>
"""
                
                for result in top_results:
                    params = result['params']
                    metrics = result['metrics']
                    return_class = "positive" if metrics['total_return_pct'] > 0 else "negative"
                    
                    html_content += f"""
                    <tr>
                        <td>{params['use_pattern']}</td>
                        <td>{params['pattern_lag']}</td>
                        <td>{params['stop_loss_pct']}%</td>
                        <td>{params['take_profit_pct']}%</td>
                        <td>{params['trailing_stop_pct']}%</td>
                        <td>{params['max_holding_time']}</td>
                        <td class="{return_class}">{metrics['total_return_pct']:.2f}%</td>
                        <td>{metrics['win_rate']*100:.1f}%</td>
                        <td>{metrics['total_trades']}</td>
                    </tr>
"""
                
                html_content += """
                </tbody>
            </table>
        </div>
"""
            
            # Close the HTML document
            html_content += """
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
            
            # Write the HTML file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            print(f"Generated HTML report: {output_file}")
            return output_file
                
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()

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

    def _analyze_trailing_stop_impact(self):
        """Analyze the impact of trailing stop on strategy performance."""
        print("Analyzing trailing stop impact...")
        
        # Group results by trailing stop percentage
        trailing_stop_results = {}
        for result in self.results:
            ts = result['params'].get('trailing_stop_pct', 0)
            if ts not in trailing_stop_results:
                trailing_stop_results[ts] = []
            trailing_stop_results[ts].append(result)
        
        # Calculate average metrics for each trailing stop value
        trailing_stop_metrics = {}
        for ts, results in trailing_stop_results.items():
            avg_return = np.mean([r['metrics']['total_return_pct'] for r in results])
            avg_win_rate = np.mean([r['metrics']['win_rate'] for r in results])
            avg_trades = np.mean([r['metrics']['total_trades'] for r in results])
            
            trailing_stop_metrics[ts] = {
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'avg_trades': avg_trades,
                'count': len(results)
            }
        
        # Store for reporting
        self.trailing_stop_analysis = trailing_stop_metrics
        
        print("Trailing stop analysis completed")
        
        # Return best trailing stop value based on return * win rate
        best_ts = max(trailing_stop_metrics.items(), 
                      key=lambda x: x[1]['avg_return'] * x[1]['avg_win_rate'], 
                      default=(None, {}))[0]
        
        if best_ts is not None:
            print(f"Best trailing stop value: {best_ts}%")
            print(f"  Avg Return: {trailing_stop_metrics[best_ts]['avg_return']:.2f}%")
            print(f"  Avg Win Rate: {trailing_stop_metrics[best_ts]['avg_win_rate']*100:.1f}%")
        
        return best_ts
