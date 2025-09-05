"""Generate trading strategies based on BTC price action leading altcoin moves."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

class DirectionalImpactStrategies:
    def __init__(self, combined_data, directional_impact_ml, altcoin_name):
        """Initialize with data and directional impact analysis results."""
        self.data = combined_data
        self.ml_results = directional_impact_ml
        self.altcoin_name = altcoin_name.upper()
        self.alt_prefix = altcoin_name.lower()
        self.strategies = {}
        self.backtest_results = {}
        
        # Find price column
        self.price_column = None
        for col in self.data.columns:
            if 'close' in col.lower() and self.alt_prefix in col.lower():
                self.price_column = col
                break
                
        if not self.price_column:
            for col in self.data.columns:
                if 'close' in col.lower():
                    self.price_column = col
                    break
        
        print(f"Using price column: {self.price_column}")

    def generate_strategies(self):
        """Generate strategies based on real-time synchronous directional impact analysis."""
        print("Generating real-time strategies from synchronous directional impact analysis...")
        
        for scenario in self.ml_results:
            # Use synchronous data (lag=0)
            if 0 not in self.ml_results[scenario]:
                print(f"No synchronous data available for {scenario}")
                continue
                
            sync_data = self.ml_results[scenario][0]
            
            # Get metrics
            mean_return = sync_data['mean_return']
            win_rate = sync_data['win_rate']
            correlation = sync_data.get('correlation', 0)
            total_trades = sync_data.get('total_trades', 0)
            
            print(f"\nStrategy for {scenario}:")
            print(f"  Real-time synchronous analysis")
            print(f"  Mean return: {mean_return:.4f}%")
            print(f"  Win rate: {win_rate*100:.1f}%")
            print(f"  BTC correlation: {correlation:.3f}")
            print(f"  Total trades: {total_trades}")
            
            # Create strategy parameters based on scenario
            is_long = mean_return > 0
            
            # Calculate optimal stop-loss and take-profit based on price action
            if 'strong' in scenario:
                sl_pct = 2.0
                tp_pct = 5.0
                ts_pct = 3.0  # Trailing stop
            elif 'medium' in scenario:
                sl_pct = 1.5
                tp_pct = 3.0
                ts_pct = 2.0
            else:
                sl_pct = 1.0
                tp_pct = 2.0
                ts_pct = 1.0
            
            # Store strategy info
            self.strategies[scenario] = {
                'name': scenario.replace('btc_', '').replace('_', ' ').title(),
                'entry_signals': self.data[scenario],
                'exit_signals': pd.Series(False, index=self.data.index),
                'is_long': is_long,
                'synchronous': True,  # Flag for real-time analysis
                'lag': 0,  # No lag for synchronous
                'mean_return': mean_return,
                'win_rate': win_rate,
                'correlation': correlation,
                'total_trades': total_trades,
                'stop_loss': sl_pct,
                'take_profit': tp_pct,
                'trailing_stop': ts_pct
            }
            
            print(f"  Strategy type: {'LONG' if is_long else 'SHORT'}")
            print(f"  Stop loss: {sl_pct}%")
            print(f"  Take profit: {tp_pct}%")
            print(f"  Trailing stop: {ts_pct}%")
        
        return self.strategies
    
    def backtest_strategies(self, initial_capital=10000):
        """Backtest strategies with holding times and trailing stops."""
        print("\nBacktesting strategies with holding time and trailing stop optimization...")
        
        # Fix for index frequency error
        if 'index' in dir(self.data) and hasattr(self.data.index, 'freq') and self.data.index.freq is None:
            # Try to infer frequency
            try:
                # Check for datetime index and fix it
                if isinstance(self.data.index, pd.DatetimeIndex):
                    inferred_freq = pd.infer_freq(self.data.index)
                    if inferred_freq:
                        self.data = self.data.copy()
                        self.data.index.freq = inferred_freq
                    else:
                        # Manually set frequency based on data inspection
                        time_diffs = self.data.index.to_series().diff().dropna()
                        if len(time_diffs) > 0:
                            median_seconds = time_diffs.median().total_seconds()
                            if median_seconds <= 60:
                                print("  Setting frequency to 1min based on data")
                                self.data = self.data.asfreq('1min')
                            elif median_seconds <= 300:
                                print("  Setting frequency to 5min based on data")
                                self.data = self.data.asfreq('5min')
                            else:
                                print("  Setting frequency to 15min based on data")
                                self.data = self.data.asfreq('15min')
                else:
                    print("  Warning: Index is not DatetimeIndex, frequency not set")
            except Exception as e:
                print(f"  Warning: Could not set frequency: {e}")
        
        # Holding times to test (in minutes)
        holding_times = [1, 5, 15, 20, 30, 45, 60, 90, 120, 240, 480, 720]
        
        # Trailing stop values to test (in percent)
        trailing_stops = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        
        # Loop through strategies
        for scenario, strategy in self.strategies.items():
            print(f"\nBacktesting {strategy['name']}...")
            
            try:
                # Get price data
                price = self.data[self.price_column]
                
                # Get entry signals based on the scenario
                # We delay by the optimal lag to enter at the right time
                lag = strategy['lag']
                entries = self.data[scenario].shift(lag).fillna(False)
                
                # Test different combinations of holding times and trailing stops
                best_score = -float('inf')
                best_holding_time = None
                best_trailing_stop = None
                best_portfolio = None
                combination_results = {}
                
                print("  Testing combinations of holding times and trailing stops...")
                
                for holding_time in holding_times:
                    for trailing_stop in trailing_stops:
                        try:
                            # Use proper vectorized backtesting with stop-loss, take-profit, and trailing stop
                            tp_pct = strategy['take_profit'] / 100.0
                            sl_pct = strategy['stop_loss'] / 100.0
                            ts_pct = trailing_stop / 100.0  # Use current trailing stop from loop
                            
                            import vectorbt as vbt
                            from vectorbt.portfolio.enums import SizeType
                            
                            # Run backtest with proper parameters - remove exits to avoid conflicts
                            pf = vbt.Portfolio.from_signals(
                                price,
                                entries,
                                size=0.1,  # Use 10% of capital per trade
                                size_type=SizeType.Percent,
                                init_cash=initial_capital,
                                fees=0.001,  # 0.1% trading fee
                                freq=f"{holding_time}min",  # Set proper frequency
                                sl_stop=sl_pct,
                                tp_stop=tp_pct,
                                sl_trail=ts_pct  # Use trailing stop
                            )
                            
                            # Calculate equity curve
                            equity_curve = pf.value()
                                
                            # Calculate performance metrics
                            total_return = pf.total_return() * 100
                            win_rate = pf.trades.win_rate() if pf.trades.count() > 0 else 0
                            total_trades = pf.trades.count()
                            max_drawdown = pf.max_drawdown() * 100 if pf.trades.count() > 0 else 0
                            profit_factor = pf.trades.profit_factor() if pf.trades.count() > 0 else 0
                            sharpe_ratio = pf.sharpe_ratio() if pf.trades.count() > 0 else 0
                            final_value = pf.final_value()
                            
                            # Calculate additional trade details
                            avg_trade_return = 0
                            avg_win_return = 0
                            avg_loss_return = 0
                            avg_trade_duration = 0
                            winning_trades = 0
                            losing_trades = 0
                            
                            if total_trades > 0:
                                trade_returns = pf.trades.returns
                                avg_trade_return = trade_returns.mean() * 100
                                
                                # Separate winning and losing trades
                                winning_returns = trade_returns[trade_returns > 0]
                                losing_returns = trade_returns[trade_returns <= 0]
                                
                                winning_trades = len(winning_returns)
                                losing_trades = len(losing_returns)
                                
                                if len(winning_returns) > 0:
                                    avg_win_return = winning_returns.mean() * 100
                                if len(losing_returns) > 0:
                                    avg_loss_return = losing_returns.mean() * 100
                                
                                # Average trade duration
                                trade_durations = pf.trades.duration
                                if len(trade_durations) > 0:
                                    avg_trade_duration = trade_durations.mean().total_seconds() / 60  # in minutes
                            
                            # Store results for this combination
                            combo_key = (holding_time, trailing_stop)
                            combination_results[combo_key] = {
                                'portfolio': pf,
                                'equity_curve': equity_curve,
                                'total_return_pct': total_return,
                                'win_rate': win_rate,
                                'total_trades': total_trades,
                                'profit_factor': profit_factor,
                                'holding_time': holding_time,
                                'trailing_stop_pct': trailing_stop,
                                'max_drawdown_pct': max_drawdown,
                                'sharpe_ratio': sharpe_ratio,
                                'final_value': final_value,
                                'avg_trade_return_pct': avg_trade_return,
                                'avg_win_return_pct': avg_win_return,
                                'avg_loss_return_pct': avg_loss_return,
                                'avg_trade_duration_min': avg_trade_duration,
                                'winning_trades': winning_trades,
                                'losing_trades': losing_trades
                            }
                            
                            # Enhanced logging with much more detail
                            print(f"    ╭─ Hold: {holding_time:>3}min, Trail: {trailing_stop:>4.1f}% ─────────────────────────────────────────╮")
                            print(f"    │ 💰 Return: {total_return:>8.2f}% | Final Value: ${final_value:>10,.0f} | Drawdown: {max_drawdown:>6.2f}% │")
                            print(f"    │ 🎯 Win Rate: {win_rate*100:>6.1f}% | Trades: {total_trades:>3} | Sharpe: {sharpe_ratio:>7.2f} | PF: {profit_factor:>6.2f} │")
                            if total_trades > 0:
                                print(f"    │ 📊 Avg Trade: {avg_trade_return:>6.2f}% | Wins: {winning_trades:>2}({avg_win_return:>6.2f}%) | Loss: {losing_trades:>2}({avg_loss_return:>6.2f}%) │")
                                print(f"    │ ⏱️  Avg Duration: {avg_trade_duration:>6.1f}min                                           │")
                            else:
                                print(f"    │ ❌ No trades executed - insufficient signals or conditions not met      │")
                            print(f"    ╰─────────────────────────────────────────────────────────────────────────────╯")
                            
                            # Score based on return * win rate, only if we have enough trades
                            if pf.trades.count() >= 5:
                                score = total_return * win_rate
                                if score > best_score:
                                    best_score = score
                                    best_holding_time = holding_time
                                    best_trailing_stop = trailing_stop
                                    best_portfolio = pf
                        
                        except Exception as e:
                            print(f"    Error testing combination {holding_time}min/{trailing_stop}%: {e}")
                            continue
                
                # Select best results if any
                if best_holding_time is not None and best_trailing_stop is not None:
                    best_combo = (best_holding_time, best_trailing_stop)
                    print(f"  Best combination: Holding time {best_holding_time} minutes, Trailing stop {best_trailing_stop}%")
                    self.backtest_results[scenario] = combination_results[best_combo]
                    self.backtest_results[scenario]['all_combination_results'] = combination_results
                    
                    # Update the strategy dictionary with optimal parameters
                    self.strategies[scenario]['optimal_holding_time'] = best_holding_time
                    self.strategies[scenario]['optimal_trailing_stop'] = best_trailing_stop
                    
                    # Print metrics for best combination
                    result = self.backtest_results[scenario]
                    print(f"  Total return: {result['total_return_pct']:.2f}%")
                    print(f"  Win rate: {result['win_rate']*100:.1f}%")
                    print(f"  Total trades: {result['total_trades']}")
                    print(f"  Profit factor: {result.get('profit_factor', 0):.2f}")
                else:
                    print("  No valid combination found")
                    self.backtest_results[scenario] = {
                        'portfolio': None,
                        'equity_curve': pd.Series(),  # Empty series
                        'total_return_pct': 0,
                        'win_rate': 0,
                        'total_trades': 0,
                        'holding_time': strategy['lag'],  # Default to the lag
                        'trailing_stop_pct': strategy['trailing_stop'], # Default to strategy trailing stop
                        'error': "No valid holding time/trailing stop combination found"
                    }
            
            except Exception as e:
                print(f"  Error backtesting {strategy['name']}: {e}")
                self.backtest_results[scenario] = {
                    'portfolio': None,
                    'equity_curve': pd.Series(),  # Empty series
                    'total_return_pct': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'error': str(e)
                }
        
        # Generate trailing stop analysis visualization
        # Note: Chart will be generated later in generate_strategy_reports with proper path
        self._generate_trailing_stop_analysis()
        
        return self.backtest_results
    
    def generate_strategy_reports(self, results_dirs):
        """Generate HTML reports and charts for strategy backtests."""
        print("\nGenerating strategy reports and charts...")
        
        # Get the proper directories
        charts_dir = results_dirs['charts']
        html_dir = results_dirs['html']
        
        # Make sure directories exist
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(html_dir, exist_ok=True)
        
        # Generate charts for each strategy
        for scenario, result in self.backtest_results.items():
            strategy_name = self.strategies[scenario]['name'].replace(' ', '_').lower()
            
            try:
                # Generate equity curve chart
                if 'equity_curve' in result and isinstance(result['equity_curve'], (pd.Series, pd.DataFrame)) and not result['equity_curve'].empty:
                    equity_curve_path = os.path.join(charts_dir, f'equity_curve_{strategy_name}.png')
                    
                    # Create the equity curve chart
                    plt.figure(figsize=(10, 6))
                    plt.plot(result['equity_curve'], color='blue', linewidth=2)
                    plt.title(f'Equity Curve - {self.strategies[scenario]["name"]}')
                    plt.xlabel('Date')
                    plt.ylabel('Portfolio Value')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(equity_curve_path)
                    plt.close()
                    
                    print(f"  Generated equity curve chart for {strategy_name}")
                else:
                    # Create placeholder chart for missing equity curve
                    equity_curve_path = os.path.join(charts_dir, f'equity_curve_{strategy_name}.png')
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, "Equity curve data not available", 
                            ha='center', va='center', fontsize=14)
                    plt.title(f'Equity Curve - {self.strategies[scenario]["name"]}')
                    plt.savefig(equity_curve_path)
                    plt.close()
                    
                    print(f"  Created placeholder equity curve chart for {strategy_name}")
            except Exception as e:
                print(f"  Error generating chart for {strategy_name}: {e}")
        
        # Generate HTML report with all strategies
        html_output = os.path.join(html_dir, 'directional_strategies.html')
        self._generate_html_report(html_output, charts_dir)
        
        # Generate trailing stop analysis chart with correct path
        self._generate_trailing_stop_analysis(charts_dir)
        
        print(f"Strategy reports generated at {html_output}")
        return html_output

    def _generate_trailing_stop_analysis(self, charts_dir=None):
        """Generate analysis of trailing stop impact across all strategies."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Collect data on trailing stop performance
        trailing_stop_data = {}
        
        for scenario, result in self.backtest_results.items():
            if 'all_combination_results' not in result:
                continue
            
            for combo, combo_result in result['all_combination_results'].items():
                holding_time, trailing_stop = combo
                
                if trailing_stop not in trailing_stop_data:
                    trailing_stop_data[trailing_stop] = []
                    
                trailing_stop_data[trailing_stop].append({
                    'return': combo_result['total_return_pct'],
                    'win_rate': combo_result['win_rate'],
                    'trades': combo_result['total_trades'],
                    'scenario': scenario
                })
        
        # Calculate averages for each trailing stop value
        ts_analysis = {}
        for ts, data in trailing_stop_data.items():
            if not data:
                continue
                
            avg_return = np.mean([d['return'] for d in data])
            avg_win_rate = np.mean([d['win_rate'] for d in data])
            avg_trades = np.mean([d['trades'] for d in data])
            
            ts_analysis[ts] = {
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'avg_trades': avg_trades,
                'count': len(data)
            }
        
        # Store for reporting
        self.trailing_stop_analysis = ts_analysis
        
        # Generate visualization if we have data and charts_dir is provided
        if ts_analysis and charts_dir:
            try:
                plt.figure(figsize=(10, 8))
                
                ts_values = sorted(ts_analysis.keys())
                returns = [ts_analysis[ts]['avg_return'] for ts in ts_values]
                win_rates = [ts_analysis[ts]['avg_win_rate'] * 100 for ts in ts_values]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Plot returns
                ax1.bar(ts_values, returns, color='blue', alpha=0.7)
                ax1.set_ylabel('Average Return %')
                ax1.set_title('Trailing Stop Impact Analysis')
                ax1.grid(True, alpha=0.3)
                
                # Plot win rates
                ax2.bar(ts_values, win_rates, color='green', alpha=0.7)
                ax2.set_xlabel('Trailing Stop %')
                ax2.set_ylabel('Average Win Rate %')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save to the correct charts directory
                chart_path = os.path.join(charts_dir, 'trailing_stop_impact.png')
                plt.savefig(chart_path)
                print(f"Trailing stop analysis chart saved to: {chart_path}")
                
                plt.close()
                
            except Exception as e:
                print(f"Error generating trailing stop chart: {str(e)}")
        elif ts_analysis and not charts_dir:
            # If no charts_dir provided, still store the analysis but don't generate chart
            print("Trailing stop analysis data collected, chart will be generated later with proper path")
    
    def _generate_html_report(self, output_file, charts_dir, strategy_reports=None, trade_detail_reports=None):
        """Generate HTML report for strategies."""
        # Sort strategies by return
        sorted_strategies = sorted(
            self.backtest_results.items(), 
            key=lambda x: x[1]['total_return_pct'], 
            reverse=True
        )
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Price Action Strategies for {self.altcoin_name}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .strategy-card {{ margin-bottom: 30px; }}
        .metric-box {{ padding: 10px; margin: 5px; background-color: #f8f9fa; border-radius: 4px; text-align: center; }}
        .metric-title {{ font-size: 0.9rem; color: #6c757d; }}
        .metric-value {{ font-size: 1.1rem; font-weight: bold; }}
        .metric-subtitle {{ font-size: 0.7rem; color: #6c757d; font-style: italic; }}
        .up-value {{ color: #28a745; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">BTC Price Action Trading Strategies for {self.altcoin_name}</h1>
        
        <div class="alert alert-info">
            <p>This report shows trading strategies based on BTC price action leading to {self.altcoin_name} movements after specific time lags.</p>
        </div>
        
        <h2>Strategy Performance Summary</h2>
        <div class="table-responsive mb-4">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>BTC Movement</th>
                        <th>Total Return</th>
                        <th>Win Rate</th>
                        <th>Trades</th>
                        <th>Optimal Lag</th>
                        <th>Holding Time</th>
                        <th>Trailing Stop</th>
                    </tr>
                </thead>
                <tbody>
                """
        
        for scenario, results in sorted_strategies:
            strategy = self.strategies[scenario]
            return_class = 'text-success' if results['total_return_pct'] > 0 else 'text-danger'
            
            html += f"""
                    <tr>
                        <td>{strategy['name']}</td>
                        <td>{strategy['lag']} min</td>
                        <td class="{return_class}">{results['total_return_pct']:.2f}%</td>
                        <td>{results['win_rate']*100 if isinstance(results['win_rate'], (int, float)) else 0:.1f}%</td>
                        <td>{results['total_trades']}</td>
                        <td>{strategy.get('lag', 'N/A')} min</td>
                        <td>{results.get('holding_time', 'N/A')} min</td>
                        <td>{results.get('trailing_stop_pct', strategy.get('trailing_stop', 'N/A'))}%</td>
                    </tr>
            """
                
        html += """
                </tbody>
            </table>
        </div>
        """
        
        # Add trailing stop analysis if available
        if hasattr(self, 'trailing_stop_analysis'):
            html += """
            <div class="my-4">
                <h2>Trailing Stop Impact Analysis</h2>
                <p>This analysis shows how different trailing stop percentages affect strategy performance.</p>
                
                <div class="mb-3">
                    <img src="../charts/trailing_stop_impact.png" class="img-fluid border rounded" alt="Trailing Stop Impact Analysis">
                </div>
                
                <div class="table-responsive">
                    <table class="table table-sm table-bordered">
                        <thead>
                            <tr>
                                <th>Trailing Stop %</th>
                                <th>Avg Return %</th>
                                <th>Avg Win Rate %</th>
                                <th>Avg Trades</th>
                                <th>Combinations Tested</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for ts in sorted(self.trailing_stop_analysis.keys()):
                metrics = self.trailing_stop_analysis[ts]
                html += f"""
                            <tr>
                                <td>{ts}%</td>
                                <td>{metrics['avg_return']:.2f}%</td>
                                <td>{metrics['avg_win_rate']*100:.1f}%</td>
                                <td>{metrics['avg_trades']:.1f}</td>
                                <td>{metrics['count']}</td>
                            </tr>
                """
                
            html += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
        
        # Individual strategy details section
        html += """
            <h2>Individual Strategy Details</h2>
            
            <div class="row">
        """
        
        # Add a card for each strategy
        for scenario, results in sorted_strategies:
            strategy = self.strategies[scenario]
            strategy_name = strategy['name'].replace(' ', '_').lower()
            return_class = 'text-success' if results['total_return_pct'] > 0 else 'text-danger'
            
            html += f"""
                <div class="col-md-6 mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h3>{strategy['name']}</h3>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <img src="../charts/equity_curve_{strategy_name}.png" class="img-fluid border rounded" alt="Equity Curve">
                            </div>
                            
                            <div class="row mb-3">
                                <div class="col-6">
                                    <div class="metric-box">
                                        <div class="metric-title">Return</div>
                                        <div class="metric-value {return_class.replace('text-', '')}">{results['total_return_pct']:.2f}%</div>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="metric-box">
                                        <div class="metric-title">Win Rate</div>
                                        <div class="metric-value">{results['win_rate']*100 if isinstance(results['win_rate'], (int, float)) else 0:.1f}%</div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row mb-3">
                                <div class="col-4">
                                    <div class="metric-box">
                                        <div class="metric-title">BTC Pattern</div>
                                        <div class="metric-value">{scenario.replace('btc_', '')}</div>
                                    </div>
                                </div>
                                <div class="col-4">
                                    <div class="metric-box">
                                        <div class="metric-title">Optimal Lag</div>
                                        <div class="metric-value">{strategy.get('lag', 'N/A')} min</div>
                                    </div>
                                </div>
                                <div class="col-4">
                                    <div class="metric-box">
                                        <div class="metric-title">Trades</div>
                                        <div class="metric-value">{results['total_trades']}</div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-4">
                                    <div class="metric-box">
                                        <div class="metric-title">Stop Loss</div>
                                        <div class="metric-value">{strategy.get('stop_loss', 'N/A')}%</div>
                                    </div>
                                </div>
                                <div class="col-4">
                                    <div class="metric-box">
                                        <div class="metric-title">Take Profit</div>
                                        <div class="metric-value">{strategy.get('take_profit', 'N/A')}%</div>
                                    </div>
                                </div>
                                <div class="col-4">
                                    <div class="metric-box">
                                        <div class="metric-title">Trailing Stop</div>
                                        <div class="metric-value">{results.get('trailing_stop_pct', strategy.get('trailing_stop', 'N/A'))}%</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        # Close the HTML
        html += """
            </div>
        </div>
    </body>
</html>
        """
        
        # Write the HTML file
        with open(output_file, 'w') as f:
            f.write(html)
        
        return html