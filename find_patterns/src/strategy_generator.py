"""Generate trading strategies based on ML directional impact analysis."""

import os
import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt

class DirectionalImpactStrategies:
    def __init__(self, combined_data, directional_impact_ml, altcoin_name):
        """Initialize with data and ML analysis results."""
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
    
    def _get_backtest_duration(self):
        """Calculate the duration of the backtest data."""
        if len(self.data) < 2:
            return "N/A"
            
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        
        # Calculate duration
        duration_days = (end_date - start_date).days
        
        if duration_days < 1:
            hours = (end_date - start_date).seconds / 3600
            return f"{hours:.1f} hours"
        elif duration_days < 30:
            return f"{duration_days} days"
        elif duration_days < 365:
            months = duration_days / 30.4  # Approximate months
            return f"{months:.1f} months"
        else:
            years = duration_days / 365.25  # Approximate years
            return f"{years:.1f} years"

    def generate_strategies(self):
        """Generate strategies based on ML results."""
        print("Generating strategies from ML directional impact analysis...")
        
        for scenario in self.ml_results:
            # Find best minute with highest accuracy
            best_minute = max(self.ml_results[scenario].keys(), 
                             key=lambda x: self.ml_results[scenario][x]['accuracy'])
            
            # Get ML model for this scenario
            model = self.ml_results[scenario][best_minute]['model']
            
            # Get top features for reporting only
            feature_importance = self.ml_results[scenario][best_minute]['feature_importance']
            top_features = [f[0] for f in feature_importance[:5]]
            
            print(f"\nStrategy for {scenario}:")
            print(f"  Best prediction minute: {best_minute}")
            print(f"  Prediction accuracy: {self.ml_results[scenario][best_minute]['accuracy']*100:.1f}%")
            print(f"  Top features: {', '.join(top_features)}")
            
            # IMPORTANT: Get all features that were used during training, not just from feature_importance
            try:
                # Get the actual feature names the model was trained on
                all_features = model.feature_names_in_  # For newer scikit-learn versions
            except AttributeError:
                # Fallback for older scikit-learn versions
                all_features = [f[0] for f in feature_importance]  # Use what we have
                print(f"  Warning: Using extracted feature names - may be incomplete")
            
            # Check if all features are in the data
            missing_features = [f for f in all_features if f not in self.data.columns]
            if missing_features:
                print(f"  Warning: Missing {len(missing_features)} features from training data:")
                print(f"  {', '.join(missing_features[:5])}{'...' if len(missing_features) > 5 else ''}")
                continue
            
            # Create feature matrix with exact same columns as training in exact same order
            X = self.data[all_features].fillna(0)
            
            # Generate predictions
            try:
                predictions = model.predict(X)
                print(f"  Generated {sum(predictions)} signals out of {len(predictions)} data points")
            except Exception as e:
                print(f"  Error generating predictions: {e}")
                continue
            
            # Rest of the code remains the same...
            # Create strategy parameters based on scenario
            # If up scenario, go long when model predicts up
            # If down scenario, go short when model predicts down
            is_long = 'up' in scenario
            signal_col = f"{scenario}_ml_signal"
            entry_col = f"{scenario}_ml_entry"
            exit_col = f"{scenario}_ml_exit"
            
            # Calculate optimal stop-loss and take-profit
            if 'strong' in scenario:
                sl_pct = 2.0
                tp_pct = 6.0
            elif 'medium' in scenario:
                sl_pct = 1.5
                tp_pct = 4.5
            else:
                sl_pct = 1.0
                tp_pct = 3.0
                
            # Adjust SL/TP based on direction
            if not is_long:
                sl_pct, tp_pct = tp_pct, sl_pct  # Reverse for short trades
            
            # Store strategy info
            self.strategies[scenario] = {
                'name': scenario.replace('btc_', '').replace('_', ' ').title(),
                'predictions': predictions,
                'entry_signals': pd.Series(predictions, index=self.data.index),
                'exit_signals': pd.Series(False, index=self.data.index),  # Use SL/TP for exits
                'is_long': is_long,
                'features': all_features,  # Store ALL features used
                'best_minute': best_minute,
                'accuracy': self.ml_results[scenario][best_minute]['accuracy'],
                'stop_loss_pct': sl_pct,
                'take_profit_pct': tp_pct
            }
            
            print(f"  Strategy type: {'LONG' if is_long else 'SHORT'}")
            print(f"  Stop loss: {sl_pct}%")
            print(f"  Take profit: {tp_pct}%")
        
        return self.strategies
    
    def backtest_strategies(self, initial_capital=10000):
        """Backtest generated strategies using VectorBT with holding time optimization."""
        print("\nBacktesting strategies with holding time optimization...")
        
        # Holding times to test (in minutes) - more short-term options
        holding_times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 45, 60, 90, 120, 240, 480, 720]
        
        # Loop through strategies
        for scenario, strategy in self.strategies.items():
            print(f"\nBacktesting {strategy['name']}...")
            
            try:
                # Get price data
                price = self.data[self.price_column]
                
                # Get entry signals
                entries = strategy['entry_signals'].fillna(False)
                
                # Test different holding times
                best_return = -float('inf')
                best_holding_time = None
                best_portfolio = None
                holding_time_results = {}
                
                print("  Testing different holding times...")
                
                for holding_time in holding_times:
                    # Create time-based exit signals
                    exits = pd.Series(False, index=self.data.index)
                    
                    # For each entry, create an exit signal after holding_time periods
                    entry_indices = entries[entries].index
                    for entry_idx in entry_indices:
                        try:
                            # Find the index position
                            pos = self.data.index.get_loc(entry_idx)
                            # If we can exit holding_time bars later, set exit
                            if pos + holding_time < len(self.data):
                                exit_idx = self.data.index[pos + holding_time]
                                exits.loc[exit_idx] = True
                        except:
                            continue
                    
                    # Configure parameters based on long or short
                    size = 0.95  # 95% of capital
                    sl_stop = strategy['stop_loss_pct']/100
                    tp_stop = strategy['take_profit_pct']/100

                    # Run backtest with both time-based exits and SL/TP
                    pf = vbt.Portfolio.from_signals(
                        close=price,
                        entries=entries,
                        exits=exits,
                        init_cash=initial_capital,
                        fees=0.001,
                        size=size,
                        size_type='percent',
                        sl_stop=sl_stop,
                        tp_stop=tp_stop,
                        direction='longonly' if strategy['is_long'] else 'shortonly'
                    )
                    
                    # Calculate performance metrics
                    total_return = pf.total_return() * 100
                    sharpe = pf.sharpe_ratio() if hasattr(pf, 'sharpe_ratio') else 0
                    win_rate = pf.trades.win_rate() if pf.trades.count() > 0 else 0
                    
                    # Score is return * win_rate * sharpe to balance these factors
                    # This rewards strategies with good returns, consistency, and risk-adjusted performance
                    score = total_return * (win_rate if win_rate > 0 else 0.01) * (sharpe if sharpe > 0 else 0.01)
                    
                    # Store results for this holding time
                    holding_time_results[holding_time] = {
                        'portfolio': pf,
                        'total_return_pct': total_return,
                        'win_rate': win_rate,
                        'total_trades': pf.trades.count(),
                        'profit_factor': pf.trades.profit_factor() if pf.trades.count() > 0 else 0,
                        'sharpe_ratio': sharpe,
                        'max_drawdown': pf.max_drawdown() * 100,
                        'score': score
                    }
                    
                    print(f"    Holding time {holding_time} min: {total_return:.2f}% return, {win_rate*100:.1f}% win rate, {pf.trades.count()} trades")
                    
                    # Update best if this is better
                    if score > best_return and pf.trades.count() >= 5:
                        best_return = score
                        best_holding_time = holding_time
                        best_portfolio = pf
                
                # Use the best holding time results
                if best_holding_time:
                    print(f"  Best holding time: {best_holding_time} minutes")
                    self.backtest_results[scenario] = holding_time_results[best_holding_time]
                    self.backtest_results[scenario]['holding_time'] = best_holding_time
                    self.backtest_results[scenario]['all_holding_time_results'] = holding_time_results
                    
                    # Update the strategy dictionary with optimal holding time
                    self.strategies[scenario]['optimal_holding_time'] = best_holding_time
                    
                    # Print metrics for best holding time
                    result = self.backtest_results[scenario]
                    print(f"  Total return: {result['total_return_pct']:.2f}%")
                    print(f"  Win rate: {result['win_rate']*100:.1f}%")
                    print(f"  Total trades: {result['total_trades']}")
                    print(f"  Profit factor: {result.get('profit_factor', 0):.2f}")
                    print(f"  Max drawdown: {result.get('max_drawdown', 0):.2f}%")
                else:
                    print("  No valid holding time found. Using default.")
                    # Use the last result as fallback
                    if holding_time_results:
                        last_time = list(holding_time_results.keys())[-1]
                        self.backtest_results[scenario] = holding_time_results[last_time]
                        self.backtest_results[scenario]['holding_time'] = last_time
                        self.strategies[scenario]['optimal_holding_time'] = last_time
                    else:
                        self.backtest_results[scenario] = {
                            'portfolio': None,
                            'total_return_pct': 0,
                            'win_rate': 0,
                            'total_trades': 0,
                            'error': "No valid holding time found"
                        }
                    
            except Exception as e:
                print(f"  Error backtesting {strategy['name']}: {e}")
                self.backtest_results[scenario] = {
                    'portfolio': None,
                    'total_return_pct': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'error': str(e)
                }
        
        return self.backtest_results
    
    def generate_strategy_reports(self, results_dirs):
        """Generate comprehensive reports for backtested strategies."""
        print("\nGenerating strategy reports...")
        
        # Create charts directory if it doesn't exist
        charts_dir = results_dirs['charts']
        reports_dir = results_dirs['reports']
        html_dir = results_dirs['html']
        
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(html_dir, exist_ok=True)
        
        # Summary report containing all strategies
        summary_report_path = os.path.join(reports_dir, 'directional_strategies_report.txt')
        summary_html_path = os.path.join(html_dir, 'directional_strategies.html')
        
        # Text summary report
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            # Header section
            f.write(f"DIRECTIONAL ML STRATEGIES FOR {self.altcoin_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Sort strategies by return
            sorted_strategies = sorted(
                self.backtest_results.items(), 
                key=lambda x: x[1]['total_return_pct'], 
                reverse=True
            )
            
            # Write summary of all strategies
            f.write(f"SUMMARY ({len(sorted_strategies)} strategies)\n")
            f.write("-" * 30 + "\n\n")
            f.write(f"Backtest period: {self._get_backtest_duration()}\n")
            f.write(f"Data points analyzed: {len(self.data)}\n\n")
            f.write("Strategy               | Return %  | Win Rate | Trades | Holding Time\n")
            f.write("-" * 75 + "\n")
            
            for scenario, results in sorted_strategies:
                strategy = self.strategies[scenario]
                name = strategy['name'][:18]  # Limit name length for better formatting
                hold_time = strategy.get('optimal_holding_time', 'N/A')
                f.write(f"{name:<20} | {results['total_return_pct']:>8.2f}% | {results['win_rate']*100:>6.1f}% | {results['total_trades']:>5} | {hold_time} min\n")
            
            f.write("\n\n")
            f.write("=" * 50 + "\n\n")
            f.write("DETAILED STRATEGY REPORTS\n\n")
            
            # Write details for each strategy
            for scenario, results in sorted_strategies:
                strategy = self.strategies[scenario]
                f.write("\n" + "=" * 50 + "\n")
                f.write(f"{strategy['name'].upper()}\n")
                f.write("-" * len(strategy['name']) + "\n\n")
                
                # Strategy basics
                f.write("STRATEGY SETUP:\n")
                f.write(f"• Type: {'Long' if strategy['is_long'] else 'Short'}\n")
                f.write(f"• Best prediction minute: {strategy['best_minute']}\n")
                f.write(f"• ML model accuracy: {strategy['accuracy']*100:.1f}%\n")
                f.write(f"• Stop loss: {strategy['stop_loss_pct']}%\n")
                f.write(f"• Take profit: {strategy['take_profit_pct']}%\n")
                f.write(f"• Optimal holding time: {strategy.get('optimal_holding_time', 'N/A')} minutes\n\n")
                
                # Performance metrics
                f.write("PERFORMANCE METRICS:\n")
                backtest_duration = self._get_backtest_duration()
                f.write(f"• Total return: {results['total_return_pct']:.2f}% (over {backtest_duration})\n")
                f.write(f"• Win rate: {results['win_rate']*100:.1f}%\n")
                f.write(f"• Total trades: {results['total_trades']}\n")
                f.write(f"• Profit factor: {results.get('profit_factor', 0):.2f}\n")
                f.write(f"• Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"• Max drawdown: {results.get('max_drawdown', 0):.2f}%\n\n")
                
                # Top features
                f.write("TOP PREDICTIVE FEATURES:\n")
                for i, (feature, importance) in enumerate(self.ml_results[scenario][strategy['best_minute']]['feature_importance'][:5], 1):
                    f.write(f"  {i}. {feature}: {importance*100:.2f}%\n")
                
                f.write("\n")
        
        # Create individual strategy text reports
        strategy_reports = {}
        trade_detail_reports = {}  # Add this line to keep track of trade detail files

        for scenario, results in sorted_strategies:
            strategy = self.strategies[scenario]
            
            # Create a detailed text file for each strategy
            strategy_file_name = f"strategy_{scenario.replace('btc_', '')}.txt"
            strategy_file_path = os.path.join(reports_dir, strategy_file_name)
            
            # Add code to create trade details file
            trade_details_file_name = f"trades_{scenario.replace('btc_', '')}.csv"
            trade_details_file_path = os.path.join(reports_dir, trade_details_file_name)
            
            # Export trade details if portfolio exists
            if results['portfolio'] is not None:
                if self._export_trade_details(results['portfolio'], trade_details_file_path):
                    trade_detail_reports[scenario] = trade_details_file_name
            
            with open(strategy_file_path, 'w', encoding='utf-8') as sf:
                sf.write(f"DETAILED BACKTEST RESULTS: {strategy['name'].upper()}\n")
                sf.write("=" * 50 + "\n\n")
                
                # Basic strategy information
                sf.write("STRATEGY SETUP\n")
                sf.write("-" * 30 + "\n")
                sf.write(f"Type: {'Long' if strategy['is_long'] else 'Short'}\n")
                sf.write(f"Best prediction minute: {strategy['best_minute']}\n")
                sf.write(f"Optimal holding time: {strategy.get('optimal_holding_time', 'N/A')} minutes\n")
                sf.write(f"ML model accuracy: {strategy['accuracy']*100:.1f}%\n")
                sf.write(f"Stop loss: {strategy['stop_loss_pct']}%\n")
                sf.write(f"Take profit: {strategy['take_profit_pct']}%\n\n")
                
                # Performance metrics
                sf.write("PERFORMANCE METRICS\n")
                sf.write("-" * 30 + "\n")
                sf.write(f"Total return: {results['total_return_pct']:.2f}% (over {self._get_backtest_duration()})\n")
                sf.write(f"Win rate: {results['win_rate']*100:.1f}%\n")
                sf.write(f"Total trades: {results['total_trades']}\n")
                sf.write(f"Profit factor: {results.get('profit_factor', 0):.2f}\n")
                sf.write(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}\n")
                sf.write(f"Max drawdown: {results.get('max_drawdown', 0):.2f}%\n\n")
                
                # Top features
                sf.write("TOP PREDICTIVE FEATURES\n")
                sf.write("-" * 30 + "\n")
                for feature, importance in self.ml_results[scenario][strategy['best_minute']]['feature_importance'][:10]:
                    sf.write(f"{feature}: {importance*100:.2f}%\n")
                sf.write("\n")
                
                # Holding time analysis
                sf.write("HOLDING TIME ANALYSIS\n")
                sf.write("-" * 30 + "\n")
                
                if 'all_holding_time_results' in results:
                    # Sort by return
                    sorted_times = sorted(
                        results['all_holding_time_results'].items(),
                        key=lambda x: x[1]['total_return_pct'],
                        reverse=True
                    )
                    
                    sf.write(f"{'Holding Time (min)':<15}{'Return %':<15}{'Win Rate %':<15}{'Trades':<10}{'Sharpe':<10}\n")
                    sf.write("-" * 65 + "\n")
                    
                    for hold_time, ht_result in sorted_times:
                        is_optimal = hold_time == results.get('holding_time')
                        mark = "* " if is_optimal else "  "
                        sf.write(f"{mark + str(hold_time):<15}{ht_result['total_return_pct']:.2f}%{' ':<9}{ht_result['win_rate']*100:.1f}%{' ':<8}{ht_result['total_trades']:<10}{ht_result.get('sharpe_ratio', 0):.2f}\n")
                    
                    sf.write("\n* Optimal holding time based on combined performance metrics\n\n")
                else:
                    sf.write("No holding time analysis available\n\n")
                
                # Trade statistics
                sf.write("TRADE STATISTICS\n")
                sf.write("-" * 30 + "\n")
                if results['portfolio'] is not None and results['portfolio'].trades.count() > 0:
                    trades = results['portfolio'].trades
                    sf.write(f"Average win: {trades.win_rate()*100:.1f}%\n")
                    sf.write(f"Average trade duration: {trades.duration.mean():.1f} minutes\n")
                    sf.write(f"Average profit per trade: {trades.pnl.mean():.2f}%\n")
                    sf.write(f"Best trade: {trades.pnl.max():.2f}%\n")
                    sf.write(f"Worst trade: {trades.pnl.min():.2f}%\n")
                else:
                    sf.write("No trade statistics available\n")
            
            # Store the file path for reference in the HTML
            strategy_reports[scenario] = strategy_file_name
        
        # Generate charts for each strategy
        for scenario, results in self.backtest_results.items():
            strategy = self.strategies[scenario]
            
            if results['portfolio'] is not None:
                # Equity curve - improved
                plt.figure(figsize=(12, 6))
                if results['portfolio'] is not None:
                    # Get portfolio value series
                    equity = results['portfolio'].value()
                    
                    # Create a proper matplotlib plot instead of using vectorbt's plot method
                    plt.plot(equity.index, equity.values, linewidth=2)
                    
                    # Add initial cash as first point if missing
                    if len(equity) > 0:
                        init_cash = results['portfolio'].init_cash
                        if equity.index[0] != self.data.index[0]:
                            plt.plot([self.data.index[0], equity.index[0]], [init_cash, equity.iloc[0]], 'b--', linewidth=1, alpha=0.5)
                    
                    plt.fill_between(equity.index, equity.values, alpha=0.2)
                    plt.grid(True, alpha=0.3)
                    plt.title(f"Equity Curve: {strategy['name']}", fontsize=14)
                    plt.xlabel('Date')
                    plt.ylabel('Portfolio Value ($)')
                    
                    # Add annotations for key metrics
                    ret_pct = results['total_return_pct']
                    color = 'green' if ret_pct > 0 else 'red'
                    plt.annotate(f"Return: {ret_pct:.2f}%", 
                                xy=(0.02, 0.95), xycoords='axes fraction', 
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8),
                                color=color, fontsize=12)
                                
                    # Add win rate annotation
                    win_rate = results['win_rate']*100
                    plt.annotate(f"Win Rate: {win_rate:.1f}%", 
                                xy=(0.02, 0.89), xycoords='axes fraction', 
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8),
                                fontsize=12)
                else:
                    plt.text(0.5, 0.5, 'No equity data available', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=plt.gca().transAxes, fontsize=14)

                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, f"{scenario}_equity_curve.png"), dpi=100)
                plt.close()
                
                # Drawdowns
                plt.figure(figsize=(12, 6))
                results['portfolio'].drawdown().plot()
                plt.title(f"Drawdown: {strategy['name']}")
                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, f"{scenario}_drawdown.png"))
                plt.close()
                
                # Monthly returns heatmap if we have enough data
                if len(self.data) > 1000:
                    try:
                        plt.figure(figsize=(12, 8))
                        results['portfolio'].returns_monthwise().plot(kind='heatmap')
                        plt.title(f"Monthly Returns: {strategy['name']}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(charts_dir, f"{scenario}_monthly_returns.png"))
                        plt.close()
                    except:
                        pass
        
        # Create HTML report
        self._generate_html_report(summary_html_path, charts_dir, strategy_reports, trade_detail_reports)
        
        return {
            'txt_report': summary_report_path,
            'html_report': summary_html_path,
            'trade_details': trade_detail_reports  # Add this line
        }
    
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
    <title>ML Directional Strategies for {self.altcoin_name}</title>
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
        <h1 class="mb-4">ML Directional Strategies for {self.altcoin_name}</h1>
        
        <div class="alert alert-info">
            <p>This report shows trading strategies based on ML analysis of BTC directional movements.</p>
        </div>
        
        <h2>Strategy Performance Summary</h2>
        <div class="table-responsive mb-4">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>Type</th>
                        <th>Total Return</th>
                        <th>Period</th>
                        <th>Win Rate</th>
                        <th>Trades</th>
                        <th>Holding Time</th>
                        <th>Sharpe</th>
                        <th>Max DD</th>
                    </tr>
                </thead>
                <tbody>
                """
        
        for scenario, results in sorted_strategies:
            strategy = self.strategies[scenario]
            html += f"""
                    <tr>
                        <td>{strategy['name']}</td>
                        <td>{'Long' if strategy['is_long'] else 'Short'}</td>
                        <td class="{'up-value' if results['total_return_pct'] > 0 else 'down-value'}">{results['total_return_pct']:.2f}%</td>
                        <td>{self._get_backtest_duration()}</td>
                        <td>{results['win_rate']*100:.1f}%</td>
                        <td>{results['total_trades']}</td>
                        <td>{strategy.get('optimal_holding_time', 'N/A')} min</td>
                        <td>{results.get('sharpe_ratio', 0):.2f}</td>
                        <td>{results.get('max_drawdown', 0):.2f}%</td>
                    </tr>
                    """
                    
        html += """
                </tbody>
            </table>
        </div>
        
        <h2>Individual Strategy Details</h2>"""
        
        for scenario, results in sorted_strategies:
            strategy = self.strategies[scenario]
            html += f"""
        <div class="card strategy-card">
            <div class="card-header bg-{'success' if strategy['is_long'] else 'danger'} text-white">
                <h3>{strategy['name']}</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <h4>Strategy Parameters</h4>
                        <ul class="list-group">
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Type
                                <span class="badge bg-{'success' if strategy['is_long'] else 'danger'} rounded-pill">{'Long' if strategy['is_long'] else 'Short'}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Best Prediction Minute
                                <span class="badge bg-primary rounded-pill">{strategy['best_minute']}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                ML Accuracy
                                <span class="badge bg-info rounded-pill">{strategy['accuracy']*100:.1f}%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Stop Loss
                                <span class="badge bg-warning text-dark rounded-pill">{strategy['stop_loss_pct']}%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Take Profit
                                <span class="badge bg-success rounded-pill">{strategy['take_profit_pct']}%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Optimal Holding Time
                                <span class="badge bg-primary rounded-pill">{strategy.get('optimal_holding_time', 'N/A')} min</span>
                            </li>
                        </ul>
                        <div class="mt-3">
                            <a href="../reports/{strategy_reports[scenario]}" class="btn btn-sm btn-outline-secondary mb-2" target="_blank">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-file-text" viewBox="0 0 16 16">
                                    <path d="M5 4a.5.5 0 0 0 0 1h6a.5.5 0 0 0 0-1zm0 2a.5.5 0 0 0 0 1h3a.5.5 0 0 0 0-1z"/>
                                    <path d="M2 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1"></path>
                                </svg>
                                Download Backtest Report
                            </a>
                            """
                            
            # Add trade details download button if available
            if trade_detail_reports and scenario in trade_detail_reports:
                html += f"""
                            <a href="../reports/{trade_detail_reports[scenario]}" class="btn btn-sm btn-outline-primary" target="_blank">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-table" viewBox="0 0 16 16">
                                    <path d="M0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V2zm15 2h-4v3h4V4zm0 4h-4v3h4V8zm0 4h-4v3h3a1 1 0 0 0 1-1v-2zm-5 3v-3H6v3h4zm-5 0v-3H1v2a1 1 0 0 0 1 1h3zm-4-4h4V8H1v3zm0-4h4V4H1v3zm5-3v3h4V4H6zm4 4H6v3h4V8z"/>
                                </svg>
                                Download Trade Details (CSV)
                            </a>
                            """
                            
            html += """
                        </div>
                    </div>
                    
                    <div class="col-md-8">
                        <h4>Performance Metrics</h4>
                        <div class="d-flex flex-wrap">"""
                        
            html += f"""
                            <div class="metric-box flex-fill">
                                <div class="metric-title">Total Return</div>
                                <div class="metric-value {'up-value' if results['total_return_pct'] > 0 else 'down-value'}">{results['total_return_pct']:.2f}%</div>
                                <div class="metric-subtitle">over {self._get_backtest_duration()}</div>
                            </div>
                            <div class="metric-box flex-fill">
                                <div class="metric-title">Win Rate</div>
                                <div class="metric-value">{results['win_rate']*100:.1f}%</div>
                            </div>
                            <div class="metric-box flex-fill">
                                <div class="metric-title">Total Trades</div>
                                <div class="metric-value">{results['total_trades']}</div>
                            </div>
                            <div class="metric-box flex-fill">
                                <div class="metric-title">Profit Factor</div>
                                <div class="metric-value">{results.get('profit_factor', 0):.2f}</div>
                            </div>
                            <div class="metric-box flex-fill">
                                <div class="metric-title">Sharpe Ratio</div>
                                <div class="metric-value">{results.get('sharpe_ratio', 0):.2f}</div>
                            </div>
                            <div class="metric-box flex-fill">
                                <div class="metric-title">Max Drawdown</div>
                                <div class="metric-value">{results.get('max_drawdown', 0):.2f}%</div>
                            </div>"""
                            
            html += """
                        </div>
                        
                        <h4 class="mt-4">Top Predictive Features</h4>
                        <ul class="list-group">"""
            
            for feature, importance in self.ml_results[scenario][strategy['best_minute']]['feature_importance'][:5]:
                html += f"""
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                {feature}
                                <span class="badge bg-primary rounded-pill">{importance*100:.2f}%</span>
                            </li>"""
                
            html += """
                        </ul>
                    </div>
                </div>
                
                <div class="mt-4">
                    <h4>Equity Curve</h4>
                    <img src="../charts/{scenario}_equity_curve.png" class="img-fluid" alt="Equity Curve" onerror="if(this && this.parentElement) { this.style.display='none'; this.parentElement.innerHTML += '<div class=\\'alert alert-warning\\'>Chart not available</div>'; }">
                </div>
                
                <div class="mt-4">
                    <h4>Drawdown</h4>"""
                    
            # Check if chart exists
            drawdown_chart_path = os.path.join(charts_dir, f"{scenario}_drawdown.png")
            if os.path.exists(drawdown_chart_path):
                html += f"""
                    <img src="../charts/{scenario}_drawdown.png" class="img-fluid" alt="Drawdown" onerror="this.style.display='none'; this.parentElement.innerHTML += '<div class=\\'alert alert-warning\\'>Chart not available</div>';">"""
            else:
                html += """
                    <div class="alert alert-warning">Drawdown chart not available</div>"""
            
            html += """
                </div>
                
                <div class="mt-4">
                    <h4>Holding Time Analysis</h4>"""
                    
            # Check if we have holding time data
            if 'all_holding_time_results' in results:
                # Create a table showing different holding times and their performance
                html += """
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Holding Time (min)</th>
                                    <th>Return %</th>
                                    <th>Win Rate</th>
                                    <th>Trades</th>
                                </tr>
                            </thead>
                            <tbody>"""
                
                # Get top 5 holding times by return
                best_holding_times = sorted(
                    results['all_holding_time_results'].items(),
                    key=lambda x: x[1]['total_return_pct'],
                    reverse=True
                )[:5]
                
                for hold_time, ht_result in best_holding_times:
                    is_optimal = hold_time == results.get('holding_time')
                    html += f"""
                                <tr class="{'table-success' if is_optimal else ''}">
                                    <td>{"* " if is_optimal else ""}{hold_time}</td>
                                    <td class="{'up-value' if ht_result['total_return_pct'] > 0 else 'down-value'}">{ht_result['total_return_pct']:.2f}%</td>
                                    <td>{ht_result['win_rate']*100:.1f}%</td>
                                    <td>{ht_result['total_trades']}</td>
                                </tr>"""
                
                html += """
                            </tbody>
                        </table>
                        <small class="text-muted">* Optimal holding time based on combined performance metrics</small>
                    </div>"""
            else:
                html += """
                    <div class="alert alert-warning">
                        No holding time analysis data available
                    </div>"""
            
            html += """
                </div>
                
                <div class="mt-4">
                    <h4>Detailed Report</h4>"""
                    
            # Check if individual strategy report exists
            if strategy_reports and scenario in strategy_reports:
                report_path = strategy_reports[scenario]
                html += f"""
                    <a href="../reports/{report_path}" class="btn btn-primary">Download Detailed Report</a>"""
            else:
                html += """
                    <div class="alert alert-warning">Detailed report not available</div>"""
            
            html += """
                </div>
            </div>
        </div>"""
            
        html += """
        <div class="mt-4">
            <a href="index.html" class="btn btn-secondary">Back to Results</a>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"Generated HTML report at {output_file}")
        return output_file

    def _export_trade_details(self, portfolio, output_file):
        """Export detailed information about each individual trade to a CSV file."""
        if portfolio is None or portfolio.trades.count() == 0:
            print(f"No trades to export to {output_file}")
            return False
        
        try:
            # Get all trade data
            trades_df = portfolio.trades.records
            
            # Convert to readable format - this is where the duration gets calculated
            trades_df_readable = portfolio.trades.records_readable
            
            # Merge the two dataframes if needed
            final_df = trades_df.copy()
            
            # Add some additional calculated columns
            if len(final_df) > 0:
                # Duration calculation - check available columns first
                if 'exit_idx' in final_df.columns and 'entry_idx' in final_df.columns:
                    try:
                        # Calculate duration from indices
                        final_df['duration_minutes'] = (final_df['exit_idx'] - final_df['entry_idx']).astype(float)
                    except Exception as e:
                        print(f"  Warning: Couldn't calculate duration from indices: {e}")
                        
                # Try to copy duration from readable records if available
                if hasattr(trades_df_readable, 'duration'):
                    try:
                        final_df['duration'] = trades_df_readable['duration']
                        # Calculate minutes if duration is a timedelta
                        final_df['duration_minutes'] = trades_df_readable['duration'].dt.total_seconds() / 60
                    except Exception as e:
                        print(f"  Warning: Couldn't import duration from readable records: {e}")
                
                # Format return and PnL columns as percentages
                for col in ['return', 'pnl']:
                    if col in final_df.columns:
                        final_df[f'{col}_pct'] = final_df[col] * 100
                
                # Add entry and exit times if available
                if 'entry_idx' in final_df.columns and isinstance(portfolio.wrapper.index, pd.DatetimeIndex):
                    try:
                        index_array = portfolio.wrapper.index.values
                        final_df['entry_time'] = index_array[final_df['entry_idx'].astype(int)]
                        final_df['exit_time'] = index_array[final_df['exit_idx'].astype(int)]
                        
                        # Calculate duration in minutes directly if we have timestamps
                        final_df['duration_minutes'] = (pd.Series(final_df['exit_time']) - 
                                                        pd.Series(final_df['entry_time'])).dt.total_seconds() / 60
                    except Exception as e:
                        print(f"  Warning: Couldn't calculate timestamps: {e}")
                
                # Save to CSV
                final_df.to_csv(output_file, index=False)
                print(f"Exported {len(final_df)} trades to {output_file}")
                return True
            else:
                print(f"No trades data available to export")
                return False
        except Exception as e:
            print(f"Error exporting trade details: {e}")
            return False