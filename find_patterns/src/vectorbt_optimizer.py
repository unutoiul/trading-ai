"""
Enhanced strategy optimization using VectorBT for comprehensive backtesting.
This module provides vectorized backtesting of pattern analysis results with 
comprehensive parameter combinations including trailing stops.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType
import matplotlib.pyplot as plt
plt.style.use('default')
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import itertools
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')


class VectorBTOptimizer:
    """Enhanced strategy optimizer using VectorBT for comprehensive backtesting."""
    
    def __init__(self, data, pattern_stats, initial_cash=10000):
        """
        Initialize the VectorBT optimizer.
        
        Args:
            data: DataFrame with market data and patterns
            pattern_stats: Dictionary with pattern statistics
            initial_cash: Initial capital for backtesting
        """
        self.data = data.copy()
        self.pattern_stats = pattern_stats
        self.initial_cash = initial_cash
        self.results = []
        self.best_results = {}
        self.optimization_results = {}
        
        # Detect altcoin name
        self.altcoin_name = self._detect_altcoin_name()
        
        # Find price column for the altcoin
        self.price_column = self._find_price_column()
        
        # Get available patterns
        self.available_patterns = self._get_available_patterns()
        
        print(f"VectorBT Optimizer initialized for {self.altcoin_name.upper()}")
        print(f"Using price column: {self.price_column}")
        print(f"Available patterns: {len(self.available_patterns)}")
        
        # Debug: Show data shape and columns
        print(f"🔍 Debug - Data shape: {self.data.shape}")
        print(f"🔍 Debug - Pattern stats type: {type(self.pattern_stats)}")
        if isinstance(self.pattern_stats, dict):
            print(f"🔍 Debug - Pattern stats keys: {list(self.pattern_stats.keys())[:10]}")  # First 10
        
        # Debug: Show column types and pattern columns
        pattern_cols = [col for col in self.data.columns if col.startswith('btc_')]
        print(f"🔍 Debug - Found {len(pattern_cols)} columns starting with 'btc_'")
        if pattern_cols:
            print(f"🔍 Debug - First 5 pattern columns: {pattern_cols[:5]}")
            # Check signal counts for first few patterns
            for i, col in enumerate(pattern_cols[:3]):
                if col in self.data.columns:
                    signal_count = self.data[col].sum() if self.data[col].dtype == bool else "Not boolean"
                    print(f"🔍 Debug - {col}: {signal_count} signals")
        
        if len(self.available_patterns) == 0:
            print("⚠️  WARNING: No valid patterns found for optimization!")
            print("🔍 Troubleshooting pattern detection...")
            
            # Check all boolean columns
            bool_cols = [col for col in self.data.columns if self.data[col].dtype == bool]
            print(f"🔍 Found {len(bool_cols)} boolean columns total")
            
            # Check signal counts for all boolean columns
            if bool_cols:
                print("🔍 Signal counts for boolean columns:")
                for col in bool_cols[:10]:  # First 10
                    signal_count = self.data[col].sum()
                    print(f"   {col}: {signal_count} signals")
        else:
            print(f"✅ Pattern detection successful: {self.available_patterns[:5]}")  # Show first 5
        
    def _detect_altcoin_name(self):
        """Detect altcoin name from data columns."""
        for col in self.data.columns:
            if col.endswith('_returns') and not col.startswith('btc'):
                return col.split('_')[0]
        
        if 'altcoin_name' in self.data.columns and not self.data['altcoin_name'].isna().all():
            return self.data['altcoin_name'].iloc[0].lower()
            
        return "altcoin"
    
    def _find_price_column(self):
        """Find the appropriate price column for the altcoin."""
        possible_columns = [
            f'{self.altcoin_name}_close',
            f'close_{self.altcoin_name}',
            f'{self.altcoin_name}_price',
            f'price_{self.altcoin_name}'
        ]
        
        for col in possible_columns:
            if col in self.data.columns:
                return col
                
        # Fallback: find any column with close and altcoin name
        for col in self.data.columns:
            if 'close' in col.lower() and self.altcoin_name in col.lower():
                return col
                
        raise ValueError(f"Could not find price column for {self.altcoin_name}")
    
    def _get_available_patterns(self):
        """Get list of available pattern columns."""
        patterns = []
        
        # From pattern_stats
        if isinstance(self.pattern_stats, dict):
            patterns.extend([p for p in self.pattern_stats.keys() if p in self.data.columns])
        
        # From data columns (pattern columns typically start with 'btc_')
        pattern_cols = [col for col in self.data.columns 
                       if col.startswith('btc_') and 
                       self.data[col].dtype == bool and 
                       self.data[col].sum() > 5]  # Reduced from 10 to 5 occurrences
        
        patterns.extend(pattern_cols)
        
        # Also check for any boolean columns that might be patterns (fallback)
        if len(patterns) == 0:
            print("🔍 No 'btc_' patterns found, checking all boolean columns...")
            bool_cols = [col for col in self.data.columns 
                        if self.data[col].dtype == bool and 
                        self.data[col].sum() > 5 and
                        'pattern' in col.lower()]  # Look for anything with 'pattern' in name
            patterns.extend(bool_cols)
            
        # Even more fallback - any boolean column with reasonable signals
        if len(patterns) == 0:
            print("🔍 No pattern columns found, using any boolean columns with signals...")
            bool_cols = [col for col in self.data.columns 
                        if self.data[col].dtype == bool and 
                        self.data[col].sum() > 5]
            patterns.extend(bool_cols[:10])  # Take first 10
        
        # Remove duplicates and filter
        patterns = list(set(patterns))
        valid_patterns = []
        
        for pattern in patterns:
            if pattern in self.data.columns and self.data[pattern].sum() > 5:  # Reduced threshold
                valid_patterns.append(pattern)
        
        # Sort by signal count (descending) to prioritize patterns with more signals
        if valid_patterns:
            valid_patterns.sort(key=lambda p: self.data[p].sum(), reverse=True)
        
        return valid_patterns
    
    def define_parameter_space(self, ml_results=None):
        """
        Define comprehensive parameter space for optimization.
        
        Args:
            ml_results: Optional ML results to guide parameter selection
            
        Returns:
            Dictionary with parameter ranges
        """
        # Base parameter space with comprehensive combinations
        param_space = {
            'patterns': self.available_patterns[:15],  # Limit to top 15 patterns for performance
            'lags': list(range(1, 21)),  # 1 to 20 periods lag
            'stop_loss': [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],  # 1% to 5%
            'take_profit': [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15],  # 2% to 15%
            'trailing_stop': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],  # 0.5% to 5%
            'position_size': [0.1, 0.2, 0.3, 0.5],  # 10% to 50% of capital
            'holding_time': [5, 10, 15, 30, 60, 120, 240, 480],  # 5 min to 8 hours
        }
        
        # Enhance with ML guidance if available
        if ml_results:
            ml_guided_params = self._extract_ml_guidance(ml_results)
            param_space.update(ml_guided_params)
        
        return param_space
    
    def _extract_ml_guidance(self, ml_results):
        """Extract parameter guidance from synchronous ML results."""
        ml_guided_params = {}
        
        # Extract recommended patterns from synchronous analysis
        recommended_patterns = []
        
        for scenario_name, scenario_data in ml_results.items():
            if isinstance(scenario_data, dict):
                # Look for synchronous data (lag=0)
                if 0 in scenario_data:
                    sync_data = scenario_data[0]
                    if isinstance(sync_data, dict):
                        win_rate = sync_data.get('win_rate', 0)
                        mean_return = sync_data.get('mean_return', 0)
                        correlation = sync_data.get('correlation', 0)
                        
                        # Include patterns with good synchronous performance
                        if win_rate > 0.5 or mean_return > 0 or abs(correlation) > 0.3:
                            if scenario_name in self.data.columns:
                                recommended_patterns.append(scenario_name)
        
        if recommended_patterns:
            ml_guided_params['ml_patterns'] = list(set(recommended_patterns))
            # For synchronous analysis, we don't need lag parameters
            ml_guided_params['synchronous_analysis'] = True
            
        return ml_guided_params
    
    def _analyze_enhanced_parameter_impact(self):
        """Enhanced analysis of parameter impact on performance."""
        self.parameter_analysis = {}
        
        # Analyze each parameter
        for param_name in ['pattern', 'lag', 'stop_loss', 'take_profit', 'trailing_stop', 'position_size', 'holding_time']:
            param_impact = {}
            
            # Group results by parameter value
            param_groups = {}
            for result in self.results:
                param_value = result['params'][param_name]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result)
            
            # Calculate enhanced metrics for each parameter value
            for param_value, group in param_groups.items():
                if len(group) >= 3:  # Only include values with at least 3 results
                    returns = [r['metrics']['total_return'] for r in group]
                    win_rates = [r['metrics']['win_rate'] for r in group]
                    sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in group if not np.isnan(r['metrics']['sharpe_ratio'])]
                    trade_counts = [r['metrics']['total_trades'] for r in group]
                    drawdowns = [r['metrics']['max_drawdown'] for r in group]
                    
                    param_impact[param_value] = {
                        'avg_return': np.mean(returns),
                        'median_return': np.median(returns),
                        'std_return': np.std(returns),
                        'best_return': max(returns),
                        'worst_return': min(returns),
                        'avg_win_rate': np.mean(win_rates),
                        'avg_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                        'avg_trades': np.mean(trade_counts),
                        'avg_drawdown': np.mean(drawdowns),
                        'profit_probability': sum(1 for r in returns if r > 0) / len(returns),
                        'sample_size': len(group),
                        'count': len(group)
                    }
            
            self.parameter_analysis[param_name] = param_impact
    
    def _get_empty_summary_stats(self):
        """Return empty summary stats structure for when no results are available."""
        return {
            'total_combinations': 0,
            'profitable_strategies': 0,
            'profitability_rate': 0.0,
            'avg_return': 0.0,
            'median_return': 0.0,
            'best_return': 0.0,
            'worst_return': 0.0,
            'return_std': 0.0,
            'avg_win_rate': 0.0,
            'median_win_rate': 0.0,
            'avg_trade_count': 0.0,
            'avg_sharpe': 0.0,
            'avg_drawdown': 0.0,
            'avg_profit_factor': 0.0,
            'consistency_score': 0.0,
            'risk_adjusted_return': 0.0
        }
    
    def _calculate_enhanced_summary_stats(self):
        """Calculate enhanced summary statistics across all results."""
        if not self.results:
            return {}
        
        returns = [r['metrics']['total_return'] for r in self.results]
        win_rates = [r['metrics']['win_rate'] for r in self.results]
        trade_counts = [r['metrics']['total_trades'] for r in self.results]
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in self.results if not np.isnan(r['metrics']['sharpe_ratio'])]
        drawdowns = [r['metrics']['max_drawdown'] for r in self.results]
        profit_factors = [r['metrics']['profit_factor'] for r in self.results if not np.isnan(r['metrics']['profit_factor'])]
        
        return {
            'total_combinations': len(self.results),
            'profitable_strategies': sum(1 for r in returns if r > 0),
            'profitability_rate': sum(1 for r in returns if r > 0) / len(returns),
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'return_std': np.std(returns),
            'avg_win_rate': np.mean(win_rates),
            'median_win_rate': np.median(win_rates),
            'avg_trade_count': np.mean(trade_counts),
            'avg_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'avg_drawdown': np.mean(drawdowns),
            'avg_profit_factor': np.mean(profit_factors) if profit_factors else 0,
            'consistency_score': sum(1 for r in returns if r > 0.01) / len(returns),  # >1% return
            'risk_adjusted_return': np.mean(returns) / max(np.mean(drawdowns), 0.01)
        }
    
    def _calculate_performance_distribution(self):
        """Calculate performance distribution metrics."""
        if not self.results:
            return {}
        
        returns = [r['metrics']['total_return'] for r in self.results]
        
        return {
            'percentiles': {
                '10th': np.percentile(returns, 10),
                '25th': np.percentile(returns, 25),
                '50th': np.percentile(returns, 50),
                '75th': np.percentile(returns, 75),
                '90th': np.percentile(returns, 90),
                '95th': np.percentile(returns, 95),
                '99th': np.percentile(returns, 99)
            },
            'distribution_stats': {
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns),
                'outlier_count': sum(1 for r in returns if abs(r - np.mean(returns)) > 2 * np.std(returns))
            }
        }
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        n = len(data)
        if n < 3:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        skew = sum(((x - mean) / std) ** 3 for x in data) / n
        return skew
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data."""
        n = len(data)
        if n < 4:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        kurt = sum(((x - mean) / std) ** 4 for x in data) / n - 3
        return kurt
    
    def optimize_strategies(self, ml_results=None, momentum_results=None, results_dir=None, 
                          enable_trailing_stop=True, max_combinations=2000, n_jobs=4):
        """
        Enhanced comprehensive strategy optimization with UI-controlled parameters.
        Browser-safe implementation with memory optimization and timeout protection.
        
        Args:
            ml_results: ML analysis results for guidance
            momentum_results: Pattern analysis results
            results_dir: Directory to save results
            enable_trailing_stop: Whether to include trailing stop optimization
            max_combinations: Maximum parameter combinations to test (default: 2000 for browser safety)
            n_jobs: Number of parallel jobs (forced to 1 for stability)
            
        Returns:
            Dictionary with comprehensive optimization results
        """
        print(f"\n🚀 Starting Browser-Safe VectorBT Strategy Optimization for {self.altcoin_name.upper()}")
        print(f"🎯 Trailing Stop Testing: {'ENABLED' if enable_trailing_stop else 'DISABLED'}")
        print(f"📊 Max Combinations: {max_combinations} (Browser-safe limit)")
        print(f"🧠 Memory Optimization: ENABLED")
        print(f"⚡ Parallel Jobs: 1 (Forced for stability)")
        print(f"⏱️  Timeout Protection: ENABLED")
        
        # Enhanced parameter space with browser-safe limits
        param_space = self._define_enhanced_parameter_space(enable_trailing_stop, ml_results)
        
        # Debug: Check if we have any patterns to work with
        if len(self.available_patterns) == 0:
            print("❌ CRITICAL: No patterns available for optimization!")
            print("🔧 Attempting to find any usable patterns in the data...")
            
            # Emergency pattern detection
            all_bool_cols = [col for col in self.data.columns if self.data[col].dtype == bool]
            print(f"🔍 Found {len(all_bool_cols)} boolean columns in data")
            
            if all_bool_cols:
                # Use any boolean column with at least 3 signals
                emergency_patterns = [col for col in all_bool_cols if self.data[col].sum() >= 3]
                if emergency_patterns:
                    print(f"🚨 Using emergency patterns: {emergency_patterns[:5]}")
                    self.available_patterns = emergency_patterns[:10]  # Use up to 10
                    param_space['patterns'] = self.available_patterns
                else:
                    print("❌ No boolean columns with sufficient signals found")
                    return {'error': 'No valid patterns found for optimization'}
            else:
                print("❌ No boolean columns found in data at all")
                return {'error': 'No boolean pattern columns found in data'}
        
        # Create parameter combinations with intelligent sampling and browser limits
        combinations = self._create_intelligent_combinations(param_space, max_combinations, ml_results)
        
        print(f"🔄 Testing {len(combinations)} parameter combinations with browser-safe execution...")
        print(f"📊 Using {len(self.available_patterns)} patterns: {self.available_patterns[:3]}...")  # Show first 3
        
        # Browser-safe optimization with time tracking (no signal timeout)
        import time
        start_time = time.time()
        max_execution_time = 1800  # 30 minutes maximum
        
        try:
            # Run optimization with browser-safe progress tracking
            results = self._run_optimization_with_progress(combinations, n_jobs=1, max_time=max_execution_time, start_time=start_time)
                
        except Exception as e:
            print(f"❌ VectorBT optimization failed: {str(e)}")
            results = getattr(self, 'partial_results', [])
        
        execution_time = time.time() - start_time
        print(f"✅ Browser-safe optimization complete! Got {len(results)} valid results in {execution_time/60:.1f} minutes.")
        
        # Store and analyze results
        self.results = results
        self._analyze_comprehensive_results()
        
        # Generate detailed reports
        if results_dir:
            report_file = self.generate_enhanced_reports(results_dir, enable_trailing_stop)
            print(f"📋 Detailed reports saved to: {report_file}")
        
        # Return comprehensive optimization results
        return {
            'optimization_summary': self.optimization_results,
            'best_strategies': self.best_results,
            'parameter_insights': self.parameter_analysis,
            'trailing_stop_analysis': self.trailing_stop_analysis if enable_trailing_stop else None,
            'total_tested': len(results),
            'profitable_count': sum(1 for r in results if r['metrics']['total_return'] > 0),
            'best_return_pct': max(r['metrics']['total_return'] for r in results) * 100 if results else 0,
            'recommended_params': self._get_recommended_parameters()
        }
    
    def _define_enhanced_parameter_space(self, enable_trailing_stop=True, ml_results=None):
        """Define enhanced parameter space with more granular options."""
        
        # Enhanced stop loss percentages (more granular around common values)
        stop_loss_values = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10]
        
        # Enhanced take profit percentages (wider range with more precision)
        take_profit_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
        
        # Enhanced trailing stop percentages (if enabled)
        trailing_stop_values = []
        if enable_trailing_stop:
            trailing_stop_values = [0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
        else:
            trailing_stop_values = [0.0]  # No trailing stop
        
        # Position sizes (conservative to aggressive)
        position_sizes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        
        # Holding times (5 minutes to 24 hours)
        holding_times = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440]
        
        param_space = {
            'patterns': self.available_patterns[:20],  # Top 20 patterns
            'stop_loss': stop_loss_values,
            'take_profit': take_profit_values,
            'trailing_stop': trailing_stop_values,
            'position_size': position_sizes,
            'holding_time': holding_times
        }
        
        # Add ML-guided parameters if available
        if ml_results:
            ml_guided = self._extract_ml_guidance(ml_results)
            param_space.update(ml_guided)
        
        print(f"📊 Parameter Space Defined:")
        print(f"   • Patterns: {len(param_space['patterns'])}")
        print(f"   • Stop Loss: {len(param_space['stop_loss'])} values ({min(param_space['stop_loss'])*100:.1f}% - {max(param_space['stop_loss'])*100:.1f}%)")
        print(f"   • Take Profit: {len(param_space['take_profit'])} values ({min(param_space['take_profit'])*100:.1f}% - {max(param_space['take_profit'])*100:.1f}%)")
        print(f"   • Trailing Stop: {len(param_space['trailing_stop'])} values {'(DISABLED)' if not enable_trailing_stop else ''}")
        print(f"   • Position Sizes: {len(param_space['position_size'])} values")
        print(f"   • Holding Times: {len(param_space['holding_time'])} values")
        
        return param_space
    
    def _create_intelligent_combinations(self, param_space, max_combinations, ml_results=None):
        """Create intelligent parameter combinations with prioritization."""
        combinations = []
        
        print(f"🔧 Creating parameter combinations from space:")
        print(f"   • Patterns: {len(param_space.get('patterns', []))}")
        print(f"   • Stop Loss values: {len(param_space.get('stop_loss', []))}")
        print(f"   • Take Profit values: {len(param_space.get('take_profit', []))}")
        print(f"   • Trailing Stop values: {len(param_space.get('trailing_stop', []))}")
        
        if not param_space.get('patterns'):
            print("❌ No patterns in parameter space!")
            return []
        
        # Priority 1: ML-guided combinations (if available)
        if ml_results and 'ml_patterns' in param_space:
            ml_combinations = self._create_ml_guided_combinations(param_space, max_combinations // 3)
            combinations.extend(ml_combinations)
            print(f"🎯 Added {len(ml_combinations)} ML-guided combinations")
        
        # Priority 2: High-frequency pattern combinations
        high_freq_combinations = self._create_high_frequency_combinations(param_space, max_combinations // 3)
        combinations.extend(high_freq_combinations)
        print(f"🔥 Added {len(high_freq_combinations)} high-frequency pattern combinations")
        
        # Priority 3: Random sampling from remaining space
        remaining_slots = max_combinations - len(combinations)
        if remaining_slots > 0:
            random_combinations = self._create_random_combinations(param_space, remaining_slots)
            combinations.extend(random_combinations)
            print(f"🎲 Added {len(random_combinations)} random combinations")
        
        # Remove duplicates
        unique_combinations = []
        seen = set()
        for combo in combinations:
            combo_key = tuple(sorted(combo.items()))
            if combo_key not in seen:
                unique_combinations.append(combo)
                seen.add(combo_key)
        
        print(f"🔧 Final combinations after deduplication: {len(unique_combinations)}")
        
        # Debug: Show first few combinations
        if unique_combinations:
            print("🔍 Sample combinations:")
            for i, combo in enumerate(unique_combinations[:3]):
                print(f"   {i+1}. Pattern: {combo['pattern']}, SL: {combo['stop_loss']*100:.1f}%, TP: {combo['take_profit']*100:.1f}%")
        
        return unique_combinations
    
    def _create_ml_guided_combinations(self, param_space, max_count):
        """Create combinations guided by synchronous ML results."""
        combinations = []
        ml_patterns = param_space.get('ml_patterns', [])
        
        if not ml_patterns:
            return combinations
        
        # Focus on top ML patterns with optimized parameters (no lag needed for synchronous)
        import itertools
        import random
        
        ml_combos = list(itertools.product(
            ml_patterns,
            param_space['stop_loss'][:8],  # Top 8 stop loss values
            param_space['take_profit'][:8],  # Top 8 take profit values
            param_space['trailing_stop'][:6] if param_space['trailing_stop'] else [0.0],  # Top 6 trailing stop values
            param_space['position_size'][:4],  # Top 4 position sizes
            param_space['holding_time'][:6]  # Top 6 holding times
        ))
        
        # Shuffle and take what we need
        random.shuffle(ml_combos)
        
        for combo in ml_combos[:max_count]:
            combinations.append({
                'pattern': combo[0],
                'lag': 0,  # Always 0 for synchronous analysis
                'stop_loss': combo[1],
                'take_profit': combo[2],
                'trailing_stop': combo[4],
                'position_size': combo[5],
                'holding_time': combo[6],
                'source': 'ml_guided'
            })
        
        return combinations
    
    def _create_high_frequency_combinations(self, param_space, max_count):
        """Create combinations focusing on high-frequency patterns."""
        combinations = []
        
        # Get patterns with high occurrence rates
        high_freq_patterns = []
        available_patterns = param_space.get('patterns', [])
        
        if not available_patterns:
            print("⚠️  No patterns available for high-frequency combinations")
            return combinations
        
        for pattern in available_patterns:
            if pattern in self.data.columns:
                occurrence_rate = self.data[pattern].sum() / len(self.data)
                if occurrence_rate > 0.005:  # Reduced from 0.01 to 0.005 (0.5% occurrence)
                    high_freq_patterns.append(pattern)
        
        # If no high-frequency patterns, use all available patterns
        if not high_freq_patterns:
            print(f"⚠️  No high-frequency patterns found, using all {len(available_patterns)} available patterns")
            high_freq_patterns = available_patterns
        else:
            print(f"✅ Found {len(high_freq_patterns)} high-frequency patterns")
        
        import itertools
        import random
        
        # Create combinations with focus on performance (synchronous analysis)
        stop_loss_vals = param_space.get('stop_loss', [0.02])[:10]  # Top 10 or default
        take_profit_vals = param_space.get('take_profit', [0.04])[:10]  # Top 10 or default  
        trailing_stop_vals = param_space.get('trailing_stop', [0.0])[:8] if param_space.get('trailing_stop') else [0.0]
        position_size_vals = param_space.get('position_size', [0.2])[:3]  # Top 3 or default
        holding_time_vals = param_space.get('holding_time', [60])[:8]  # Top 8 or default
        
        hf_combos = list(itertools.product(
            high_freq_patterns,
            stop_loss_vals,
            take_profit_vals,
            trailing_stop_vals,
            position_size_vals,
            holding_time_vals
        ))
        
        random.shuffle(hf_combos)
        
        for combo in hf_combos[:max_count]:
            combinations.append({
                'pattern': combo[0],
                'lag': 0,  # Synchronous analysis
                'stop_loss': combo[1],
                'take_profit': combo[2],
                'trailing_stop': combo[3],
                'position_size': combo[4],
                'holding_time': combo[5],
                'source': 'high_frequency'
            })
        
        print(f"🔥 Created {len(combinations)} high-frequency combinations")
        return combinations
    
    def _create_random_combinations(self, param_space, max_count):
        """Create random combinations from full parameter space."""
        combinations = []
        import random
        
        available_patterns = param_space.get('patterns', [])
        if not available_patterns:
            print("⚠️  No patterns available for random combinations")
            return combinations
        
        # Ensure we have defaults for all parameters
        stop_loss_vals = param_space.get('stop_loss', [0.015, 0.02, 0.025])
        take_profit_vals = param_space.get('take_profit', [0.03, 0.04, 0.05])
        trailing_stop_vals = param_space.get('trailing_stop', [0.0, 0.01, 0.02])
        position_size_vals = param_space.get('position_size', [0.1, 0.2, 0.3])
        holding_time_vals = param_space.get('holding_time', [30, 60, 120])
        
        for _ in range(max_count):
            combinations.append({
                'pattern': random.choice(available_patterns),
                'lag': 0,  # Synchronous analysis only
                'stop_loss': random.choice(stop_loss_vals),
                'take_profit': random.choice(take_profit_vals),
                'trailing_stop': random.choice(trailing_stop_vals),
                'position_size': random.choice(position_size_vals),
                'holding_time': random.choice(holding_time_vals),
                'source': 'random'
            })
        
        print(f"🎲 Created {len(combinations)} random combinations")
        return combinations
    
    def _run_optimization_with_progress(self, combinations, n_jobs, max_time=1800, start_time=None):
        """Run optimization with enhanced progress tracking and browser-friendly execution."""
        results = []
        
        if start_time is None:
            start_time = time.time()
        
        # Limit combinations to prevent browser issues
        max_safe_combinations = 2000  # Reduced for browser stability
        if len(combinations) > max_safe_combinations:
            print(f"⚠️ Limiting combinations from {len(combinations)} to {max_safe_combinations} for browser stability")
            combinations = combinations[:max_safe_combinations]
        
        # Force single-threaded to avoid VectorBT multiprocessing caching issues
        print(f"🔄 Running browser-safe optimization (single-threaded)...")
        print(f"📊 Total combinations to test: {len(combinations)}")
        print(f"🧠 Memory optimization: Lightweight result storage enabled")
        print(f"⏱️  Time limit: {max_time/60:.0f} minutes")
        print(f"┌─────────────────────────────────────────────────────────────────────────┐")
        
        # Process in smaller batches to reduce memory pressure
        batch_size = 100
        total_batches = (len(combinations) + batch_size - 1) // batch_size
        
        # Store partial results for timeout scenarios
        self.partial_results = []
        
        for batch_idx in range(total_batches):
            # Check time limit before each batch
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                print(f"│ ⏱️  Time limit reached ({elapsed_time/60:.1f} minutes) - stopping optimization")
                print(f"│ 📊 Returning partial results: {len(results)} combinations completed")
                break
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(combinations))
            batch = combinations[start_idx:end_idx]
            
            remaining_time = max_time - elapsed_time
            print(f"│ 🔄 Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} combinations) | {remaining_time/60:.1f}min left")
            
            batch_results = []
            for i, combo in enumerate(batch):
                # Check time limit more frequently
                if (time.time() - start_time) > max_time:
                    print(f"│   ⏱️  Time limit reached during batch processing")
                    break
                    
                global_idx = start_idx + i
                progress = (global_idx / len(combinations)) * 100
                
                # Show progress more frequently for user feedback
                if i % 25 == 0 or i == len(batch) - 1:
                    eta_mins = ((len(combinations) - global_idx) * 0.008) if global_idx > 0 else 0  # More accurate estimate
                    print(f"│   Progress: {progress:>6.1f}% ({global_idx:>4}/{len(combinations):<4}) | ETA: ~{eta_mins:>4.1f}min │")
                
                result = self._backtest_combination_lightweight(combo)
                if result:
                    batch_results.append(result)
                    
                    # Show details of good results but less frequently
                    if result['metrics']['total_return_pct'] > 10:  # Only show very good results
                        metrics = result['metrics']
                        params = result['params']
                        print(f"│   🌟 Excellent result: {metrics['total_return_pct']:>6.2f}% ret, {metrics['win_rate']*100:>5.1f}% wr │")
            
            results.extend(batch_results)
            self.partial_results = results.copy()  # Store for timeout scenarios
            
            # Memory cleanup after each batch
            import gc
            gc.collect()
            
            print(f"│   ✅ Batch {batch_idx + 1} complete: {len(batch_results)} valid results │")
        
        print(f"└─────────────────────────────────────────────────────────────────────────┘")
        print(f"✅ Browser-safe optimization complete! Found {len(results)} valid results.")
        print(f"💾 Memory optimized: Lightweight storage used to prevent browser issues")
        
        return results
    
    def _analyze_comprehensive_results(self):
        """Enhanced analysis of optimization results."""
        if not self.results:
            print("❌ No valid results to analyze!")
            # Initialize empty structures for the case of no results
            self.best_results = {
                'best_return': None,
                'best_sharpe': None,
                'best_risk_adjusted': None,
                'best_win_rate': None,
                'best_profit_factor': None,
                'top_20_return': [],
                'top_20_sharpe': [],
                'top_20_risk_adjusted': [],
                'all_results': []
            }
            self.parameter_analysis = {}
            self.trailing_stop_analysis = {}
            self.optimization_results = {
                'best_strategies': self.best_results,
                'parameter_analysis': self.parameter_analysis,
                'trailing_stop_analysis': self.trailing_stop_analysis,
                'summary_stats': self._get_empty_summary_stats(),
                'total_combinations_tested': 0,
                'performance_distribution': {},
                'optimal_ranges': {}
            }
            return
        
        # Enhanced sorting and analysis
        by_return = sorted(self.results, key=lambda x: x['metrics']['total_return'], reverse=True)
        by_sharpe = sorted(self.results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
        by_risk_adjusted = sorted(self.results, key=lambda x: x['metrics']['risk_adjusted_return'], reverse=True)
        by_win_rate = sorted(self.results, key=lambda x: x['metrics']['win_rate'], reverse=True)
        by_profit_factor = sorted(self.results, key=lambda x: x['metrics']['profit_factor'], reverse=True)
        
        # Store comprehensive best results
        self.best_results = {
            'best_return': by_return[0] if by_return else None,
            'best_sharpe': by_sharpe[0] if by_sharpe else None,
            'best_risk_adjusted': by_risk_adjusted[0] if by_risk_adjusted else None,
            'best_win_rate': by_win_rate[0] if by_win_rate else None,
            'best_profit_factor': by_profit_factor[0] if by_profit_factor else None,
            'top_20_return': by_return[:20],
            'top_20_sharpe': by_sharpe[:20],
            'top_20_risk_adjusted': by_risk_adjusted[:20],
            'all_results': self.results
        }
        
        # Enhanced parameter analysis
        self._analyze_enhanced_parameter_impact()
        
        # Trailing stop specific analysis
        self._analyze_trailing_stop_impact()
        
        # Store comprehensive results
        self.optimization_results = {
            'best_strategies': self.best_results,
            'parameter_analysis': self.parameter_analysis,
            'trailing_stop_analysis': self.trailing_stop_analysis,
            'summary_stats': self._calculate_enhanced_summary_stats(),
            'total_combinations_tested': len(self.results),
            'performance_distribution': self._calculate_performance_distribution(),
            'optimal_ranges': self._find_optimal_parameter_ranges()
        }
        
        # Print enhanced summary
        self._print_enhanced_optimization_summary()
    
    def _analyze_trailing_stop_impact(self):
        """Detailed analysis of trailing stop impact."""
        ts_analysis = {}
        
        # Group results by trailing stop value
        ts_groups = {}
        for result in self.results:
            ts_value = result['params']['trailing_stop']
            if ts_value not in ts_groups:
                ts_groups[ts_value] = []
            ts_groups[ts_value].append(result)
        
        # Analyze each trailing stop value
        for ts_value, group in ts_groups.items():
            if len(group) >= 10:  # Only analyze values with sufficient data
                returns = [r['metrics']['total_return'] for r in group]
                win_rates = [r['metrics']['win_rate'] for r in group]
                sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in group]
                drawdowns = [r['metrics']['max_drawdown'] for r in group]
                trade_counts = [r['metrics']['total_trades'] for r in group]
                
                ts_analysis[ts_value] = {
                    'avg_return': np.mean(returns),
                    'median_return': np.median(returns),
                    'std_return': np.std(returns),
                    'best_return': max(returns),
                    'worst_return': min(returns),
                    'avg_win_rate': np.mean(win_rates),
                    'avg_sharpe': np.mean([s for s in sharpe_ratios if not np.isnan(s)]),
                    'avg_drawdown': np.mean(drawdowns),
                    'avg_trades': np.mean(trade_counts),
                    'profit_probability': sum(1 for r in returns if r > 0) / len(returns),
                    'sample_size': len(group)
                }
        
        self.trailing_stop_analysis = ts_analysis
        
        # Find optimal trailing stop range
        if ts_analysis:
            optimal_ts = max(ts_analysis.items(), key=lambda x: x[1]['avg_return'])
            print(f"🎯 Optimal Trailing Stop: {optimal_ts[0]*100:.1f}% (Avg Return: {optimal_ts[1]['avg_return']*100:.2f}%)")
    
    def _find_optimal_parameter_ranges(self):
        """Find optimal ranges for each parameter."""
        optimal_ranges = {}
        
        # Analyze top 10% of strategies
        top_strategies = sorted(self.results, key=lambda x: x['metrics']['total_return'], reverse=True)
        top_10_percent = top_strategies[:max(1, len(top_strategies) // 10)]
        
        if not top_10_percent:
            return optimal_ranges
        
        # Extract parameter values from top strategies
        for param in ['stop_loss', 'take_profit', 'trailing_stop', 'lag', 'position_size', 'holding_time']:
            values = [s['params'][param] for s in top_10_percent]
            
            if values:
                optimal_ranges[param] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'most_common': max(set(values), key=values.count)
                }
        
        return optimal_ranges
    
    def _get_recommended_parameters(self):
        """Get recommended parameter values based on optimization results."""
        if not self.optimization_results:
            return {}
        
        optimal_ranges = self.optimization_results.get('optimal_ranges', {})
        best_strategy = self.best_results.get('best_return')
        
        recommendations = {}
        
        if best_strategy:
            recommendations['best_single_strategy'] = best_strategy['params']
        
        if optimal_ranges:
            recommendations['optimal_ranges'] = {
                param: {
                    'recommended': ranges['median'],
                    'range': f"{ranges['min']:.3f} - {ranges['max']:.3f}",
                    'mean': ranges['mean']
                } for param, ranges in optimal_ranges.items()
            }
        
        return recommendations
    
    def _print_enhanced_optimization_summary(self):
        """Print enhanced optimization summary."""
        print("\n" + "="*80)
        print("🚀 ENHANCED VECTORBT OPTIMIZATION SUMMARY")
        print("="*80)
        
        summary = self.optimization_results['summary_stats']
        print(f"📊 Total combinations tested: {summary['total_combinations']}")
        print(f"💰 Profitable strategies: {summary['profitable_strategies']} ({summary['profitability_rate']*100:.1f}%)")
        print(f"📈 Best return: {summary['best_return']*100:.2f}%")
        print(f"📊 Average return: {summary['avg_return']*100:.2f}%")
        print(f"🎯 Average win rate: {summary['avg_win_rate']*100:.1f}%")
        print(f"⚡ Average Sharpe ratio: {summary['avg_sharpe']:.3f}")
        
        # Print top strategy details
        if self.best_results['best_return']:
            print("\n🏆 BEST STRATEGY BY RETURN:")
            best = self.best_results['best_return']
            params = best['params']
            metrics = best['metrics']
            
            print(f"   🎯 Pattern: {params['pattern']}")
            print(f"   ⏱️  Lag: {params['lag']} periods")
            print(f"   🛑 Stop Loss: {params['stop_loss']*100:.2f}%")
            print(f"   🎯 Take Profit: {params['take_profit']*100:.2f}%")
            print(f"   📈 Trailing Stop: {params['trailing_stop']*100:.2f}%")
            print(f"   💼 Position Size: {params['position_size']*100:.0f}%")
            print(f"   ⏰ Holding Time: {params['holding_time']} periods")
            print(f"   📊 Performance:")
            print(f"      • Return: {metrics['total_return']*100:.2f}%")
            print(f"      • Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"      • Sharpe: {metrics['sharpe_ratio']:.3f}")
            print(f"      • Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"      • Total Trades: {metrics['total_trades']}")
            print(f"      • Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Print trailing stop analysis if available
        if hasattr(self, 'trailing_stop_analysis') and self.trailing_stop_analysis:
            print("\n📈 TRAILING STOP ANALYSIS:")
            sorted_ts = sorted(self.trailing_stop_analysis.items(), 
                             key=lambda x: x[1]['avg_return'], reverse=True)
            
            print("   Top 5 Trailing Stop Values by Average Return:")
            for i, (ts, data) in enumerate(sorted_ts[:5]):
                print(f"   {i+1}. {ts*100:.1f}%: Avg Return {data['avg_return']*100:.2f}%, "
                      f"Win Rate {data['avg_win_rate']*100:.1f}%, "
                      f"Samples {data['sample_size']}")
        
        print("="*80)
    
    def _create_parameter_combinations(self, param_space, max_combinations):
        """Create parameter combinations, prioritizing ML-guided patterns for synchronous analysis."""
        combinations = []
        
        # Prioritize ML-guided combinations if available
        ml_patterns = param_space.get('ml_patterns', [])
        
        # Create ML-prioritized combinations first (no lags needed for synchronous)
        if ml_patterns:
            ml_combinations = list(itertools.product(
                ml_patterns,
                param_space['stop_loss'][:4],  # Top 4 stop loss values
                param_space['take_profit'][:4],  # Top 4 take profit values
                param_space['trailing_stop'][:4],  # Top 4 trailing stop values
                param_space['position_size'][:2],  # Top 2 position sizes
                param_space['holding_time'][:4]  # Top 4 holding times
            ))
            
            # Convert to dictionaries
            for combo in ml_combinations[:max_combinations//2]:  # Use half for ML-guided
                combinations.append({
                    'pattern': combo[0],
                    'lag': 0,  # Synchronous analysis
                    'stop_loss': combo[1],
                    'take_profit': combo[2],
                    'trailing_stop': combo[3],
                    'position_size': combo[4],
                    'holding_time': combo[5],
                    'source': 'ml_guided'
                })
        
        # Fill remaining slots with regular combinations
        remaining_slots = max_combinations - len(combinations)
        
        if remaining_slots > 0:
            regular_combinations = list(itertools.product(
                param_space['patterns'][:10],  # Top 10 patterns
                param_space['stop_loss'][:3],  # Top 3 stop loss values
                param_space['take_profit'][:3],  # Top 3 take profit values
                param_space['trailing_stop'][:3],  # Top 3 trailing stop values
                param_space['position_size'][:2],  # Top 2 position sizes
                param_space['holding_time'][:3]  # Top 3 holding times
            ))
            
            # Shuffle and take what we need
            np.random.shuffle(regular_combinations)
            
            for combo in regular_combinations[:remaining_slots]:
                combinations.append({
                    'pattern': combo[0],
                    'lag': 0,  # Synchronous analysis
                    'stop_loss': combo[1],
                    'take_profit': combo[2],
                    'trailing_stop': combo[3],
                    'position_size': combo[4],
                    'holding_time': combo[5],
                    'source': 'regular'
                })
        
        print(f"Created {len(combinations)} combinations for synchronous analysis")
        if ml_patterns:
            ml_count = sum(1 for c in combinations if c['source'] == 'ml_guided')
            print(f"  - {ml_count} ML-guided combinations")
            print(f"  - {len(combinations) - ml_count} regular combinations")
        
        return combinations
    
    def _backtest_combination_lightweight(self, params):
        """
        Lightweight backtest function for browser-safe execution.
        Stores only essential metrics, not full portfolio objects.
        
        Args:
            params: Dictionary with parameters
            
        Returns:
            Dictionary with essential results or None if failed
        """
        try:
            pattern = params['pattern']
            lag = params['lag']
            
            # Check if pattern exists and has enough signals
            if pattern not in self.data.columns:
                print(f"⚠️  Pattern '{pattern}' not found in data columns")
                return None
                
            pattern_signals = self.data[pattern]
            signal_count = pattern_signals.sum()
            if signal_count < 3:  # Reduced minimum from 5 to 3 for more results
                print(f"⚠️  Pattern '{pattern}' has only {signal_count} signals (minimum 3 required)")
                return None
            
            # Create entry signals with lag
            entries = pattern_signals.shift(lag).fillna(False)
            
            # Get price data
            if self.price_column not in self.data.columns:
                print(f"⚠️  Price column '{self.price_column}' not found in data")
                available_cols = [col for col in self.data.columns if 'close' in col.lower()]
                print(f"Available price columns: {available_cols[:5]}")  # Show first 5
                return None
                
            price = self.data[self.price_column].astype(float)
            
            # Handle any missing or infinite values
            price = price.replace([np.inf, -np.inf], np.nan).dropna()
            entries = entries.reindex(price.index, fill_value=False)
            
            if len(price) == 0:
                print(f"⚠️  No valid price data after cleaning")
                return None
                
            if entries.sum() == 0:
                print(f"⚠️  No entry signals after applying lag {lag}")
                return None
            
            # Debug info for first few combinations
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
                
            if self._debug_count <= 3:  # Show debug info for first 3 combinations
                print(f"🔍 Debug {self._debug_count}: {pattern} - {entries.sum()} signals, {len(price)} price points")
            
            # Run VectorBT backtest with minimal memory usage
            pf = vbt.Portfolio.from_signals(
                price,
                entries,
                size=params['position_size'],
                size_type=SizeType.Percent,
                init_cash=self.initial_cash,
                fees=0.001,  # 0.1% trading fee
                sl_stop=params['stop_loss'],
                tp_stop=params['take_profit'],
                sl_trail=params['trailing_stop'],
                freq='1min'  # Assume 1-minute data
            )
            
            # Calculate essential metrics only
            total_return = pf.total_return()
            win_rate = pf.trades.win_rate() if pf.trades.count() > 0 else 0
            total_trades = pf.trades.count()
            max_drawdown = pf.max_drawdown()
            sharpe_ratio = pf.sharpe_ratio()
            profit_factor = pf.trades.profit_factor() if pf.trades.count() > 0 else 0
            
            # Calculate additional essential metrics
            avg_trade_duration = pf.trades.duration.mean() if total_trades > 0 else pd.Timedelta(0)
            avg_trade_return = pf.trades.returns.mean() if total_trades > 0 else 0
            
            # Safe duration calculation
            if total_trades > 0 and avg_trade_duration is not None:
                if hasattr(avg_trade_duration, 'total_seconds'):
                    duration_minutes = avg_trade_duration.total_seconds() / 60
                else:
                    try:
                        duration_minutes = float(avg_trade_duration) / (60 * 1_000_000_000)
                    except:
                        duration_minutes = 0
            else:
                duration_minutes = 0
            
            # Create lightweight result (no portfolio object or equity curve)
            result = {
                'params': params,
                'metrics': {
                    'total_return': total_return,
                    'total_return_pct': total_return * 100,
                    'win_rate': win_rate,
                    'total_trades': total_trades,
                    'max_drawdown': max_drawdown,
                    'max_drawdown_pct': max_drawdown * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'profit_factor': profit_factor,
                    'avg_trade_duration_minutes': duration_minutes,
                    'avg_trade_return_pct': avg_trade_return * 100,
                    'final_value': pf.final_value(),
                    'risk_adjusted_return': total_return / max(max_drawdown, 0.01) if max_drawdown > 0 else total_return
                }
                # Note: No 'portfolio', 'equity_curve', or 'trades' to save memory
            }
            
            # Debug info for first few successful results
            if self._debug_count <= 5 and total_trades > 0:
                print(f"✅ Success {self._debug_count}: {pattern} - {total_return*100:.2f}% return, {total_trades} trades")
            
            return result
            
        except Exception as e:
            # Show some errors for debugging but not all
            if hasattr(self, '_error_count'):
                self._error_count += 1
            else:
                self._error_count = 1
                
            if self._error_count <= 3:  # Show first 3 errors for debugging
                print(f"❌ Error {self._error_count} backtesting {params.get('pattern', 'unknown')}: {str(e)}")
            return None
    
    def _backtest_combination(self, params):
        """
        Backtest a single parameter combination using VectorBT.
        
        Args:
            params: Dictionary with parameters
            
        Returns:
            Dictionary with backtest results or None if failed
        """
        try:
            pattern = params['pattern']
            lag = params['lag']
            
            # Check if pattern exists and has enough signals
            if pattern not in self.data.columns:
                print(f"⚠️ Pattern '{pattern}' not found in data columns")
                return None
                
            pattern_signals = self.data[pattern]
            signal_count = pattern_signals.sum()
            if signal_count < 5:  # Need at least 5 signals
                print(f"⚠️ Pattern '{pattern}' has only {signal_count} signals (minimum 5 required)")
                return None
            
            # Create entry signals with lag
            entries = pattern_signals.shift(lag).fillna(False)
            
            # Get price data
            if self.price_column not in self.data.columns:
                print(f"⚠️ Price column '{self.price_column}' not found in data")
                available_cols = [col for col in self.data.columns if 'close' in col.lower()]
                print(f"Available price columns: {available_cols}")
                return None
                
            price = self.data[self.price_column].astype(float)
            
            # Handle any missing or infinite values
            price = price.replace([np.inf, -np.inf], np.nan).dropna()
            entries = entries.reindex(price.index, fill_value=False)
            
            if len(price) == 0:
                print(f"⚠️ No valid price data after cleaning")
                return None
                
            if entries.sum() == 0:
                print(f"⚠️ No entry signals after applying lag {lag}")
                return None
            
            print(f"✅ Testing {pattern}: {entries.sum()} signals, {len(price)} price points")
            
            # Run VectorBT backtest
            pf = vbt.Portfolio.from_signals(
                price,
                entries,
                size=params['position_size'],
                size_type=SizeType.Percent,
                init_cash=self.initial_cash,
                fees=0.001,  # 0.1% trading fee
                sl_stop=params['stop_loss'],
                tp_stop=params['take_profit'],
                sl_trail=params['trailing_stop'],
                freq='1min'  # Assume 1-minute data
            )
            
            # Calculate metrics
            total_return = pf.total_return()
            win_rate = pf.trades.win_rate() if pf.trades.count() > 0 else 0
            total_trades = pf.trades.count()
            max_drawdown = pf.max_drawdown()
            sharpe_ratio = pf.sharpe_ratio()
            profit_factor = pf.trades.profit_factor() if pf.trades.count() > 0 else 0
            
            # Calculate additional metrics
            avg_trade_duration = pf.trades.duration.mean() if total_trades > 0 else pd.Timedelta(0)
            avg_trade_return = pf.trades.returns.mean() if total_trades > 0 else 0
            
            # Safe duration calculation - handle both Timedelta and numpy types
            if total_trades > 0 and avg_trade_duration is not None:
                if hasattr(avg_trade_duration, 'total_seconds'):
                    # It's a pandas Timedelta
                    duration_minutes = avg_trade_duration.total_seconds() / 60
                else:
                    # It's likely a numpy scalar (in nanoseconds), convert to minutes
                    try:
                        # Convert nanoseconds to minutes (assuming it's in nanoseconds)
                        duration_minutes = float(avg_trade_duration) / (60 * 1_000_000_000)
                    except:
                        duration_minutes = 0
            else:
                duration_minutes = 0
            
            result = {
                'params': params,
                'metrics': {
                    'total_return': total_return,
                    'total_return_pct': total_return * 100,
                    'win_rate': win_rate,
                    'total_trades': total_trades,
                    'max_drawdown': max_drawdown,
                    'max_drawdown_pct': max_drawdown * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'profit_factor': profit_factor,
                    'avg_trade_duration_minutes': duration_minutes,
                    'avg_trade_return_pct': avg_trade_return * 100,
                    'final_value': pf.final_value(),
                    'risk_adjusted_return': total_return / max(max_drawdown, 0.01) if max_drawdown > 0 else total_return
                },
                'portfolio': pf,
                'equity_curve': pf.value(),
                'trades': pf.trades.records_readable if total_trades > 0 else pd.DataFrame()
            }
            
            print(f"✅ {pattern}: Return {total_return*100:.2f}%, Trades: {total_trades}, Win Rate: {win_rate*100:.1f}%")
            return result
            
        except Exception as e:
            print(f"❌ Error backtesting {params.get('pattern', 'unknown')}: {str(e)}")
            return None
        """
        Lightweight backtest function for browser-safe execution.
        Stores only essential metrics, not full portfolio objects.
        
        Args:
            params: Dictionary with parameters
            
        Returns:
            Dictionary with essential results or None if failed
        """
        try:
            pattern = params['pattern']
            lag = params['lag']
            
            # Check if pattern exists and has enough signals
            if pattern not in self.data.columns:
                return None
                
            pattern_signals = self.data[pattern]
            signal_count = pattern_signals.sum()
            if signal_count < 5:  # Need at least 5 signals
                return None
            
            # Create entry signals with lag
            entries = pattern_signals.shift(lag).fillna(False)
            
            # Get price data
            if self.price_column not in self.data.columns:
                return None
                
            price = self.data[self.price_column].astype(float)
            
            # Handle any missing or infinite values
            price = price.replace([np.inf, -np.inf], np.nan).dropna()
            entries = entries.reindex(price.index, fill_value=False)
            
            if len(price) == 0 or entries.sum() == 0:
                return None
            
            # Run VectorBT backtest with minimal memory usage
            pf = vbt.Portfolio.from_signals(
                price,
                entries,
                size=params['position_size'],
                size_type=SizeType.Percent,
                init_cash=self.initial_cash,
                fees=0.001,  # 0.1% trading fee
                sl_stop=params['stop_loss'],
                tp_stop=params['take_profit'],
                sl_trail=params['trailing_stop'],
                freq='1min'  # Assume 1-minute data
            )
            
            # Calculate essential metrics only
            total_return = pf.total_return()
            win_rate = pf.trades.win_rate() if pf.trades.count() > 0 else 0
            total_trades = pf.trades.count()
            max_drawdown = pf.max_drawdown()
            sharpe_ratio = pf.sharpe_ratio()
            profit_factor = pf.trades.profit_factor() if pf.trades.count() > 0 else 0
            
            # Calculate additional essential metrics
            avg_trade_duration = pf.trades.duration.mean() if total_trades > 0 else pd.Timedelta(0)
            avg_trade_return = pf.trades.returns.mean() if total_trades > 0 else 0
            
            # Safe duration calculation
            if total_trades > 0 and avg_trade_duration is not None:
                if hasattr(avg_trade_duration, 'total_seconds'):
                    duration_minutes = avg_trade_duration.total_seconds() / 60
                else:
                    try:
                        duration_minutes = float(avg_trade_duration) / (60 * 1_000_000_000)
                    except:
                        duration_minutes = 0
            else:
                duration_minutes = 0
            
            # Create lightweight result (no portfolio object or equity curve)
            result = {
                'params': params,
                'metrics': {
                    'total_return': total_return,
                    'total_return_pct': total_return * 100,
                    'win_rate': win_rate,
                    'total_trades': total_trades,
                    'max_drawdown': max_drawdown,
                    'max_drawdown_pct': max_drawdown * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'profit_factor': profit_factor,
                    'avg_trade_duration_minutes': duration_minutes,
                    'avg_trade_return_pct': avg_trade_return * 100,
                    'final_value': pf.final_value(),
                    'risk_adjusted_return': total_return / max(max_drawdown, 0.01) if max_drawdown > 0 else total_return
                }
                # Note: No 'portfolio', 'equity_curve', or 'trades' to save memory
            }
            
            return result
            
        except Exception as e:
            # Suppress individual errors to avoid spam
            return None
    
    def _analyze_results(self):
        """Analyze optimization results and find best strategies."""
        if not self.results:
            print("No valid results to analyze!")
            return
        
        # Sort by different criteria
        by_return = sorted(self.results, key=lambda x: x['metrics']['total_return'], reverse=True)
        by_sharpe = sorted(self.results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
        by_risk_adjusted = sorted(self.results, key=lambda x: x['metrics']['risk_adjusted_return'], reverse=True)
        by_win_rate = sorted(self.results, key=lambda x: x['metrics']['win_rate'], reverse=True)
        
        # Store best results
        self.best_results = {
            'best_return': by_return[0] if by_return else None,
            'best_sharpe': by_sharpe[0] if by_sharpe else None,
            'best_risk_adjusted': by_risk_adjusted[0] if by_risk_adjusted else None,
            'best_win_rate': by_win_rate[0] if by_win_rate else None,
            'top_10_return': by_return[:10],
            'top_10_sharpe': by_sharpe[:10],
            'all_results': self.results
        }
        
        # Analyze by parameter values
        self._analyze_parameter_impact()
        
        # Store comprehensive results
        self.optimization_results = {
            'best_strategies': self.best_results,
            'parameter_analysis': self.parameter_analysis,
            'summary_stats': self._calculate_summary_stats(),
            'total_combinations_tested': len(self.results)
        }
        
        # Print summary
        self._print_optimization_summary()
    
    def _analyze_parameter_impact(self):
        """Analyze the impact of different parameter values on performance."""
        parameter_analysis = {}
        
        # Analyze each parameter
        for param_name in ['pattern', 'lag', 'stop_loss', 'take_profit', 'trailing_stop', 'position_size', 'holding_time']:
            param_impact = {}
            
            # Group results by parameter value
            param_groups = {}
            for result in self.results:
                param_value = result['params'][param_name]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result)
            
            # Calculate average metrics for each parameter value
            for param_value, group in param_groups.items():
                if len(group) >= 3:  # Only include values with at least 3 results
                    avg_return = np.mean([r['metrics']['total_return'] for r in group])
                    avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in group])
                    avg_win_rate = np.mean([r['metrics']['win_rate'] for r in group])
                    avg_trades = np.mean([r['metrics']['total_trades'] for r in group])
                    
                    param_impact[param_value] = {
                        'avg_return': avg_return,
                        'avg_sharpe': avg_sharpe,
                        'avg_win_rate': avg_win_rate,
                        'avg_trades': avg_trades,
                        'count': len(group)
                    }
            
            parameter_analysis[param_name] = param_impact
        
        self.parameter_analysis = parameter_analysis
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics across all results."""
        if not self.results:
            return {}
        
        returns = [r['metrics']['total_return'] for r in self.results]
        win_rates = [r['metrics']['win_rate'] for r in self.results]
        trade_counts = [r['metrics']['total_trades'] for r in self.results]
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in self.results]
        
        return {
            'total_combinations': len(self.results),
            'profitable_strategies': sum(1 for r in returns if r > 0),
            'profitability_rate': sum(1 for r in returns if r > 0) / len(returns),
            'avg_return': np.mean(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'avg_win_rate': np.mean(win_rates),
            'avg_trade_count': np.mean(trade_counts),
            'avg_sharpe': np.mean(sharpe_ratios),
            'return_std': np.std(returns)
        }
    
    def _print_optimization_summary(self):
        """Print optimization summary."""
        print("\n" + "="*60)
        print("VECTORBT OPTIMIZATION SUMMARY")
        print("="*60)
        
        summary = self.optimization_results['summary_stats']
        print(f"Total combinations tested: {summary['total_combinations']}")
        print(f"Profitable strategies: {summary['profitable_strategies']} ({summary['profitability_rate']*100:.1f}%)")
        print(f"Average return: {summary['avg_return']*100:.2f}%")
        print(f"Best return: {summary['best_return']*100:.2f}%")
        print(f"Average win rate: {summary['avg_win_rate']*100:.1f}%")
        print(f"Average Sharpe ratio: {summary['avg_sharpe']:.3f}")
        
        # Print best strategies
        if self.best_results['best_return']:
            print("\nBEST STRATEGY BY RETURN:")
            best = self.best_results['best_return']
            print(f"  Pattern: {best['params']['pattern']}")
            print(f"  Lag: {best['params']['lag']} periods")
            print(f"  Stop Loss: {best['params']['stop_loss']*100:.1f}%")
            print(f"  Take Profit: {best['params']['take_profit']*100:.1f}%")
            print(f"  Trailing Stop: {best['params']['trailing_stop']*100:.1f}%")
            print(f"  Position Size: {best['params']['position_size']*100:.0f}%")
            print(f"  Return: {best['metrics']['total_return']*100:.2f}%")
            print(f"  Win Rate: {best['metrics']['win_rate']*100:.1f}%")
            print(f"  Sharpe: {best['metrics']['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {best['metrics']['max_drawdown']*100:.2f}%")
            print(f"  Total Trades: {best['metrics']['total_trades']}")
    
    def generate_comprehensive_report(self, output_dir):
        """Generate comprehensive HTML and chart reports."""
        print("\nGenerating comprehensive reports...")
        
        os.makedirs(output_dir, exist_ok=True)
        charts_dir = os.path.join(output_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Generate charts
        self._generate_optimization_charts(charts_dir)
        
        # Generate HTML report
        html_file = os.path.join(output_dir, f'vectorbt_optimization_{self.altcoin_name}.html')
        self._generate_html_report(html_file, charts_dir)
        
        print(f"Reports generated in: {output_dir}")
        return html_file
    
    def _generate_optimization_charts(self, charts_dir):
        """Generate comprehensive optimization charts."""
        # 1. Performance distribution
        self._plot_performance_distribution(charts_dir)
        
        # 2. Parameter impact charts
        self._plot_parameter_impact(charts_dir)
        
        # 3. Best strategy equity curves
        self._plot_best_strategies_equity(charts_dir)
        
        # 4. Risk-return scatter
        self._plot_risk_return_scatter(charts_dir)
        
        # 5. Trailing stop analysis
        self._plot_trailing_stop_analysis(charts_dir)
    
    def _plot_performance_distribution(self, charts_dir):
        """Plot distribution of strategy returns."""
        returns = [r['metrics']['total_return']*100 for r in self.results]
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Return distribution
        plt.subplot(2, 2, 1)
        plt.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Break-even')
        plt.xlabel('Total Return (%)')
        plt.ylabel('Number of Strategies')
        plt.title('Distribution of Strategy Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Win rate distribution
        plt.subplot(2, 2, 2)
        win_rates = [r['metrics']['win_rate']*100 for r in self.results]
        plt.hist(win_rates, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(x=50, color='red', linestyle='--', label='50% Win Rate')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Number of Strategies')
        plt.title('Distribution of Win Rates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Trade count distribution
        plt.subplot(2, 2, 3)
        trade_counts = [r['metrics']['total_trades'] for r in self.results]
        plt.hist(trade_counts, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Number of Trades')
        plt.ylabel('Number of Strategies')
        plt.title('Distribution of Trade Counts')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Sharpe ratio distribution
        plt.subplot(2, 2, 4)
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in self.results if not np.isnan(r['metrics']['sharpe_ratio'])]
        plt.hist(sharpe_ratios, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Sharpe = 0')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Number of Strategies')
        plt.title('Distribution of Sharpe Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'performance_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_impact(self, charts_dir):
        """Plot parameter impact analysis."""
        # Focus on numerical parameters
        numerical_params = ['lag', 'stop_loss', 'take_profit', 'trailing_stop', 'position_size', 'holding_time']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(numerical_params):
            if param in self.parameter_analysis:
                param_data = self.parameter_analysis[param]
                
                values = []
                returns = []
                
                for val, metrics in param_data.items():
                    if metrics['count'] >= 3:  # Only include values with enough data
                        values.append(val)
                        returns.append(metrics['avg_return'] * 100)
                
                if values and returns:
                    # Sort by parameter value
                    sorted_data = sorted(zip(values, returns))
                    values, returns = zip(*sorted_data)
                    
                    axes[i].plot(values, returns, 'o-', linewidth=2, markersize=6)
                    axes[i].set_xlabel(param.replace('_', ' ').title())
                    axes[i].set_ylabel('Average Return (%)')
                    axes[i].set_title(f'Impact of {param.replace("_", " ").title()}')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'parameter_impact.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_best_strategies_equity(self, charts_dir):
        """Plot equity curves for best strategies."""
        plt.figure(figsize=(15, 10))
        
        # Get top 5 strategies by different metrics
        strategies_to_plot = []
        
        if self.best_results['best_return']:
            strategies_to_plot.append(('Best Return', self.best_results['best_return']))
        
        if self.best_results['best_sharpe']:
            strategies_to_plot.append(('Best Sharpe', self.best_results['best_sharpe']))
        
        if self.best_results['best_risk_adjusted']:
            strategies_to_plot.append(('Best Risk-Adjusted', self.best_results['best_risk_adjusted']))
        
        # Add top 2 from return list if available
        for i, strategy in enumerate(self.best_results['top_10_return'][:2]):
            strategies_to_plot.append((f'Top Return #{i+1}', strategy))
        
        # Plot equity curves
        for label, strategy in strategies_to_plot:
            if strategy and 'equity_curve' in strategy:
                equity = strategy['equity_curve']
                if len(equity) > 0:
                    # Normalize to percentage change from initial value
                    pct_change = (equity / equity.iloc[0] - 1) * 100
                    plt.plot(pct_change.index, pct_change.values, label=label, linewidth=2)
        
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value Change (%)')
        plt.title('Equity Curves - Best Strategies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'best_strategies_equity.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_return_scatter(self, charts_dir):
        """Plot risk-return scatter plot."""
        returns = [r['metrics']['total_return']*100 for r in self.results]
        drawdowns = [r['metrics']['max_drawdown']*100 for r in self.results]
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in self.results]
        
        plt.figure(figsize=(12, 8))
        
        # Color by Sharpe ratio
        scatter = plt.scatter(drawdowns, returns, c=sharpe_ratios, cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        plt.xlabel('Max Drawdown (%)')
        plt.ylabel('Total Return (%)')
        plt.title('Risk-Return Profile of All Strategies')
        plt.grid(True, alpha=0.3)
        
        # Add quadrant lines
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='10% Drawdown')
        
        # Highlight best strategies
        if self.best_results['best_return']:
            best = self.best_results['best_return']
            plt.scatter(best['metrics']['max_drawdown']*100, 
                       best['metrics']['total_return']*100,
                       color='red', s=200, marker='*', label='Best Return', edgecolor='black')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'risk_return_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trailing_stop_analysis(self, charts_dir):
        """Plot trailing stop impact analysis."""
        if 'trailing_stop' not in self.parameter_analysis:
            return
        
        ts_data = self.parameter_analysis['trailing_stop']
        
        ts_values = []
        avg_returns = []
        avg_win_rates = []
        avg_drawdowns = []
        
        for ts, metrics in ts_data.items():
            if metrics['count'] >= 5:  # Need sufficient data
                ts_values.append(ts * 100)  # Convert to percentage
                avg_returns.append(metrics['avg_return'] * 100)
                avg_win_rates.append(metrics['avg_win_rate'] * 100)
                
                # Calculate average drawdown for this trailing stop
                relevant_results = [r for r in self.results if r['params']['trailing_stop'] == ts]
                avg_dd = np.mean([r['metrics']['max_drawdown'] * 100 for r in relevant_results])
                avg_drawdowns.append(avg_dd)
        
        if not ts_values:
            return
        
        # Sort by trailing stop value
        sorted_data = sorted(zip(ts_values, avg_returns, avg_win_rates, avg_drawdowns))
        ts_values, avg_returns, avg_win_rates, avg_drawdowns = zip(*sorted_data)
        
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Returns vs Trailing Stop
        plt.subplot(2, 2, 1)
        plt.plot(ts_values, avg_returns, 'o-', linewidth=2, markersize=8, color='blue')
        plt.xlabel('Trailing Stop (%)')
        plt.ylabel('Average Return (%)')
        plt.title('Impact of Trailing Stop on Returns')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Subplot 2: Win Rate vs Trailing Stop
        plt.subplot(2, 2, 2)
        plt.plot(ts_values, avg_win_rates, 'o-', linewidth=2, markersize=8, color='green')
        plt.xlabel('Trailing Stop (%)')
        plt.ylabel('Average Win Rate (%)')
        plt.title('Impact of Trailing Stop on Win Rate')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        
        # Subplot 3: Drawdown vs Trailing Stop
        plt.subplot(2, 2, 3)
        plt.plot(ts_values, avg_drawdowns, 'o-', linewidth=2, markersize=8, color='red')
        plt.xlabel('Trailing Stop (%)')
        plt.ylabel('Average Max Drawdown (%)')
        plt.title('Impact of Trailing Stop on Drawdown')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Risk-adjusted return
        plt.subplot(2, 2, 4)
        risk_adj_returns = [ret / max(dd, 1) for ret, dd in zip(avg_returns, avg_drawdowns)]
        plt.plot(ts_values, risk_adj_returns, 'o-', linewidth=2, markersize=8, color='purple')
        plt.xlabel('Trailing Stop (%)')
        plt.ylabel('Risk-Adjusted Return')
        plt.title('Risk-Adjusted Performance vs Trailing Stop')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'trailing_stop_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, output_file, charts_dir):
        """Generate comprehensive HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Strategy Optimization - {self.altcoin_name.upper()}</title>
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
        .strategy-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">VectorBT Comprehensive Strategy Optimization - {self.altcoin_name.upper()}</h1>
        
        <div class="alert alert-info">
            <p>This report presents the results of comprehensive strategy optimization using VectorBT. 
            All pattern analysis results were backtested with extensive parameter combinations including trailing stops.</p>
        </div>

        <h2>Optimization Summary</h2>
        <div class="row mb-4">
"""
        
        # Add summary metrics
        summary = self.optimization_results['summary_stats']
        
        summary_metrics = [
            ("Total Combinations", summary['total_combinations'], ""),
            ("Profitable Strategies", f"{summary['profitable_strategies']} ({summary['profitability_rate']*100:.1f}%)", "positive" if summary['profitability_rate'] > 0.5 else ""),
            ("Best Return", f"{summary['best_return']*100:.2f}%", "positive" if summary['best_return'] > 0 else "negative"),
            ("Average Return", f"{summary['avg_return']*100:.2f}%", "positive" if summary['avg_return'] > 0 else "negative"),
            ("Average Win Rate", f"{summary['avg_win_rate']*100:.1f}%", "positive" if summary['avg_win_rate'] > 0.5 else ""),
            ("Average Sharpe", f"{summary['avg_sharpe']:.3f}", "positive" if summary['avg_sharpe'] > 0 else "negative")
        ]
        
        for title, value, css_class in summary_metrics:
            html_content += f"""
            <div class="col-md-2">
                <div class="metric-box">
                    <div class="metric-title">{title}</div>
                    <div class="metric-value {css_class}">{value}</div>
                </div>
            </div>
"""
        
        html_content += """
        </div>
        
        <h2>Best Strategies</h2>
"""
        
        # Add best strategies
        strategy_types = [
            ("Best Return", self.best_results['best_return']),
            ("Best Sharpe Ratio", self.best_results['best_sharpe']),
            ("Best Risk-Adjusted", self.best_results['best_risk_adjusted']),
            ("Best Win Rate", self.best_results['best_win_rate'])
        ]
        
        for strategy_name, strategy in strategy_types:
            if strategy:
                params = strategy['params']
                metrics = strategy['metrics']
                
                html_content += f"""
        <div class="strategy-card">
            <h4>{strategy_name}</h4>
            <div class="row">
                <div class="col-md-6">
                    <h5>Parameters:</h5>
                    <ul>
                        <li><strong>Pattern:</strong> {params['pattern']}</li>
                        <li><strong>Lag:</strong> {params['lag']} periods</li>
                        <li><strong>Stop Loss:</strong> {params['stop_loss']*100:.1f}%</li>
                        <li><strong>Take Profit:</strong> {params['take_profit']*100:.1f}%</li>
                        <li><strong>Trailing Stop:</strong> {params['trailing_stop']*100:.1f}%</li>
                        <li><strong>Position Size:</strong> {params['position_size']*100:.0f}%</li>
                        <li><strong>Holding Time:</strong> {params['holding_time']} periods</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h5>Performance:</h5>
                    <ul>
                        <li><strong>Total Return:</strong> <span class="{'positive' if metrics['total_return'] > 0 else 'negative'}">{metrics['total_return']*100:.2f}%</span></li>
                        <li><strong>Win Rate:</strong> {metrics['win_rate']*100:.1f}%</li>
                        <li><strong>Total Trades:</strong> {metrics['total_trades']}</li>
                        <li><strong>Max Drawdown:</strong> {metrics['max_drawdown']*100:.2f}%</li>
                        <li><strong>Sharpe Ratio:</strong> {metrics['sharpe_ratio']:.3f}</li>
                        <li><strong>Profit Factor:</strong> {metrics['profit_factor']:.2f}</li>
                        <li><strong>Final Value:</strong> ${metrics['final_value']:.2f}</li>
                    </ul>
                </div>
            </div>
        </div>
"""
        
        # Add charts
        html_content += """
        <h2>Analysis Charts</h2>
        
        <div class="chart-container">
            <h3>Performance Distributions</h3>
            <img src="charts/performance_distributions.png" class="img-fluid border rounded" alt="Performance Distributions">
        </div>
        
        <div class="chart-container">
            <h3>Parameter Impact Analysis</h3>
            <img src="charts/parameter_impact.png" class="img-fluid border rounded" alt="Parameter Impact">
        </div>
        
        <div class="chart-container">
            <h3>Best Strategies Equity Curves</h3>
            <img src="charts/best_strategies_equity.png" class="img-fluid border rounded" alt="Best Strategies Equity">
        </div>
        
        <div class="chart-container">
            <h3>Risk-Return Analysis</h3>
            <img src="charts/risk_return_scatter.png" class="img-fluid border rounded" alt="Risk Return Scatter">
        </div>
        
        <div class="chart-container">
            <h3>Trailing Stop Analysis</h3>
            <img src="charts/trailing_stop_analysis.png" class="img-fluid border rounded" alt="Trailing Stop Analysis">
        </div>
"""
        
        # Add top strategies table
        html_content += """
        <h2>Top 20 Strategies by Return</h2>
        <div class="table-responsive">
            <table class="table table-striped table-sm">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Pattern</th>
                        <th>Lag</th>
                        <th>Return %</th>
                        <th>Win Rate %</th>
                        <th>Trades</th>
                        <th>Sharpe</th>
                        <th>Max DD %</th>
                        <th>Stop Loss %</th>
                        <th>Take Profit %</th>
                        <th>Trail Stop %</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add top 20 strategies
        for i, strategy in enumerate(self.best_results['top_10_return'][:20]):
            params = strategy['params']
            metrics = strategy['metrics']
            
            html_content += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{params['pattern']}</td>
                        <td>{params['lag']}</td>
                        <td class="{'positive' if metrics['total_return'] > 0 else 'negative'}">{metrics['total_return']*100:.2f}</td>
                        <td>{metrics['win_rate']*100:.1f}</td>
                        <td>{metrics['total_trades']}</td>
                        <td>{metrics['sharpe_ratio']:.3f}</td>
                        <td>{metrics['max_drawdown']*100:.2f}</td>
                        <td>{params['stop_loss']*100:.1f}</td>
                        <td>{params['take_profit']*100:.1f}</td>
                        <td>{params['trailing_stop']*100:.1f}</td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_file}")

    def generate_enhanced_reports(self, results_dir, enable_trailing_stop=True):
        """Generate comprehensive optimization reports."""
        # Save files directly to results_dir (no subfolder)
        report_dir = results_dir
        
        # Create proper folder structure in the main results directory
        data_dir = os.path.join(report_dir, 'data')
        charts_dir = os.path.join(report_dir, 'charts') 
        html_dir = os.path.join(report_dir, 'html')
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(html_dir, exist_ok=True)
        
        # Check if we have any results to report
        if not self.results or len(self.results) == 0:
            print("⚠️ No results to generate reports for. Creating a minimal report...")
            html_file = os.path.join(report_dir, 'optimization_report.html')
            self._generate_no_results_report(html_file)
            return html_file
        
        # Generate HTML report in root (for main linking)
        html_file = os.path.join(report_dir, 'optimization_report.html')
        self._generate_enhanced_html_report(html_file, enable_trailing_stop, data_dir, charts_dir)
        
        # Generate CSV reports in data folder
        self._generate_csv_reports(data_dir)
        
        # Generate visualizations in charts folder
        self._generate_enhanced_visualizations(charts_dir, enable_trailing_stop)
        
        # Generate parameter analysis in data folder
        self._generate_parameter_analysis_report(data_dir)
        
        # Generate individual trades data
        self._generate_trades_data(data_dir)
        
        return html_file
    
    def _generate_no_results_report(self, filename):
        """Generate a report when no valid results are found."""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Strategy Optimization Report - No Results - {self.altcoin_name.upper()}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                .alert-custom {{
                    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                    border: none;
                    border-radius: 10px;
                    padding: 30px;
                    margin: 20px 0;
                }}
                .recommendations {{
                    background: #f8f9fa;
                    border-left: 4px solid #007bff;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 0 5px 5px 0;
                }}
            </style>
        </head>
        <body class="bg-light">
            <div class="container-fluid">
                <div class="row">
                    <div class="col-12">
                        <div class="d-flex justify-content-between align-items-center py-3">
                            <h1 class="display-4"><i class="fas fa-exclamation-triangle text-warning"></i> Strategy Optimization Report</h1>
                            <div class="text-muted">
                                <i class="fas fa-coins"></i> {self.altcoin_name.upper()}
                                <br>
                                <small>{datetime.now().strftime("%B %d, %Y at %H:%M")}</small>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12">
                        <div class="alert-custom text-center">
                            <h2><i class="fas fa-search text-danger" style="font-size: 3em;"></i></h2>
                            <h3>No Valid Trading Strategies Found</h3>
                            <p class="lead">The optimization process completed but did not find any profitable strategies with the current parameters and data.</p>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="recommendations">
                            <h4><i class="fas fa-lightbulb text-warning"></i> Possible Reasons</h4>
                            <ul>
                                <li><strong>Limited Data:</strong> The dataset might not contain enough pattern occurrences</li>
                                <li><strong>Pattern Quality:</strong> The detected patterns might not be profitable</li>
                                <li><strong>Parameter Constraints:</strong> Stop loss/take profit levels might be too restrictive</li>
                                <li><strong>Market Conditions:</strong> The time period might not favor the selected patterns</li>
                                <li><strong>Data Quality:</strong> Issues with price data or pattern detection</li>
                            </ul>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="recommendations">
                            <h4><i class="fas fa-cogs text-primary"></i> Recommendations</h4>
                            <ul>
                                <li><strong>Adjust Parameters:</strong> Try wider stop loss/take profit ranges</li>
                                <li><strong>Different Time Period:</strong> Test with different market conditions</li>
                                <li><strong>More Data:</strong> Use longer historical periods</li>
                                <li><strong>Pattern Review:</strong> Check if patterns are being detected correctly</li>
                                <li><strong>Lower Constraints:</strong> Reduce minimum trade requirements</li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="text-center">
                            <div class="card border-0 shadow-sm mb-4">
                                <div class="card-body">
                                    <h5 class="mb-3">Navigation</h5>
                                    <a href="../reports/index.html" class="btn btn-outline-primary me-2">← Back to Main Analysis</a>
                                    <a href="../../index.html" class="btn btn-outline-secondary">View All Results</a>
                                </div>
                            </div>
                        </div>
                        <div class="text-center text-muted">
                            <hr>
                            <p><i class="fas fa-robot"></i> Strategy Optimization Report Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            <p><i class="fas fa-info-circle"></i> No trading strategies found - consider adjusting parameters and trying again</p>
                        </div>
                    </div>
                </div>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📝 No-results report generated: {filename}")
    
    def _generate_enhanced_html_report(self, filename, enable_trailing_stop=True, data_dir=None, charts_dir=None):
        """Generate enhanced HTML report with comprehensive analysis."""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced Strategy Optimization Report - {self.altcoin_name.upper()}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                .metric-card {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 15px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                .strategy-card {{
                    border: 1px solid #e3e6f0;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    transition: all 0.3s ease;
                }}
                .strategy-card:hover {{
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    transform: translateY(-2px);
                }}
                .positive {{ color: #28a745; font-weight: bold; }}
                .negative {{ color: #dc3545; font-weight: bold; }}
                .neutral {{ color: #6c757d; }}
                .performance-badge {{
                    padding: 5px 10px;
                    border-radius: 20px;
                    font-size: 0.8em;
                    font-weight: bold;
                }}
                .performance-excellent {{ background-color: #28a745; color: white; }}
                .performance-good {{ background-color: #17a2b8; color: white; }}
                .performance-average {{ background-color: #ffc107; color: black; }}
                .performance-poor {{ background-color: #dc3545; color: white; }}
                .chart-container {{
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                    margin-bottom: 20px;
                }}
                .parameter-insight {{
                    background: #f8f9fa;
                    border-left: 4px solid #007bff;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 0 5px 5px 0;
                }}
                .trailing-stop-analysis {{
                    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body class="bg-light">
            <div class="container">
                <div class="row">
                    <div class="col-12">
                        <div class="d-flex justify-content-between align-items-center py-3">
                            <h1 class="display-4"><i class="fas fa-chart-line text-primary"></i> Enhanced Strategy Optimization Report</h1>
                            <div class="text-muted">
                                <i class="fas fa-coins"></i> {self.altcoin_name.upper()}
                                <br>
                                <small>{datetime.now().strftime("%B %d, %Y at %H:%M")}</small>
                            </div>
                        </div>
                    </div>
                </div>
        """
        
        # Add summary metrics
        summary = self.optimization_results['summary_stats']
        html_content += f"""
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3><i class="fas fa-vials"></i></h3>
                            <h2>{summary['total_combinations']:,}</h2>
                            <p class="mb-0">Combinations Tested</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3><i class="fas fa-trophy"></i></h3>
                            <h2>{summary['profitable_strategies']:,}</h2>
                            <p class="mb-0">Profitable Strategies</p>
                            <small>({summary['profitability_rate']*100:.1f}% success rate)</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3><i class="fas fa-chart-area"></i></h3>
                            <h2>{summary['best_return']*100:.2f}%</h2>
                            <p class="mb-0">Best Return</p>
                            <small>Avg: {summary['avg_return']*100:.2f}%</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3><i class="fas fa-bullseye"></i></h3>
                            <h2>{summary['avg_win_rate']*100:.1f}%</h2>
                            <p class="mb-0">Average Win Rate</p>
                            <small>Sharpe: {summary['avg_sharpe']:.3f}</small>
                        </div>
                    </div>
                </div>
        """
        
        # Add best strategies section
        html_content += """
                <div class="row">
                    <div class="col-12">
                        <h2 class="mb-4"><i class="fas fa-star text-warning"></i> Top Performing Strategies</h2>
                    </div>
                </div>
        """
        
        # Top strategies by different metrics
        strategy_types = [
            ('best_return', 'Highest Return', 'fas fa-chart-line', 'success'),
            ('best_sharpe', 'Best Sharpe Ratio', 'fas fa-balance-scale', 'info'),
            ('best_win_rate', 'Highest Win Rate', 'fas fa-bullseye', 'warning'),
            ('best_profit_factor', 'Best Profit Factor', 'fas fa-coins', 'primary')
        ]
        
        for strategy_key, title, icon, color in strategy_types:
            if strategy_key in self.best_results and self.best_results[strategy_key]:
                strategy = self.best_results[strategy_key]
                params = strategy['params']
                metrics = strategy['metrics']
                
                html_content += f"""
                <div class="col-md-6 mb-4">
                    <div class="strategy-card border-{color}">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="text-{color}"><i class="{icon}"></i> {title}</h5>
                            <span class="performance-badge performance-excellent">
                                {self._get_performance_class(metrics['total_return'])}
                            </span>
                        </div>
                        <div class="row">
                            <div class="col-6">
                                <strong>Parameters:</strong>
                                <ul class="list-unstyled mt-2">
                                    <li><i class="fas fa-project-diagram"></i> Pattern: <code>{params['pattern']}</code></li>
                                    <li><i class="fas fa-clock"></i> Lag: {params['lag']} periods</li>
                                    <li><i class="fas fa-hand-paper"></i> Stop Loss: {params['stop_loss']*100:.2f}%</li>
                                    <li><i class="fas fa-flag-checkered"></i> Take Profit: {params['take_profit']*100:.2f}%</li>
                                    <li><i class="fas fa-chart-area"></i> Trailing Stop: {params['trailing_stop']*100:.2f}%</li>
                                    <li><i class="fas fa-percentage"></i> Position Size: {params['position_size']*100:.0f}%</li>
                                    <li><i class="fas fa-hourglass-half"></i> Holding Time: {params['holding_time']} periods</li>
                                </ul>
                            </div>
                            <div class="col-6">
                                <strong>Performance:</strong>
                                <ul class="list-unstyled mt-2">
                                    <li>📈 Return: <span class="{self._get_return_class(metrics['total_return'])}">{metrics['total_return']*100:.2f}%</span></li>
                                    <li>🎯 Win Rate: {metrics['win_rate']*100:.1f}%</li>
                                    <li>⚡ Sharpe: {metrics['sharpe_ratio']:.3f}</li>
                                    <li>📉 Max Drawdown: {metrics['max_drawdown']*100:.2f}%</li>
                                    <li>🔢 Total Trades: {metrics['total_trades']}</li>
                                    <li>💰 Profit Factor: {metrics['profit_factor']:.2f}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                """
        
        # Add trailing stop analysis if enabled
        if enable_trailing_stop and hasattr(self, 'trailing_stop_analysis') and self.trailing_stop_analysis:
            html_content += """
                <div class="row mt-5">
                    <div class="col-12">
                        <h2 class="mb-4"><i class="fas fa-chart-area text-info"></i> Trailing Stop Analysis</h2>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="trailing-stop-analysis">
                            <h5><i class="fas fa-microscope"></i> Trailing Stop Performance Analysis</h5>
                            <p class="mb-3">Analysis of different trailing stop percentages and their impact on strategy performance.</p>
            """
            
            # Sort trailing stop analysis by average return
            sorted_ts = sorted(self.trailing_stop_analysis.items(), 
                             key=lambda x: x[1]['avg_return'], reverse=True)
            
            html_content += """
                            <div class="table-responsive">
                                <table class="table table-hover bg-white">
                                    <thead class="table-dark">
                                        <tr>
                                            <th>Trailing Stop %</th>
                                            <th>Avg Return %</th>
                                            <th>Win Rate %</th>
                                            <th>Avg Sharpe</th>
                                            <th>Profit Probability</th>
                                            <th>Sample Size</th>
                                        </tr>
                                    </thead>
                                    <tbody>
            """
            
            for ts_value, data in sorted_ts[:10]:  # Top 10 trailing stop values
                html_content += f"""
                                        <tr>
                                            <td><strong>{ts_value*100:.1f}%</strong></td>
                                            <td><span class="{self._get_return_class(data['avg_return'])}">{data['avg_return']*100:.2f}%</span></td>
                                            <td>{data['avg_win_rate']*100:.1f}%</td>
                                            <td>{data['avg_sharpe']:.3f}</td>
                                            <td>{data['profit_probability']*100:.1f}%</td>
                                            <td>{data['sample_size']}</td>
                                        </tr>
                """
            
            html_content += """
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        # Add parameter insights
        if 'optimal_ranges' in self.optimization_results:
            html_content += """
                <div class="row mt-5">
                    <div class="col-12">
                        <h2 class="mb-4"><i class="fas fa-lightbulb text-warning"></i> Parameter Insights</h2>
                    </div>
                </div>
            """
            
            optimal_ranges = self.optimization_results['optimal_ranges']
            for param, ranges in optimal_ranges.items():
                param_name = param.replace('_', ' ').title()
                html_content += f"""
                <div class="col-md-6 mb-3">
                    <div class="parameter-insight">
                        <h6><i class="fas fa-cog"></i> {param_name}</h6>
                        <p class="mb-2">
                            <strong>Optimal Range:</strong> {ranges['min']:.3f} - {ranges['max']:.3f}<br>
                            <strong>Recommended Value:</strong> {ranges['median']:.3f}<br>
                            <strong>Most Common:</strong> {ranges['most_common']:.3f}
                        </p>
                    </div>
                </div>
                """
        
        # Add footer with navigation
        html_content += f"""
                <div class="row mt-5">
                    <div class="col-12">
                        <div class="text-center">
                            <div class="card border-0 shadow-sm mb-4">
                                <div class="card-body">
                                    <h5 class="mb-3">Navigation</h5>
                                    <a href="../reports/index.html" class="btn btn-outline-primary me-2">← Back to Main Analysis</a>
                                    <a href="../../index.html" class="btn btn-outline-secondary">View All Results</a>
                                </div>
                            </div>
                        </div>
                        <div class="text-center text-muted">
                            <hr>
                            <p><i class="fas fa-robot"></i> Enhanced Strategy Optimization Report Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            <p>{'<i class="fas fa-chart-area"></i> Trailing Stop Analysis Enabled' if enable_trailing_stop else '<i class="fas fa-ban"></i> Trailing Stop Analysis Disabled'}</p>
                        </div>
                    </div>
                </div>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _get_performance_class(self, return_value):
        """Get performance class based on return value."""
        if return_value > 0.1:  # >10%
            return "Excellent"
        elif return_value > 0.05:  # >5%
            return "Good"
        elif return_value > 0:  # >0%
            return "Average"
        else:
            return "Poor"
    
    def _get_return_class(self, return_value):
        """Get CSS class based on return value."""
        if return_value > 0:
            return "positive"
        elif return_value < 0:
            return "negative"
        else:
            return "neutral"
    
    def _generate_csv_reports(self, report_dir):
        """Generate CSV reports for detailed analysis."""
        # All results CSV
        results_data = []
        for result in self.results:
            row = {}
            row.update(result['params'])
            row.update(result['metrics'])
            results_data.append(row)
        
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(os.path.join(report_dir, 'all_results.csv'), index=False)
        
        # Top strategies CSV
        top_strategies = sorted(self.results, key=lambda x: x['metrics']['total_return'], reverse=True)[:100]
        top_data = []
        for result in top_strategies:
            row = {}
            row.update(result['params'])
            row.update(result['metrics'])
            top_data.append(row)
        
        df_top = pd.DataFrame(top_data)
        df_top.to_csv(os.path.join(report_dir, 'top_100_strategies.csv'), index=False)
        
        # Parameter analysis CSV
        if hasattr(self, 'parameter_analysis'):
            param_df = pd.DataFrame(self.parameter_analysis).T
            param_df.to_csv(os.path.join(report_dir, 'parameter_analysis.csv'))
        
        # Trailing stop analysis CSV
        if hasattr(self, 'trailing_stop_analysis'):
            ts_df = pd.DataFrame(self.trailing_stop_analysis).T
            ts_df.to_csv(os.path.join(report_dir, 'trailing_stop_analysis.csv'))
        
        print(f"📄 CSV reports saved to {report_dir}")
    
    def _generate_enhanced_visualizations(self, charts_dir, enable_trailing_stop=True):
        """Generate enhanced visualizations."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Charts directory is already provided, just ensure it exists
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. Return distribution
        returns = [r['metrics']['total_return'] for r in self.results]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns)*100:.2f}%')
        plt.axvline(np.median(returns), color='green', linestyle='--', label=f'Median: {np.median(returns)*100:.2f}%')
        plt.xlabel('Total Return')
        plt.ylabel('Frequency')
        plt.title('Distribution of Strategy Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Return vs Risk scatter
        plt.subplot(1, 2, 2)
        risks = [r['metrics']['max_drawdown'] for r in self.results]
        plt.scatter(risks, returns, alpha=0.6, c=returns, cmap='RdYlGn')
        plt.xlabel('Max Drawdown')
        plt.ylabel('Total Return')
        plt.title('Risk vs Return Analysis')
        plt.colorbar(label='Return')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'return_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Parameter impact analysis
        if hasattr(self, 'parameter_analysis'):
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            params = ['stop_loss', 'take_profit', 'trailing_stop', 'lag', 'position_size', 'holding_time']
            
            for i, param in enumerate(params):
                if i < len(axes) and param in self.parameter_analysis:
                    data = self.parameter_analysis[param]
                    values = list(data.keys())
                    avg_returns = [data[v]['avg_return'] for v in values]
                    
                    axes[i].scatter(values, avg_returns, alpha=0.7)
                    axes[i].set_xlabel(param.replace('_', ' ').title())
                    axes[i].set_ylabel('Average Return')
                    axes[i].set_title(f'{param.replace("_", " ").title()} Impact')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'parameter_impact.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Trailing stop analysis visualization
        if enable_trailing_stop and hasattr(self, 'trailing_stop_analysis'):
            plt.figure(figsize=(14, 8))
            
            ts_values = list(self.trailing_stop_analysis.keys())
            avg_returns = [self.trailing_stop_analysis[ts]['avg_return'] for ts in ts_values]
            win_rates = [self.trailing_stop_analysis[ts]['avg_win_rate'] for ts in ts_values]
            sample_sizes = [self.trailing_stop_analysis[ts]['sample_size'] for ts in ts_values]
            
            # Plot 1: Return vs Trailing Stop
            plt.subplot(2, 2, 1)
            plt.scatter(ts_values, avg_returns, s=[s/10 for s in sample_sizes], alpha=0.7)
            plt.xlabel('Trailing Stop %')
            plt.ylabel('Average Return')
            plt.title('Trailing Stop Impact on Returns')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Win Rate vs Trailing Stop
            plt.subplot(2, 2, 2)
            plt.scatter(ts_values, win_rates, s=[s/10 for s in sample_sizes], alpha=0.7, color='orange')
            plt.xlabel('Trailing Stop %')
            plt.ylabel('Average Win Rate')
            plt.title('Trailing Stop Impact on Win Rate')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Sample sizes
            plt.subplot(2, 2, 3)
            plt.bar(range(len(ts_values)), sample_sizes, alpha=0.7, color='green')
            plt.xlabel('Trailing Stop Index')
            plt.ylabel('Sample Size')
            plt.title('Sample Sizes per Trailing Stop Value')
            plt.xticks(range(len(ts_values)), [f'{ts*100:.1f}%' for ts in ts_values], rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Return distribution by trailing stop
            plt.subplot(2, 2, 4)
            ts_data = []
            ts_labels = []
            for ts in sorted(ts_values)[:10]:  # Top 10 values
                ts_returns = [r['metrics']['total_return'] for r in self.results 
                            if r['params']['trailing_stop'] == ts]
                if len(ts_returns) >= 10:
                    ts_data.append(ts_returns)
                    ts_labels.append(f'{ts*100:.1f}%')
            
            if ts_data:
                plt.boxplot(ts_data, labels=ts_labels)
                plt.xlabel('Trailing Stop %')
                plt.ylabel('Return Distribution')
                plt.title('Return Distribution by Trailing Stop')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'trailing_stop_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Enhanced visualizations saved to {charts_dir}")
    
    def _generate_parameter_analysis_report(self, report_dir):
        """Generate detailed parameter analysis report."""
        if not hasattr(self, 'parameter_analysis'):
            return
        
        report_file = os.path.join(report_dir, 'parameter_analysis.txt')
        
        with open(report_file, 'w') as f:
            f.write("ENHANCED PARAMETER ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            for param, data in self.parameter_analysis.items():
                f.write(f"{param.upper().replace('_', ' ')} ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                # Sort by average return
                sorted_values = sorted(data.items(), key=lambda x: x[1]['avg_return'], reverse=True)
                
                f.write("Top 10 values by average return:\n")
                for i, (value, metrics) in enumerate(sorted_values[:10]):
                    # Handle both numeric and string values
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    
                    f.write(f"{i+1:2d}. Value: {value_str} | ")
                    f.write(f"Avg Return: {metrics['avg_return']*100:6.2f}% | ")
                    f.write(f"Win Rate: {metrics['avg_win_rate']*100:5.1f}% | ")
                    f.write(f"Samples: {metrics['count']:4d}\n")
                
                f.write("\n")
            
            # Add optimal ranges if available
            if 'optimal_ranges' in self.optimization_results:
                f.write("\nOPTIMAL PARAMETER RANGES (Top 10% Strategies):\n")
                f.write("-" * 50 + "\n")
                
                for param, ranges in self.optimization_results['optimal_ranges'].items():
                    f.write(f"{param.upper().replace('_', ' ')}:\n")
                    f.write(f"  Range: {ranges['min']:.4f} - {ranges['max']:.4f}\n")
                    f.write(f"  Recommended: {ranges['median']:.4f}\n")
                    f.write(f"  Most Common: {ranges['most_common']:.4f}\n")
                    f.write(f"  Mean: {ranges['mean']:.4f}\n\n")
        
        print(f"📋 Parameter analysis report saved to {report_file}")

    def _generate_trades_data(self, data_dir):
        """Generate individual trades data for the best strategies."""
        try:
            print("🔄 Generating individual trades data...")
            
            if not self.results or len(self.results) == 0:
                print("⚠️ No results available for trades data generation")
                return
            
            # Get top 10 strategies for trades export
            sorted_results = sorted(self.results, key=lambda x: x.get('total_return', 0), reverse=True)
            top_strategies = sorted_results[:10]
            
            trades_data = []
            
            for i, result in enumerate(top_strategies):
                try:
                    # Get the vectorbt portfolio from result
                    if 'portfolio' in result:
                        pf = result['portfolio']
                        trades = pf.trades.records_readable
                        
                        # Add strategy info to each trade
                        for _, trade in trades.iterrows():
                            trade_data = {
                                'strategy_rank': i + 1,
                                'pattern': result.get('pattern', 'unknown'),
                                'lag': result.get('lag', 0),
                                'stop_loss': result.get('stop_loss', 0),
                                'take_profit': result.get('take_profit', 0),
                                'trailing_stop': result.get('trailing_stop', 0),
                                'position_size': result.get('position_size', 0),
                                'holding_time': result.get('holding_time', 0),
                                'strategy_total_return': result.get('total_return', 0),
                                'strategy_win_rate': result.get('win_rate', 0),
                                'trade_id': trade.get('Id', ''),
                                'entry_timestamp': trade.get('Entry Timestamp', ''),
                                'exit_timestamp': trade.get('Exit Timestamp', ''),
                                'entry_price': trade.get('Avg Entry Price', 0),
                                'exit_price': trade.get('Avg Exit Price', 0),
                                'size': trade.get('Size', 0),
                                'pnl': trade.get('PnL', 0),
                                'return_pct': trade.get('Return', 0) * 100,
                                'duration_minutes': trade.get('Duration', 0),
                                'direction': 'Long' if trade.get('Size', 0) > 0 else 'Short',
                                'status': trade.get('Status', ''),
                            }
                            trades_data.append(trade_data)
                
                except Exception as e:
                    print(f"⚠️ Could not extract trades for strategy {i+1}: {str(e)}")
                    continue
            
            if trades_data:
                import pandas as pd
                trades_df = pd.DataFrame(trades_data)
                
                # Save individual trades
                trades_file = os.path.join(data_dir, 'individual_trades.csv')
                trades_df.to_csv(trades_file, index=False)
                print(f"💾 Individual trades data saved to {trades_file} ({len(trades_data)} trades)")
                
                # Generate trades summary
                summary_data = []
                for strategy_rank in trades_df['strategy_rank'].unique():
                    strategy_trades = trades_df[trades_df['strategy_rank'] == strategy_rank]
                    
                    summary = {
                        'strategy_rank': strategy_rank,
                        'pattern': strategy_trades['pattern'].iloc[0],
                        'total_trades': len(strategy_trades),
                        'winning_trades': len(strategy_trades[strategy_trades['pnl'] > 0]),
                        'losing_trades': len(strategy_trades[strategy_trades['pnl'] <= 0]),
                        'win_rate': len(strategy_trades[strategy_trades['pnl'] > 0]) / len(strategy_trades) * 100,
                        'total_pnl': strategy_trades['pnl'].sum(),
                        'avg_trade_return': strategy_trades['return_pct'].mean(),
                        'best_trade_return': strategy_trades['return_pct'].max(),
                        'worst_trade_return': strategy_trades['return_pct'].min(),
                        'avg_duration_minutes': strategy_trades['duration_minutes'].mean(),
                    }
                    summary_data.append(summary)
                
                summary_df = pd.DataFrame(summary_data)
                summary_file = os.path.join(data_dir, 'trades_summary.csv')
                summary_df.to_csv(summary_file, index=False)
                print(f"📊 Trades summary saved to {summary_file}")
                
            else:
                print("⚠️ No trade data could be extracted from results")
                
        except Exception as e:
            print(f"❌ Error generating trades data: {str(e)}")
            import traceback
            traceback.print_exc()
