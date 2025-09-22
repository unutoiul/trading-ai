"""Strategy management for pattern-based trading."""

import os
import csv
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd


class StrategyManager:
    """Manages trading strategies and generates signals."""
    
    def __init__(self, config: Dict[str, Any], data_path: str):
        """Initialize strategy manager."""
        self.config = config
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)
        
        # Strategy storage
        self.strategies: List[Dict[str, Any]] = []
        self.active_strategies: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> None:
        """Initialize strategy manager and load strategies."""
        self.logger.info("📊 Initializing Strategy Manager...")
        
        # Create data directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Load strategies from CSV files
        await self.load_strategies()
        
        # Filter strategies based on configuration
        self.filter_strategies()
        
        self.logger.info(f"✅ Loaded {len(self.active_strategies)} active strategies")
    
    async def load_strategies(self) -> None:
        """Load strategies from CSV files in the data directory."""
        csv_files = list(self.data_path.glob("*.csv"))
        
        if not csv_files:
            self.logger.warning(f"⚠️ No strategy CSV files found in {self.data_path}")
            # Create a sample strategy file
            await self.create_sample_strategy()
            return
        
        for csv_file in csv_files:
            try:
                strategies = self.load_strategy_from_csv(csv_file)
                self.strategies.extend(strategies)
                self.logger.info(f"📄 Loaded {len(strategies)} strategies from {csv_file.name}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load strategies from {csv_file}: {e}")
    
    def load_strategy_from_csv(self, csv_file: Path) -> List[Dict[str, Any]]:
        """Load strategies from a single CSV file."""
        strategies = []
        
        try:
            df = pd.read_csv(csv_file)
            
            # Expected columns
            required_columns = ['pattern', 'lag', 'stop_loss', 'take_profit', 'position_size', 'total_return', 'win_rate']
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"⚠️ Missing columns in {csv_file}: {missing_columns}")
                return strategies
            
            for _, row in df.iterrows():
                strategy = {
                    'id': f"{csv_file.stem}_{len(strategies)}",
                    'source_file': str(csv_file),
                    'pattern': row.get('pattern', ''),
                    'symbol': row.get('symbol', 'BTCUSDT'),  # Default symbol
                    'side': 'BUY',  # Default to long positions
                    'lag': int(row.get('lag', 0)),
                    'stop_loss': float(row.get('stop_loss', 0.02)),
                    'take_profit': float(row.get('take_profit', 0.04)),
                    'position_size': float(row.get('position_size', 0.1)),
                    'holding_time': int(row.get('holding_time', 60)),  # minutes
                    'total_return': float(row.get('total_return', 0)),
                    'win_rate': float(row.get('win_rate', 0)),
                    'trailing_stop': float(row.get('trailing_stop', 0)),
                    'leverage': int(row.get('leverage', self.config.get('default_leverage', 10))),
                    'enabled': True,
                    'last_signal_time': 0
                }
                strategies.append(strategy)
        
        except Exception as e:
            self.logger.error(f"❌ Error reading CSV file {csv_file}: {e}")
        
        return strategies
    
    def filter_strategies(self) -> None:
        """Filter strategies based on performance criteria."""
        min_win_rate = self.config.get('min_win_rate', 0.6)
        min_return = self.config.get('min_return', 0.05)
        
        self.active_strategies = []
        
        for strategy in self.strategies:
            # Check performance criteria
            if (strategy['win_rate'] >= min_win_rate and 
                strategy['total_return'] >= min_return):
                self.active_strategies.append(strategy)
            else:
                self.logger.debug(f"🚫 Strategy filtered out: {strategy['id']} "
                                f"(WR: {strategy['win_rate']:.1%}, Ret: {strategy['total_return']:.1%})")
    
    async def get_trading_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on active strategies and market data."""
        signals = []
        
        # For development, generate sample signals
        if not market_data.get('tickers'):
            return signals
        
        current_time = market_data.get('timestamp', 0)
        
        for strategy in self.active_strategies:
            try:
                # Check if enough time has passed since last signal
                if current_time - strategy['last_signal_time'] < strategy['lag'] * 60:
                    continue
                
                # Simple signal generation logic (to be enhanced with real pattern detection)
                signal = await self.generate_signal_for_strategy(strategy, market_data)
                
                if signal:
                    signals.append(signal)
                    strategy['last_signal_time'] = current_time
                
            except Exception as e:
                self.logger.error(f"❌ Error generating signal for strategy {strategy['id']}: {e}")
        
        return signals
    
    async def generate_signal_for_strategy(self, strategy: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a trading signal for a specific strategy."""
        try:
            # Simple pattern detection (placeholder - to be enhanced)
            symbol = strategy['symbol']
            
            # Find ticker for this symbol
            tickers = market_data.get('tickers', [])
            symbol_ticker = None
            
            for ticker in tickers:
                if ticker.get('symbol') == symbol:
                    symbol_ticker = ticker
                    break
            
            if not symbol_ticker:
                return None
            
            # Simple signal generation based on price change
            price_change_pct = float(symbol_ticker.get('priceChangePercent', 0))
            
            # Generate signal based on strategy pattern and price movement
            if self.should_generate_signal(strategy, price_change_pct):
                signal = {
                    'strategy_id': strategy['id'],
                    'symbol': symbol,
                    'side': strategy['side'],
                    'quantity': self.calculate_position_size(strategy, symbol_ticker),
                    'price': float(symbol_ticker.get('lastPrice', 0)),
                    'stop_loss': strategy['stop_loss'],
                    'take_profit': strategy['take_profit'],
                    'leverage': strategy['leverage'],
                    'pattern': strategy['pattern'],
                    'timestamp': market_data.get('timestamp', 0)
                }
                
                self.logger.info(f"📊 Signal generated: {symbol} {strategy['pattern']}")
                return signal
        
        except Exception as e:
            self.logger.error(f"❌ Error in signal generation: {e}")
        
        return None
    
    def should_generate_signal(self, strategy: Dict[str, Any], price_change_pct: float) -> bool:
        """Determine if a signal should be generated based on pattern and market conditions."""
        pattern = strategy['pattern'].lower()
        
        # Simple pattern-based signal logic
        if 'bullish' in pattern or 'up' in pattern:
            return price_change_pct > 0.5  # Price rising
        elif 'bearish' in pattern or 'down' in pattern:
            return price_change_pct < -0.5  # Price falling
        elif 'breakout' in pattern:
            return abs(price_change_pct) > 1.0  # Significant movement
        elif 'volatility' in pattern or 'vol' in pattern:
            return abs(price_change_pct) > 0.8  # Volatile movement
        
        # Default: generate signal occasionally for testing
        import random
        return random.random() < 0.1  # 10% chance
    
    def calculate_position_size(self, strategy: Dict[str, Any], ticker: Dict[str, Any]) -> float:
        """Calculate position size based on strategy and account balance."""
        try:
            base_size = strategy['position_size']
            price = float(ticker.get('lastPrice', 0))
            
            # For development, use a fixed notional amount
            notional_amount = base_size * 10000  # $10,000 base
            quantity = notional_amount / price if price > 0 else 0
            
            return round(quantity, 6)
        
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0.001  # Minimum position size
    
    async def should_exit_position(self, position: Dict[str, Any]) -> bool:
        """Determine if a position should be exited based on strategy rules."""
        try:
            # Find the strategy that created this position
            strategy_id = position.get('strategy_id')
            strategy = next((s for s in self.active_strategies if s['id'] == strategy_id), None)
            
            if not strategy:
                return False
            
            # Check holding time
            position_age = position.get('timestamp', 0) - position.get('entry_time', 0)
            max_holding_time = strategy['holding_time'] * 60  # Convert to seconds
            
            if position_age > max_holding_time:
                self.logger.info(f"⏰ Position exit: Max holding time reached for {position.get('symbol')}")
                return True
            
            # Additional exit conditions can be added here
            
        except Exception as e:
            self.logger.error(f"❌ Error checking position exit: {e}")
        
        return False
    
    async def create_sample_strategy(self) -> None:
        """Create a sample strategy file for demonstration."""
        sample_file = self.data_path / "sample_strategies.csv"
        
        sample_data = [
            {
                'pattern': 'btc_bullish',
                'symbol': 'BTCUSDT',
                'lag': 0,
                'stop_loss': 0.025,
                'take_profit': 0.05,
                'position_size': 0.1,
                'holding_time': 60,
                'total_return': 0.15,
                'win_rate': 0.75,
                'trailing_stop': 0.015,
                'leverage': 10
            },
            {
                'pattern': 'eth_breakout',
                'symbol': 'ETHUSDT',
                'lag': 5,
                'stop_loss': 0.03,
                'take_profit': 0.06,
                'position_size': 0.08,
                'holding_time': 90,
                'total_return': 0.12,
                'win_rate': 0.68,
                'trailing_stop': 0.02,
                'leverage': 8
            }
        ]
        
        try:
            df = pd.DataFrame(sample_data)
            df.to_csv(sample_file, index=False)
            self.logger.info(f"📝 Created sample strategy file: {sample_file}")
            
            # Load the sample strategies
            strategies = self.load_strategy_from_csv(sample_file)
            self.strategies.extend(strategies)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create sample strategy: {e}")
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance statistics for all strategies."""
        performance = {
            'total_strategies': len(self.strategies),
            'active_strategies': len(self.active_strategies),
            'avg_win_rate': sum(s['win_rate'] for s in self.active_strategies) / len(self.active_strategies) if self.active_strategies else 0,
            'avg_return': sum(s['total_return'] for s in self.active_strategies) / len(self.active_strategies) if self.active_strategies else 0,
            'strategies': self.active_strategies.copy()
        }
        
        return performance
