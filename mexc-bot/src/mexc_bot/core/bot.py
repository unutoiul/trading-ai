"""Core trading bot implementation for MEXC exchange."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..api.mexc_client import MexcClient
from ..strategies.strategy_manager import StrategyManager
from ..risk_management.risk_manager import RiskManager
from ..utils.config import Config


class MexcTradingBot:
    """Main trading bot class that orchestrates all trading operations."""
    
    def __init__(self, config: Config):
        """Initialize the trading bot with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.mexc_client: Optional[MexcClient] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Bot state
        self.is_running = False
        self.is_initialized = False
        self.active_positions: Dict[str, Any] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Task management
        self.tasks: List[asyncio.Task] = []
        
    async def initialize(self) -> None:
        """Initialize all bot components."""
        try:
            self.logger.info("🔧 Initializing MEXC Trading Bot...")
            
            # Validate configuration
            if not self.config.is_valid_for_trading():
                raise ValueError("Invalid configuration for trading - check API credentials")
            
            # Initialize MEXC client
            self.mexc_client = MexcClient(self.config.get_mexc_config())
            await self.mexc_client.initialize()
            
            # Test API connection
            account_info = await self.mexc_client.get_account_info()
            self.logger.info(f"✅ Connected to MEXC - Account: {account_info.get('email', 'Unknown')}")
            
            # Initialize strategy manager
            self.strategy_manager = StrategyManager(
                self.config.get_trading_config(),
                self.config.strategy_data_path
            )
            await self.strategy_manager.initialize()
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                self.config.get_trading_config(),
                self.mexc_client
            )
            await self.risk_manager.initialize()
            
            self.is_initialized = True
            self.logger.info("✅ MEXC Trading Bot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize bot: {e}")
            raise
    
    async def start(self) -> None:
        """Start the trading bot main loop."""
        if not self.is_initialized:
            raise RuntimeError("Bot must be initialized before starting")
        
        if self.is_running:
            self.logger.warning("⚠️ Bot is already running")
            return
        
        self.logger.info("🚀 Starting MEXC Trading Bot...")
        self.is_running = True
        
        try:
            # Start background tasks
            self.tasks = [
                asyncio.create_task(self._main_trading_loop()),
                asyncio.create_task(self._position_monitoring_loop()),
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._market_data_loop())
            ]
            
            # Wait for all tasks to complete (or until stopped)
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"❌ Error in bot execution: {e}")
            raise
        finally:
            self.is_running = False
    
    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        self.logger.info("🛑 Stopping MEXC Trading Bot...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close all positions if configured
        if self.config.get('close_positions_on_stop', False):
            await self._close_all_positions()
        
        # Cleanup
        if self.mexc_client:
            await self.mexc_client.close()
        
        self.logger.info("✅ MEXC Trading Bot stopped")
    
    async def _main_trading_loop(self) -> None:
        """Main trading logic loop."""
        self.logger.info("📊 Starting main trading loop...")
        
        while self.is_running:
            try:
                # Check if trading is enabled
                if not self.config.trading_enabled:
                    await asyncio.sleep(10)
                    continue
                
                # Get market data and signals
                market_data = await self.mexc_client.get_market_data()
                signals = await self.strategy_manager.get_trading_signals(market_data)
                
                # Process signals
                for signal in signals:
                    await self._process_trading_signal(signal)
                
                # Wait before next iteration
                await asyncio.sleep(5)  # 5-second intervals
                
            except Exception as e:
                self.logger.error(f"❌ Error in main trading loop: {e}")
                await asyncio.sleep(10)
    
    async def _position_monitoring_loop(self) -> None:
        """Monitor and manage existing positions."""
        self.logger.info("👁️ Starting position monitoring loop...")
        
        while self.is_running:
            try:
                # Update active positions
                await self._update_active_positions()
                
                # Check for position exits
                for position_id, position in self.active_positions.items():
                    await self._check_position_exit(position_id, position)
                
                await asyncio.sleep(2)  # 2-second intervals
                
            except Exception as e:
                self.logger.error(f"❌ Error in position monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _risk_monitoring_loop(self) -> None:
        """Monitor risk levels and enforce limits."""
        self.logger.info("🛡️ Starting risk monitoring loop...")
        
        while self.is_running:
            try:
                # Check account risk
                risk_status = await self.risk_manager.check_account_risk()
                
                if risk_status.get('emergency_stop', False):
                    self.logger.critical("🚨 EMERGENCY STOP TRIGGERED - Closing all positions")
                    await self._emergency_stop()
                    break
                
                await asyncio.sleep(10)  # 10-second intervals
                
            except Exception as e:
                self.logger.error(f"❌ Error in risk monitoring: {e}")
                await asyncio.sleep(15)
    
    async def _market_data_loop(self) -> None:
        """Continuously update market data."""
        self.logger.info("📈 Starting market data loop...")
        
        while self.is_running:
            try:
                # Update market data cache
                await self.mexc_client.update_market_data_cache()
                
                await asyncio.sleep(1)  # 1-second intervals
                
            except Exception as e:
                self.logger.error(f"❌ Error in market data loop: {e}")
                await asyncio.sleep(3)
    
    async def _process_trading_signal(self, signal: Dict[str, Any]) -> None:
        """Process a trading signal and execute if valid."""
        try:
            # Risk check
            if not await self.risk_manager.validate_signal(signal):
                self.logger.debug(f"🚫 Signal rejected by risk management: {signal}")
                return
            
            # Check position limits
            if len(self.active_positions) >= self.config.max_positions:
                self.logger.debug("⚠️ Maximum positions reached, skipping signal")
                return
            
            # Execute trade
            result = await self.mexc_client.execute_trade(signal)
            
            if result.get('success', False):
                self.active_positions[result['position_id']] = result
                self.logger.info(f"✅ Trade executed: {signal['symbol']} {signal['side']} - ID: {result['position_id']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing signal: {e}")
    
    async def _update_active_positions(self) -> None:
        """Update information for all active positions."""
        try:
            positions = await self.mexc_client.get_open_positions()
            
            # Update active positions dictionary
            current_position_ids = set()
            for position in positions:
                position_id = position['position_id']
                current_position_ids.add(position_id)
                self.active_positions[position_id] = position
            
            # Remove closed positions
            closed_positions = set(self.active_positions.keys()) - current_position_ids
            for position_id in closed_positions:
                closed_position = self.active_positions.pop(position_id)
                self.trade_history.append(closed_position)
                self.logger.info(f"📝 Position closed: {position_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating positions: {e}")
    
    async def _check_position_exit(self, position_id: str, position: Dict[str, Any]) -> None:
        """Check if a position should be exited."""
        try:
            # Check strategy-based exit conditions
            should_exit = await self.strategy_manager.should_exit_position(position)
            
            if should_exit:
                result = await self.mexc_client.close_position(position_id)
                if result.get('success', False):
                    self.logger.info(f"🔄 Position closed by strategy: {position_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error checking position exit: {e}")
    
    async def _close_all_positions(self) -> None:
        """Close all active positions."""
        self.logger.info("🔄 Closing all active positions...")
        
        for position_id in list(self.active_positions.keys()):
            try:
                await self.mexc_client.close_position(position_id)
                self.logger.info(f"✅ Closed position: {position_id}")
            except Exception as e:
                self.logger.error(f"❌ Failed to close position {position_id}: {e}")
    
    async def _emergency_stop(self) -> None:
        """Emergency stop procedure."""
        self.logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        
        # Disable trading
        self.config.trading_enabled = False
        
        # Close all positions
        await self._close_all_positions()
        
        # Stop the bot
        self.is_running = False
    
    # Public methods for dashboard/API access
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "trading_enabled": self.config.trading_enabled,
            "active_positions": len(self.active_positions),
            "max_positions": self.config.max_positions,
            "total_trades": len(self.trade_history)
        }
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current active positions."""
        return self.active_positions.copy()
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade history."""
        return self.trade_history[-limit:]
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if self.mexc_client:
            return await self.mexc_client.get_account_info()
        return {}
    
    async def force_close_position(self, position_id: str) -> bool:
        """Force close a specific position."""
        try:
            if position_id in self.active_positions:
                result = await self.mexc_client.close_position(position_id)
                return result.get('success', False)
        except Exception as e:
            self.logger.error(f"❌ Failed to force close position {position_id}: {e}")
        return False
