"""
MEXC Trading Bot - Main Application Entry Point
"""
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mexc_bot.core.bot import MexcTradingBot
from mexc_bot.utils.config import Config
from mexc_bot.utils.logger import setup_logging
from mexc_bot.dashboard.app import create_dashboard_app


class TradingBotApplication:
    """Main application orchestrator for the MEXC trading bot."""
    
    def __init__(self):
        self.bot: Optional[MexcTradingBot] = None
        self.dashboard_task: Optional[asyncio.Task] = None
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Start the trading bot and dashboard."""
        try:
            # Setup logging
            setup_logging(self.config.get('logging', {}))
            self.logger.info("🚀 Starting MEXC Trading Bot Application")
            
            # Initialize trading bot
            self.bot = MexcTradingBot(self.config)
            await self.bot.initialize()
            
            # Start dashboard in background
            if self.config.get('dashboard', {}).get('enabled', True):
                self.dashboard_task = asyncio.create_task(self._run_dashboard())
            
            # Start trading bot
            self.logger.info("🤖 Starting trading bot...")
            await self.bot.start()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start application: {e}")
            await self.shutdown()
            raise
    
    async def _run_dashboard(self):
        """Run the web dashboard."""
        try:
            app = create_dashboard_app(self.bot)
            import uvicorn
            
            config = uvicorn.Config(
                app,
                host=self.config.get('dashboard', {}).get('host', '127.0.0.1'),
                port=self.config.get('dashboard', {}).get('port', 8000),
                log_level="info"
            )
            server = uvicorn.Server(config)
            self.logger.info("🌐 Starting web dashboard...")
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"❌ Dashboard failed: {e}")
    
    async def shutdown(self):
        """Graceful shutdown of the application."""
        self.logger.info("🛑 Shutting down MEXC Trading Bot...")
        
        # Stop bot
        if self.bot:
            await self.bot.stop()
        
        # Stop dashboard
        if self.dashboard_task and not self.dashboard_task.done():
            self.dashboard_task.cancel()
            try:
                await self.dashboard_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("✅ Shutdown complete")


def signal_handler(signum, frame):
    """Handle termination signals."""
    print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
    # The main loop will handle the actual shutdown
    

async def main():
    """Main entry point."""
    app = TradingBotApplication()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await app.start()
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required")
        sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
