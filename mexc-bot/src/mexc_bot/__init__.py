"""MEXC Bot Package Initialization"""

__version__ = "1.0.0"
__author__ = "Trading AI Team"
__description__ = "MEXC Trading Bot with Pattern-Based Strategies"

# Package-level imports for convenience
from .core.bot import MexcTradingBot
from .api.mexc_client import MexcClient
from .strategies.strategy_manager import StrategyManager
from .risk.risk_manager import RiskManager, RiskMetrics
from .dashboard.dashboard_manager import DashboardManager
from .utils.config import Config
from .utils.logger import setup_logger

__all__ = [
    "MexcTradingBot", 
    "MexcClient",
    "StrategyManager",
    "RiskManager", 
    "RiskMetrics",
    "DashboardManager",
    "Config",
    "setup_logger"
]
