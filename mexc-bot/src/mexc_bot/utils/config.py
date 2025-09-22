"""Configuration management for MEXC trading bot."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class Config(PydanticBaseSettings):
    """Main configuration class for the MEXC trading bot."""
    
    # MEXC API Configuration
    mexc_api_key: str = Field(default="", env="MEXC_API_KEY")
    mexc_api_secret: str = Field(default="", env="MEXC_API_SECRET")
    mexc_sandbox: bool = Field(default=True, env="MEXC_SANDBOX")
    mexc_base_url: str = Field(default="https://api.mexc.com", env="MEXC_BASE_URL")
    
    # Trading Configuration
    trading_enabled: bool = Field(default=False, env="TRADING_ENABLED")
    max_positions: int = Field(default=5, env="MAX_POSITIONS")
    max_account_risk: float = Field(default=0.02, env="MAX_ACCOUNT_RISK")  # 2% of account
    default_leverage: int = Field(default=10, env="DEFAULT_LEVERAGE")
    
    # Dashboard Configuration
    dashboard_enabled: bool = Field(default=True, env="DASHBOARD_ENABLED")
    dashboard_host: str = Field(default="127.0.0.1", env="DASHBOARD_HOST")
    dashboard_port: int = Field(default=8000, env="DASHBOARD_PORT")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/mexc_bot.log", env="LOG_FILE")
    
    # Strategy Configuration
    strategy_data_path: str = Field(default="data/strategies", env="STRATEGY_DATA_PATH")
    min_win_rate: float = Field(default=0.6, env="MIN_WIN_RATE")  # 60% minimum win rate
    min_return: float = Field(default=0.05, env="MIN_RETURN")  # 5% minimum return
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from file and environment variables."""
        
        # Load from YAML file if provided or exists
        config_data = {}
        if config_file:
            config_data = self._load_yaml_config(config_file)
        else:
            # Try default config locations
            default_paths = [
                "config/config.yaml",
                "config.yaml",
                "../config/config.yaml"
            ]
            for path in default_paths:
                if Path(path).exists():
                    config_data = self._load_yaml_config(path)
                    break
        
        # Initialize with loaded data
        super().__init__(**config_data)
        
        # Ensure required directories exist
        self._create_directories()
    
    def _load_yaml_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {file_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML config: {e}")
            return {}
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            Path(self.log_file).parent,
            Path(self.strategy_data_path),
            Path("data/trades"),
            Path("logs")
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        return getattr(self, key, default)
    
    def is_valid_for_trading(self) -> bool:
        """Check if configuration is valid for trading."""
        required_fields = [
            self.mexc_api_key,
            self.mexc_api_secret
        ]
        return all(field for field in required_fields)
    
    def get_mexc_config(self) -> Dict[str, Any]:
        """Get MEXC-specific configuration."""
        return {
            "api_key": self.mexc_api_key,
            "api_secret": self.mexc_api_secret,
            "sandbox": self.mexc_sandbox,
            "base_url": self.mexc_base_url
        }
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration."""
        return {
            "enabled": self.trading_enabled,
            "max_positions": self.max_positions,
            "max_account_risk": self.max_account_risk,
            "default_leverage": self.default_leverage,
            "min_win_rate": self.min_win_rate,
            "min_return": self.min_return
        }
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard-specific configuration."""
        return {
            "enabled": self.dashboard_enabled,
            "host": self.dashboard_host,
            "port": self.dashboard_port
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging-specific configuration."""
        return {
            "level": self.log_level,
            "file": self.log_file
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
