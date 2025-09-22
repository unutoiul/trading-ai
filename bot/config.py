"""
Configuration module for the Trading Bot
Handles environment variables and application settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Binance API Configuration
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET')
    
    # Environment Configuration
    BINANCE_TESTNET = os.environ.get('BINANCE_TESTNET', 'true').lower() == 'true'
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    # Server Configuration
    PORT = int(os.environ.get('PORT', 8000))
    HOST = os.environ.get('HOST', '0.0.0.0')
    
    # API URLs
    BINANCE_TESTNET_BASE_URL = os.environ.get('BINANCE_TESTNET_BASE_URL', 'https://testnet.binancefuture.com')
    BINANCE_PRODUCTION_BASE_URL = os.environ.get('BINANCE_PRODUCTION_BASE_URL', 'https://fapi.binance.com')
    
    # Trading Configuration
    DEFAULT_LEVERAGE = int(os.environ.get('DEFAULT_LEVERAGE', 1))
    POSITION_ESTABLISHMENT_DELAY = int(os.environ.get('POSITION_ESTABLISHMENT_DELAY', 2))
    
    # Application Info
    APP_NAME = os.environ.get('APP_NAME', 'Trading Bot')
    APP_VERSION = os.environ.get('APP_VERSION', '2.0')
    
    @property
    def binance_base_url(self):
        """Get the appropriate Binance base URL based on environment"""
        return self.BINANCE_TESTNET_BASE_URL if self.BINANCE_TESTNET else self.BINANCE_PRODUCTION_BASE_URL
    
    @property
    def environment(self):
        """Get environment name"""
        return "testnet" if self.BINANCE_TESTNET else "production"
    
    def validate(self):
        """Validate required configuration"""
        if not self.BINANCE_API_KEY:
            raise ValueError("BINANCE_API_KEY environment variable is required")
        if not self.BINANCE_API_SECRET:
            raise ValueError("BINANCE_API_SECRET environment variable is required")
        
        return True

# Create global config instance
config = Config()