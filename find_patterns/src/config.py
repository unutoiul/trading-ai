"""Configuration settings for the BTC-DOGE pattern analysis."""

# Default data paths
DEFAULT_BTC_PATH = 'data/BTC_USDT_26-February-2025_to_28-March-2025.csv'
DEFAULT_DOGE_PATH = 'data/DOGE_USDT_26-February-2025_to_28-March-2025.csv'

# Results directory
RESULTS_DIR = 'results'

# Analysis parameters
MAX_LAG = 20  # Maximum lag in minutes to analyze
MOMENTUM_WINDOWS = [5, 15, 30, 60]  # Windows for momentum calculations
RSI_WINDOW = 14  # RSI period

# Web server settings
DEFAULT_SERVER_PORT = 8080  # Default port for web server