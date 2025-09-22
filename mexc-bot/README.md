# MEXC Trading Bot

A sophisticated Python trading bot for MEXC exchange with leverage trading, pattern-based strategies, and real-time monitoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- MEXC account with API access
- Basic understanding of futures trading

### Installation

1. **Clone or setup the project:**
```bash
cd mexc-bot
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure your settings:**
```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your MEXC API credentials
```

5. **Run the bot:**
```bash
python main.py
```

## 📁 Project Structure

```
mexc-bot/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files
│   ├── config.yaml        # Main configuration
│   └── strategies/        # Strategy configurations
├── src/mexc_bot/          # Main bot package
│   ├── core/              # Core bot logic
│   ├── api/               # MEXC API integration
│   ├── strategies/        # Trading strategies
│   ├── risk_management/   # Risk management
│   ├── dashboard/         # Web dashboard
│   └── utils/             # Utilities
├── data/                  # Data storage
│   ├── strategies/        # Strategy CSV files
│   └── trades/           # Trade history
├── logs/                  # Log files
└── tests/                # Unit tests
```

## ⚙️ Configuration

### API Configuration
1. Get your MEXC API credentials:
   - Log into MEXC
   - Go to API Management
   - Create new API key with futures trading permissions

2. Update `config/config.yaml`:
```yaml
mexc:
  api_key: "your_api_key_here"
  api_secret: "your_api_secret_here"
  sandbox: true  # Set to false for live trading
```

### Strategy Configuration
Place your strategy CSV files in `data/strategies/` directory. The bot will automatically detect and load compatible strategies.

## 🤖 Features

### Core Features
- ✅ MEXC Futures API integration with leverage
- ✅ Pattern-based strategy execution
- ✅ Real-time market data processing
- ✅ Advanced risk management
- ✅ Position sizing and leverage control
- ✅ Stop-loss and take-profit management
- ✅ Web dashboard for monitoring

### Dashboard Features
- 📊 Real-time portfolio overview
- 📈 Trade history and analytics
- ⚙️ Strategy management
- 🎛️ Risk parameter configuration
- 📱 Mobile-responsive design

### Risk Management
- 💰 Account balance protection
- 📉 Maximum drawdown limits
- ⏰ Position holding time limits
- 🔄 Position size optimization
- 🚨 Emergency stop functionality

## 📊 Strategy Import

The bot can import strategies from your pattern analysis tool:

1. Export strategy CSV files from your pattern finder
2. Place them in `data/strategies/`
3. The bot will automatically load and execute them

**Supported CSV Format:**
```csv
pattern,lag,stop_loss,take_profit,position_size,holding_time,total_return,win_rate
btc_bullish,0,0.025,0.05,0.1,20,0.27,0.872
```

## 🎯 Usage Examples

### Basic Trading
```python
# The bot runs automatically based on your configuration
python main.py
```

### Web Dashboard
Open http://localhost:8000 in your browser to access the dashboard.

### Strategy Management
```python
# Import new strategies
python -m mexc_bot.utils.strategy_importer --file data/strategies/new_strategy.csv

# Validate strategies
python -m mexc_bot.utils.strategy_validator --check-all
```

## 🔧 Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
flake8 src/ tests/
```

### Type Checking
```bash
mypy src/
```

## ⚠️ Risk Disclaimer

**IMPORTANT**: This is a trading bot that can execute real trades with real money. 

- Start with paper trading (sandbox mode)
- Never risk more than you can afford to lose
- Thoroughly test all strategies before live trading
- Monitor the bot continuously during operation
- Cryptocurrency trading involves significant risk

## 📝 License

This project is for educational purposes. Use at your own risk.

## 🆘 Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review configuration in `config/config.yaml`
3. Ensure API credentials are correct
4. Verify strategy CSV format

## 🔄 Updates

Keep your bot updated by pulling the latest changes and updating dependencies:
```bash
git pull
pip install -r requirements.txt --upgrade
```
