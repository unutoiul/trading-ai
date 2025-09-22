# Trading Strategy Data

This directory contains CSV files with trading strategies exported from your pattern analysis tool.

## Expected CSV Format

The strategy manager expects CSV files with the following columns:

- `pattern`: Pattern name/identifier (e.g., "btc_bullish", "eth_breakout")
- `symbol`: Trading symbol (e.g., "BTCUSDT", "ETHUSDT")  
- `lag`: Signal lag in minutes
- `stop_loss`: Stop loss percentage (e.g., 0.025 for 2.5%)
- `take_profit`: Take profit percentage (e.g., 0.05 for 5%)
- `position_size`: Position size as decimal (e.g., 0.1 for 10% of account)
- `holding_time`: Maximum holding time in minutes
- `total_return`: Historical total return for this strategy
- `win_rate`: Historical win rate (e.g., 0.75 for 75%)
- `trailing_stop`: Trailing stop percentage (optional)
- `leverage`: Leverage multiplier (e.g., 10 for 10x)

## Example

```csv
pattern,symbol,lag,stop_loss,take_profit,position_size,holding_time,total_return,win_rate,trailing_stop,leverage
btc_bullish,BTCUSDT,0,0.025,0.05,0.1,60,0.15,0.75,0.015,10
eth_breakout,ETHUSDT,5,0.03,0.06,0.08,90,0.12,0.68,0.02,8
```

## Loading Strategies

1. Export your best performing strategies from the pattern analysis tool
2. Save them as CSV files in this directory
3. The bot will automatically load all CSV files on startup
4. Only strategies meeting the configured performance criteria will be activated

## Configuration

You can adjust strategy filtering in `config/config.yaml`:

```yaml
strategies:
  min_win_rate: 0.6    # Minimum 60% win rate
  min_return: 0.05     # Minimum 5% total return
```
