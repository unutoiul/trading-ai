from flask import Blueprint
from ccxt import ExchangeError

# Create a Blueprint for the home route
home_bp = Blueprint('home', __name__)

def create_home_route(exchange):
    @home_bp.route('/')
    def home():
        try:
            # Fetch balance
            balance = exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 'N/A')

            # Fetch ticker information
            ticker = exchange.fetch_ticker('BTC/USDT')
            btc_price = ticker['last']  # Current price of BTC/USDT

            # Fetch all assets and their balances
            assets = []
            for asset, details in balance.items():
                if isinstance(details, dict) and details.get('free', 0) > 0:
                    assets.append(f"{asset}: {details['free']}")

            # Fetch open positions
            positions = exchange.fetch_positions()
            open_positions = []
            for position in positions:
                if float(position.get('contracts', 0)) > 0:  # Only include open positions
                    open_positions.append(
                        f"Symbol: {position['symbol']}, Size: {position['contracts']}, Side: {position['side']}"
                    )

            # Format the response
            assets_info = "<br>".join(assets) if assets else "No assets found."
            positions_info = "<br>".join(open_positions) if open_positions else "No open positions."

            return (
                f"Binance Testnet Webhook is running!<br>"
                f"USDT Balance: {usdt_balance}<br>"
                f"BTC/USDT Price: {btc_price}<br><br>"
                f"Assets:<br>{assets_info}<br><br>"
                f"Open Positions:<br>{positions_info}"
            )
        except ExchangeError as e:
            return f"Error fetching data from Binance: {str(e)}"
        except Exception as e:
            return f"Error fetching data: {str(e)}"