from flask import Flask
import ccxt
from routes.home import home_bp, create_home_route
from routes.webhook import webhook_bp, create_webhook_route

app = Flask(__name__)

# API keys for testing
api_key = "83947234d1fb255ea8acc5184ee3868e5a4d373afb857269ae710aa643483d0b"
api_secret = "eff30c89d0f5bcad58a072ea115c2d21212cb5e08a6844a81320c334b0849dcc"

# Configure Binance Futures Testnet
exchange = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "enableRateLimit": True,
    "options": {
        "defaultType": "future",
        "adjustForTimeDifference": True,
        "testnet": True,
    },
    "urls": {
        "api": {
            "public": "https://testnet.binancefuture.com/fapi/v1",
            "private": "https://testnet.binancefuture.com/fapi/v1",
        }
    }
})

# Register the home route
create_home_route(exchange)
app.register_blueprint(home_bp)

# Register the webhook route
create_webhook_route(exchange)
app.register_blueprint(webhook_bp)

if __name__ == '__main__':
    app.run(debug=True)



