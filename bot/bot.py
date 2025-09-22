from flask import Flask, Blueprint, request, jsonify
import os
import ccxt
import time
import requests
import hmac
import hashlib
import json
import datetime
from urllib.parse import urlencode
from ccxt import ExchangeError

app = Flask(__name__)

# API keys - use environment variables in production
api_key = os.environ.get('BINANCE_API_KEY', "bktTGxo34BKDupIiQB1PsHiQul1s2n5MpjD863CZ5IhegbiOlx0u3Jm9TH7zMK32")
api_secret = os.environ.get('BINANCE_API_SECRET', "UNaOl1KOpPXlRLWttS8da2aIFdOSMUPawCBiADYtIYPvfWhe8R1B3cbh36Ksxwml")
testnet_mode = os.environ.get('BINANCE_TESTNET', 'true').lower() == 'true'

def create_exchange():
    """Create exchange instance when needed to avoid startup hangs"""
    base_url = "https://testnet.binancefuture.com" if testnet_mode else "https://fapi.binance.com"
    
    return ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,
            "testnet": testnet_mode,
        },
        "urls": {
            "api": {
                "public": f"{base_url}/fapi/v1",
                "private": f"{base_url}/fapi/v1",
            }
        } if testnet_mode else {}
    })

# Ensure logs directory exists
def ensure_logs_dir():
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir

# Function to log webhook requests
def log_webhook_request(data, status, message, order=None):
    logs_dir = ensure_logs_dir()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(logs_dir, f"webhook_logs_{datetime.datetime.now().strftime('%Y-%m-%d')}.txt")
    
    log_entry = {
        "timestamp": timestamp,
        "status": status,
        "request_data": data,
        "message": message
    }
    
    if order:
        log_entry["order"] = order
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry, indent=2) + "\n\n")

class BinancePositionTPSL:
    """Binance Position-Level TP/SL Manager - Creates orders that show ROI% in UI"""
    
    def __init__(self):
        self.api_key = api_key
        self.secret_key = api_secret
        self.base_url = "https://testnet.binancefuture.com" if testnet_mode else "https://fapi.binance.com"
    
    def _generate_signature(self, params):
        """Generate signature for Binance API"""
        query_string = urlencode(params)
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint, params):
        """Make authenticated request to Binance API"""
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._generate_signature(params)
        
        headers = {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, params=params, headers=headers)
        
        return response
    
    def create_position_tpsl_orders(self, symbol, action, amount, current_price, tp_percent=None, sl_percent=None):
        """Create position-level TP/SL orders using batch API"""
        try:
            position_side = "LONG" if action == "buy" else "SHORT"
            batch_orders = []
            
            # Calculate TP/SL prices and create batch orders
            if tp_percent:
                tp_decimal = float(tp_percent) / 100
                if position_side == "LONG":
                    tp_price = current_price * (1 + tp_decimal)
                    tp_side = "SELL"
                else:
                    tp_price = current_price * (1 - tp_decimal)
                    tp_side = "BUY"
                
                tp_order = {
                    "symbol": symbol.replace('/', ''),
                    "side": tp_side,
                    "type": "TAKE_PROFIT_MARKET",
                    "quantity": str(amount),
                    "stopPrice": f"{tp_price:.6f}",
                    "reduceOnly": "true",
                    "workingType": "MARK_PRICE",
                    "priceProtect": "true"
                }
                batch_orders.append(tp_order)
            
            if sl_percent:
                sl_decimal = float(sl_percent) / 100
                if position_side == "LONG":
                    sl_price = current_price * (1 - sl_decimal)
                    sl_side = "SELL"
                else:
                    sl_price = current_price * (1 + sl_decimal)
                    sl_side = "BUY"
                
                sl_order = {
                    "symbol": symbol.replace('/', ''),
                    "side": sl_side,
                    "type": "STOP_MARKET",
                    "quantity": str(amount),
                    "stopPrice": f"{sl_price:.6f}",
                    "reduceOnly": "true",
                    "workingType": "MARK_PRICE",
                    "priceProtect": "true"
                }
                batch_orders.append(sl_order)
            
            if batch_orders:
                # Create batch orders
                params = {
                    "batchOrders": json.dumps(batch_orders)
                }
                
                response = self._make_request("/fapi/v1/batchOrders", params)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"Batch order failed: {response.status_code} - {response.text}")
            
            return []
            
        except Exception as e:
            raise Exception(f"Position TP/SL creation failed: {str(e)}")

# Routes
@app.route('/')
def home():
    return jsonify({
        "status": "Trading Bot is running!",
        "version": "2.0",
        "features": [
            "Position-level TP/SL with ROI% display",
            "Binance Futures integration",
            "Batch order execution",
            "Comprehensive logging"
        ],
        "endpoints": {
            "webhook": "/webhook",
            "logs": "/logs"
        },
        "environment": "testnet" if testnet_mode else "production"
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        # Log the incoming request
        request_data = request.json if request.is_json else "Invalid JSON"
        log_webhook_request(request_data, "received", "Webhook request received")
        
        # Ensure the request contains JSON data
        if not request.json or "action" not in request.json:
            error_msg = "Invalid request. 'action' key is required."
            log_webhook_request(request_data, "error", error_msg)
            return jsonify({"status": "error", "message": error_msg}), 400

        data = request.json
        symbol = data["symbol"]
        amount = data["amount"]
        action = data["action"]
        
        # Create exchange instance when needed
        exchange = create_exchange()
        
        # Set leverage if provided
        if "leverage" in data:
            leverage = data["leverage"]
            try:
                exchange.set_leverage(leverage, symbol)
                log_webhook_request(data, "info", f"Leverage set to {leverage}x for {symbol}")
            except Exception as e:
                log_webhook_request(data, "warning", f"Could not set leverage: {str(e)}")

        # Process the main action (market order first)
        if action == "buy":
            order = exchange.create_market_buy_order(symbol, amount)
            log_webhook_request(data, "success", f"Buy order executed", order)
        elif action == "sell":
            order = exchange.create_market_sell_order(symbol, amount)
            log_webhook_request(data, "success", f"Sell order executed", order)
        else:
            error_msg = "Invalid action. Use 'buy' or 'sell'."
            log_webhook_request(data, "error", error_msg)
            return jsonify({"status": "error", "message": error_msg}), 400

        # Create position-level TP/SL orders with ROI% display
        tp_sl_orders = []
        
        if ("take_profit_percent" in data and data["take_profit_percent"]) or \
           ("stop_loss_percent" in data and data["stop_loss_percent"]):
            
            try:
                # Wait for position to be established
                time.sleep(2)
                
                # Get current price for TP/SL calculations
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Create position-level TP/SL using batch API
                position_tpsl = BinancePositionTPSL()
                batch_result = position_tpsl.create_position_tpsl_orders(
                    symbol=symbol,
                    action=action,
                    amount=amount,
                    current_price=current_price,
                    tp_percent=data.get("take_profit_percent"),
                    sl_percent=data.get("stop_loss_percent")
                )
                
                # Process batch results
                for batch_order in batch_result:
                    if 'orderId' in batch_order:
                        order_type = "take_profit" if batch_order['type'] == "TAKE_PROFIT_MARKET" else "stop_loss"
                        tp_sl_orders.append({
                            "type": order_type,
                            "order": batch_order,
                            "price": float(batch_order['stopPrice'])
                        })
                        
                        roi_type = "Take Profit" if order_type == "take_profit" else "Stop Loss"
                        roi_percent = data.get("take_profit_percent") if order_type == "take_profit" else data.get("stop_loss_percent")
                        log_webhook_request(data, "success", f"{roi_type} order created at ${batch_order['stopPrice']} ({roi_percent}% ROI)", batch_order)
                
            except Exception as e:
                log_webhook_request(data, "warning", f"Failed to create position-level TP/SL: {str(e)}")

        # Prepare response
        response_data = {
            "status": "success",
            "order": order,
            "leverage_used": data.get("leverage", "default"),
            "message": f"{action.upper()} order executed successfully",
            "tp_sl_orders": tp_sl_orders if tp_sl_orders else None,
            "tp_sl_info": {
                "take_profit_percent": data.get("take_profit_percent"),
                "stop_loss_percent": data.get("stop_loss_percent"),
                "position_level_tpsl": True,
                "roi_display_enabled": True,
                "binance_native_integration": True,
                "orders_created": len(tp_sl_orders)
            } if tp_sl_orders else None
        }

        return jsonify(response_data)
    except ExchangeError as e:
        error_msg = f"Exchange error: {str(e)}"
        log_webhook_request(request.json if request.is_json else {}, "error", error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500
    except Exception as e:
        error_msg = str(e)
        log_webhook_request(request.json if request.is_json else {}, "error", error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/logs')
def logs():
    try:
        logs_dir = ensure_logs_dir()
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(logs_dir, f"webhook_logs_{today}.txt")
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
            return f"<pre>{content}</pre>"
        else:
            return "No logs found for today."
    except Exception as e:
        return f"Error reading logs: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

