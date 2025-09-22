"""
Trading Bot - A Flask webhook server for automated trading with position-level TP/SL functionality
"""

import os
import datetime
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from ccxt import ExchangeError

# Import our custom modules
from config import config
from binance_api import create_exchange, BinancePositionTPSL
from utils import log_webhook_request, get_app_info, read_logs, ensure_logs_dir

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Routes
@app.route('/')
def home():
    """Home endpoint with application information"""
    return jsonify(get_app_info())

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
    """Endpoint to view today's logs"""
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
    port = int(os.environ.get('PORT', config.PORT))
    debug_mode = os.environ.get('DEBUG', str(config.DEBUG)).lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

