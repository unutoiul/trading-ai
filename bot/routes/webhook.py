from flask import Blueprint, request, jsonify
from ccxt import ExchangeError
import os
import datetime
import json

# Create a Blueprint for the webhook route
webhook_bp = Blueprint('webhook', __name__)

# Ensure logs directory exists
def ensure_logs_dir():
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
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

def create_webhook_route(exchange):
    @webhook_bp.route('/webhook', methods=['POST'])
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

            # Process the action
            if data["action"] == "buy":
                order = exchange.create_market_buy_order(data["symbol"], data["amount"])
                log_webhook_request(data, "success", "Buy order executed", order)
            elif data["action"] == "sell":
                order = exchange.create_market_sell_order(data["symbol"], data["amount"])
                log_webhook_request(data, "success", "Sell order executed", order)
            else:
                error_msg = "Invalid action. Use 'buy' or 'sell'."
                log_webhook_request(data, "error", error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400

            # Return the order details
            return jsonify({"status": "success", "order": order})
        except ExchangeError as e:
            error_msg = f"Exchange error: {str(e)}"
            log_webhook_request(request.json if request.is_json else {}, "error", error_msg)
            return jsonify({"status": "error", "message": error_msg}), 500
        except Exception as e:
            error_msg = str(e)
            log_webhook_request(request.json if request.is_json else {}, "error", error_msg)
            return jsonify({"status": "error", "message": error_msg}), 500