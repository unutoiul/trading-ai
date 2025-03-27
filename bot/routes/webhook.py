from flask import Blueprint, request, jsonify
from ccxt import ExchangeError

# Create a Blueprint for the webhook route
webhook_bp = Blueprint('webhook', __name__)

def create_webhook_route(exchange):
    @webhook_bp.route('/webhook', methods=['POST'])
    def webhook():
        try:
            # Ensure the request contains JSON data
            if not request.json or "action" not in request.json:
                return jsonify({"status": "error", "message": "Invalid request. 'action' key is required."}), 400

            data = request.json

            # Process the action
            if data["action"] == "buy":
                order = exchange.create_market_buy_order(data["symbol"], data["amount"])
            elif data["action"] == "sell":
                order = exchange.create_market_sell_order(data["symbol"], data["amount"])
            else:
                return jsonify({"status": "error", "message": "Invalid action. Use 'buy' or 'sell'."}), 400

            # Return the order details
            return jsonify({"status": "success", "order": order})
        except ExchangeError as e:
            return jsonify({"status": "error", "message": f"Exchange error: {str(e)}"}), 500
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500