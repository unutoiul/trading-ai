"""
Binance API module for position-level TP/SL management
Handles Binance Futures API calls for creating position-level TP/SL orders
"""

import ccxt
import time
import requests
import hmac
import hashlib
import json
from urllib.parse import urlencode
from config import config

def create_exchange():
    """Create exchange instance when needed to avoid startup hangs"""
    config.validate()  # Ensure API keys are available
    
    exchange_config = {
        "apiKey": config.BINANCE_API_KEY,
        "secret": config.BINANCE_API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,
            "testnet": config.BINANCE_TESTNET,
        }
    }
    
    # Only set custom URLs for testnet
    if config.BINANCE_TESTNET:
        exchange_config["urls"] = {
            "api": {
                "public": f"{config.BINANCE_TESTNET_BASE_URL}/fapi/v1",
                "private": f"{config.BINANCE_TESTNET_BASE_URL}/fapi/v1",
            }
        }
    
    return ccxt.binance(exchange_config)

class BinancePositionTPSL:
    """Binance Position-Level TP/SL Manager - Creates orders that show ROI% in UI"""
    
    def __init__(self):
        config.validate()  # Ensure API keys are available
        self.api_key = config.BINANCE_API_KEY
        self.secret_key = config.BINANCE_API_SECRET
        self.base_url = config.binance_base_url
    
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