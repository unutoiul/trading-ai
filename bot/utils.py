"""
Utility functions for the Trading Bot
Handles logging, directory management, and other helper functions
"""

import os
import json
import datetime
from config import config

def ensure_logs_dir():
    """Ensure logs directory exists"""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir

def log_webhook_request(data, status, message, order=None):
    """Function to log webhook requests"""
    logs_dir = ensure_logs_dir()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(logs_dir, f"webhook_logs_{datetime.datetime.now().strftime('%Y-%m-%d')}.txt")
    
    log_entry = {
        "timestamp": timestamp,
        "status": status,
        "request_data": data,
        "message": message,
        "environment": config.environment
    }
    
    if order:
        log_entry["order"] = order
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry, indent=2) + "\n\n")

def get_app_info():
    """Get application information for the home endpoint"""
    return {
        "status": f"{config.APP_NAME} is running!",
        "version": config.APP_VERSION,
        "features": [
            "Position-level TP/SL with ROI% display",
            "Binance Futures integration",
            "Batch order execution",
            "Comprehensive logging",
            "Environment-based configuration"
        ],
        "endpoints": {
            "webhook": "/webhook",
            "logs": "/logs"
        },
        "environment": config.environment
    }

def read_logs():
    """Read today's logs for the logs endpoint"""
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