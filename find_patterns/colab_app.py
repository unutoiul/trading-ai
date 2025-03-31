from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory
import os
from datetime import datetime
import io
import json
from find_patterns.src.data_fetch import available_pairs as get_available_pairs
from find_patterns.src.data_fetch import fetch_data as fetch_crypto_data
import queue
import threading
import time
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Set your API key programmatically since we can't rely on .env in Colab
os.environ["ANTHROPIC_API_KEY"] = "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"
print('ANTHROPIC_API_KEY: ', os.environ.get("ANTHROPIC_API_KEY"))

app = Flask(__name__, 
            static_folder='find_patterns/results', 
            template_folder='find_patterns/templates')
app.config['FETCH_RESULTS'] = None  # Initialize the config variable

# Create a global log queue
log_queue = queue.Queue()
fetch_log_queue = queue.Queue()

def add_log(message):
    """Add a log message to the queue."""
    log_queue.put(f"{time.strftime('%H:%M:%S')} - {message}")

def add_fetch_log(message):
    """Add a fetch log message to the queue."""
    fetch_log_queue.put(f"{time.strftime('%H:%M:%S')} - {message}")

# Paste the rest of your app.py code here
# Make sure to change any file path references to include find_patterns/
# ...

# Change the last line to:
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)