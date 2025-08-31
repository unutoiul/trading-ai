"""Web application for crypto data selection and analysis."""

import re
import traceback
from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory
import os
from datetime import datetime
import io
import json

import pandas as pd
from sklearn.pipeline import islice
from src.data_fetch import available_pairs as get_available_pairs
from src.data_fetch import fetch_data as fetch_crypto_data
import queue
import threading
import time
from dotenv import load_dotenv
from contextlib import redirect_stdout
import io

import sys

# Add debugging at the top of app.py
print(f"Python path: {sys.path}")
print(f"Python executable: {sys.executable}")


# Create global log queues for different operations
fetch_log_queue = queue.Queue()    # For data fetching operations
analysis_log_queue = queue.Queue() # For analysis operations
strategy_log_queue = queue.Queue() # For strategy building operations

load_dotenv()  # Load environment variables from .env file
print('ANTHROPIC_API_KEY: ',os.environ.get("ANTHROPIC_API_KEY"))

app = Flask(__name__, static_folder='results', template_folder='templates')

# Add these configuration lines
app.config['DATA_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
app.config['RESULTS_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Create directories if they don't exist
os.makedirs(app.config['RESULTS_DIR'], exist_ok=True)

app.config['FETCH_RESULTS'] = None  # Initialize the config variable

# Create a global log queue
log_queue = queue.Queue()

# Add after your existing log_queue
fetch_log_queue = queue.Queue()


def add_log(message):
    """Add a log message to the analysis log queue."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"{timestamp} - {message}"
    analysis_log_queue.put(log_entry)

def add_strategy_log(message):
    """Add a log message to the strategy builder log queue."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"{timestamp} - {message}"
    strategy_log_queue.put(log_entry)

def add_fetch_log(message):
    """Add a fetch log message to the queue."""
    fetch_log_queue.put(f"{time.strftime('%H:%M:%S')} - {message}")

@app.route('/')
def index():
    """Render the main page."""
    try:
        # Get available files for dropdowns
        btc_files = get_available_files(os.path.join(app.config['DATA_DIR'], 'btc'))
        alt_files = get_available_files(os.path.join(app.config['DATA_DIR'], 'altcoins'))
        
        return render_template('index.html', btcFiles=btc_files, altFiles=alt_files)
    except Exception as e:
        app.logger.error(f"Error in index route: {str(e)}")
        # Create empty lists if there's an error
        return render_template('index.html', btcFiles=[], altFiles=[])

@app.route('/available_pairs')
def available_pairs():
    """Get available trading pairs from Binance."""
    try:
        usdt_pairs = get_available_pairs()
        return jsonify({'pairs': usdt_pairs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    """Fetch historical data for selected trading pairs."""
    try:
        data = request.json
        pairs = data.get('pairs', [])
        start_date = datetime.fromisoformat(data.get('start_date').replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(data.get('end_date').replace('Z', '+00:00'))
        timeframe = data.get('timeframe', '1m')
        
        # Validate inputs
        if not pairs:
            return jsonify({'error': 'No pairs selected'}), 400
        
        add_fetch_log(f"Starting data fetch for {len(pairs)} pairs from {start_date} to {end_date}")
        
        # Create a function that will run in a separate thread
        def fetch_with_logs():

            # Create a custom stdout to capture prints
            class LoggingStream(io.StringIO):
                def write(self, text):
                    if text.strip():  # Only process non-empty lines
                        add_fetch_log(text.strip())
                    super().write(text)
            
            # Redirect stdout to our custom stream
            with redirect_stdout(LoggingStream()):
                try:
                    # Use the function from data_fetch.py
                    results = fetch_crypto_data(pairs, start_date, end_date, timeframe)
                    
                    # Store results for client to retrieve - THIS IS CRITICAL
                    app.config['FETCH_RESULTS'] = results
                    
                    # Final log
                    add_fetch_log(f"Completed fetching {len(results)} data files")
                    add_fetch_log("FETCH_COMPLETE")
                except Exception as e:
                    add_fetch_log(f"FETCH_ERROR: {str(e)}")
        
        # Start fetching in a separate thread to not block the response
        threading.Thread(target=fetch_with_logs).start()
        
        return jsonify({
            'status': 'started',
            'message': 'Data fetch started, connect to log stream for updates'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch_logs')
def fetch_logs():
    """Stream fetch logs using Server-Sent Events."""
    def generate():
        yield "data: Connected to fetch log stream\n\n"
        
        while True:
            try:
                # Non-blocking queue get with timeout
                message = fetch_log_queue.get(timeout=1)
                yield f"data: {message}\n\n"
            except queue.Empty:
                # Send a heartbeat to keep the connection open
                yield "data: heartbeat\n\n"
            except GeneratorExit:
                # Client disconnected
                break
    
    return Response(generate(), mimetype='text/event-stream')

# Add a new endpoint to check fetch status and get results
@app.route('/fetch_status')
def fetch_status():
    """Get the status and results of the latest fetch operation."""
    results = app.config.get('FETCH_RESULTS')
    print(f"Fetch status requested, results: {results is not None}")  # Debug line
    
    if results is not None:  # Changed from "if results:" to handle empty lists
        return jsonify({
            'status': 'complete',
            'results': results
        })
    else:
        return jsonify({
            'status': 'not_found',
            'message': 'No completed fetch found'
        }), 404

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download a data file."""
    try:
        # Make sure to strip any path components for security
        safe_filename = os.path.basename(filename)
        filepath = os.path.join('data', safe_filename)
        
        print(f"Download requested for: {filepath}")  # Debug
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        print(f"Download error: {str(e)}")  # Debug
        return jsonify({'error': str(e)}), 500

import sys
import io

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    try:
        data = request.json
        btc_file = data.get('btc_file')
        alt_file = data.get('alt_file')
        
        # Get condition filters if provided, otherwise use all available conditions
        conditions = data.get('conditions')
        condition_type = data.get('condition_type', 'any')
        
        # Add initial log
        add_log(f"Starting directional analysis with {btc_file} and {alt_file}")
        add_log("Running only directional impact analysis and strategy generation")
        
        if conditions:
            add_log(f"Applying {len(conditions)} condition filters ({condition_type})")
        else:
            add_log("Using all market conditions (no filters applied)")
        
        # Create logging stream for console output
        class LoggingStream(io.StringIO):
            def write(self, text):
                if text.strip():  # Only process non-empty lines
                    add_log(text.strip())
                
            def flush(self):
                pass
        
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Redirect to logging stream
        logging_stream = LoggingStream()
        sys.stdout = logging_stream
        sys.stderr = logging_stream
        
        try:
            # Import the run_analysis module
            from src import run_analysis as run_analysis_module
            
            # Call our function with condition filters
            result_dir = run_analysis_module.run_directional_analysis_only(
                btc_file=btc_file,
                alt_file=alt_file,
                conditions=conditions,
                condition_type=condition_type
            )
            
            # Set the last result directory
            app.config['LAST_RESULT_DIR'] = result_dir
            add_log("ANALYSIS_COMPLETE")
            
            return jsonify({'status': 'complete', 'result_dir': result_dir})
        
        finally:
            # Always restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    except Exception as e:
        add_log(f"ANALYSIS_ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analysis_logs')
def analysis_logs():
    """Stream analysis logs using Server-Sent Events."""
    def generate():
        yield "data: Connected to analysis log stream\n\n"
        heartbeat_counter = 0
        
        while True:
            try:
                # Timeout to 1 second
                message = analysis_log_queue.get(timeout=1)
                yield f"data: {message}\n\n"
            except queue.Empty:
                # Only send heartbeat every 10 cycles (once per second)
                heartbeat_counter += 1
                if (heartbeat_counter >= 10):
                    yield "data: heartbeat\n\n"
                    heartbeat_counter = 0
            except GeneratorExit:
                break
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/analysis_result')
def analysis_result():
    """Get the result of the latest analysis."""
    result_dir = app.config.get('LAST_RESULT_DIR')
    if result_dir:
        return jsonify({
            'status': 'complete',
            'result_dir': result_dir
        })
    else:
        return jsonify({
            'status': 'not_found',
            'message': 'No completed analysis found'
        }), 404

@app.route('/list_data_files')
def list_data_files():
    """List all available data files in the data directory."""
    try:
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
            return jsonify({
                'status': 'success',
                'files': []  # Empty array if directory was just created
            })
            
        files = []
        
        # Check if the directory has any files
        if not os.listdir('data'):
            return jsonify({
                'status': 'success',
                'files': []  # Empty array if directory is empty
            })
            
        for filename in os.listdir('data'):
            if filename.endswith('.csv'):
                filepath = os.path.join('data', filename)
                
                # Try to determine the symbol from filename
                try:
                    if 'BTC_USDT' in filename:
                        symbol = 'BTC/USDT'
                    elif 'DOGE_USDT' in filename:
                        symbol = 'DOGE/USDT'
                    else:
                        # Generic parsing for other files
                        parts = filename.split('_')
                        if len(parts) >= 2 and parts[1].upper() == 'USDT':
                            symbol = f"{parts[0].upper()}/USDT"
                        else:
                            symbol = filename  # Use filename as fallback
                except:
                    symbol = filename  # Use filename as fallback
                
                # Get file metadata
                size_bytes = os.path.getsize(filepath)
                size_mb = size_bytes / (1024 * 1024)
                create_time = os.path.getctime(filepath)
                create_date = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')
                
                files.append({
                    'symbol': symbol,
                    'filename': filename,
                    'filepath': filepath,
                    'size': f"{size_mb:.2f} MB",
                    'created': create_date,
                    'display_name': f"{symbol} ({create_date})"
                })
        
        return jsonify({
            'status': 'success',
            'files': files
        })
    except Exception as e:
        print(f"Error listing data files: {str(e)}")  # Log the error server-side
        return jsonify({'error': str(e)}), 500

@app.route('/delete_file', methods=['POST'])
def delete_file():
    """Delete a data file."""
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        # Security check - make sure the file is in the data directory
        filepath = os.path.join('data', os.path.basename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        # Delete the file
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'message': f'File {filename} deleted successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add these routes after your existing view-results function
@app.route('/results/')
@app.route('/results/index.html')
def serve_results_index():
    """Serve the results index.html file."""
    try:
        # Check if the results directory exists
        if not os.path.exists('results'):
            os.makedirs('results')
            return render_template('no_results.html', 
                                  message="Results directory doesn't exist yet. Run an analysis first.")
        
        # Check if index.html exists in the results directory
        if not os.path.exists(os.path.join('results', 'index.html')):
            return render_template('no_results.html',
                                message="Results index.html doesn't exist yet. Run an analysis first.")
        
        # Serve the file directly
        return send_from_directory('results', 'index.html')
    except Exception as e:
        return render_template('no_results.html',
                              message=f"Error loading results: {str(e)}")

# Add this route right after your serve_results_index function
@app.route('/results/<path:filename>')
def serve_results_file(filename):
    """Serve any file from the results directory."""
    try:
        # Check if the file exists
        if not os.path.exists(os.path.join('results', filename)):
            return render_template('no_results.html',
                                  message=f"File not found: {filename}"), 404
        
        # Serve the file from the results directory
        return send_from_directory('results', filename)
    except Exception as e:
        return render_template('no_results.html',
                              message=f"Error loading file: {str(e)}"), 500

@app.route('/download_strategy/<path:filename>')
def download_strategy(filename):
    """Download a generated Pine Script strategy."""
    try:
        return send_from_directory('results/strategies', filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Error downloading strategy: {str(e)}'}), 404

@app.route('/list_reports')
def list_reports():
    """List all available analysis reports."""
    try:
        reports = []
        
        # Check if the results directory exists
        if not os.path.exists('results'):
            return jsonify({'reports': []})
        
        # Look for report files in result directories
        for result_dir in os.listdir('results'):
            report_dir = os.path.join('results', result_dir, 'reports')
            if os.path.exists(report_dir):
                for file in os.listdir(report_dir):
                    if file.endswith('_report.txt'):
                        reports.append({
                            'path': os.path.join(result_dir, 'reports', file),
                            'name': f"{result_dir}: {file}",
                            'date': os.path.getmtime(os.path.join(report_dir, file))
                        })
        
        # Sort by date (newest first)
        reports.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({'reports': reports})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_strategy_from_params')
def generate_strategy_from_params():
    """Generate a Pine Script strategy from saved optimization parameters."""
    try:
        result_dir = request.args.get('result_dir')
        if not result_dir:
            return jsonify({'error': 'No result directory specified'}), 400
            
        # Validate the path (security check)
        if '..' in result_dir or '/' in result_dir:
            return jsonify({'error': 'Invalid result directory'}), 400
            
        # Path to strategy parameters
        params_path = os.path.join('results', result_dir, 'reports', 'strategy_params.json')
        if not os.path.exists(params_path):
            return jsonify({'error': 'Strategy parameters not found'}), 404
            
        # Load strategy parameters
        with open(params_path, 'r') as f:
            strategy_data = json.load(f)
            
        # Generate the Pine Script
        pine_script = generate_optimized_pine_script(strategy_data)
        
        # Set filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimized_strategy_{timestamp}.pine"
        
        # Create response with file download
        response = Response(pine_script)
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        response.headers['Content-Type'] = 'text/plain'
        
        return response
        
    except Exception as e:
        return jsonify({'error': f'Error generating strategy: {str(e)}'}), 500
        
def generate_optimized_pine_script(strategy_data):
    """Generate Pine Script from optimized strategy parameters."""
    best_params = strategy_data.get('best_params', {})
    performance = strategy_data.get('performance_summary', {})

    script = f"""// @version=5
// Optimized BTC-Altcoin Strategy
// Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// 
// PERFORMANCE METRICS:
// Win Rate: {performance.get('win_rate', 0)*100:.1f}%
// Profit Factor: {performance.get('profit_factor', 0):.2f}
// Total Return: {performance.get('total_return_pct', 0):.2f}%
// Max Drawdown: {performance.get('max_drawdown', 0):.2f}%
// Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}

strategy("ML Optimized BTC-Altcoin Strategy", overlay=true)

// === Input Parameters ===
// These parameters were optimized by machine learning
stopLossPct = input.float({best_params.get('stop_loss_pct', 2.0)}, "Stop Loss %", minval=0.1, maxval=20)
takeProfitPct = input.float({best_params.get('take_profit_pct', 5.0)}, "Take Profit %", minval=0.1, maxval=50)
patternLag = input.int({best_params.get('pattern_lag', 5)}, "Pattern Signal Lag", minval=1, maxval=20)
positionSizePct = input.float({best_params.get('position_size_pct', 10.0)}, "Position Size %", minval=1, maxval=100)

// === Get BTC Data ===
// We need BTC data to detect patterns
btcSymbol = input.symbol("BTCUSDT", "BTC Symbol")
useCurrentTimeframe = input.bool(true, "Use Current Timeframe")
btcTimeframe = useCurrentTimeframe ? timeframe.period : input.timeframe("15", "BTC Timeframe")

[btcOpen, btcHigh, btcLow, btcClose] = request.security(btcSymbol, btcTimeframe, [open, high, low, close])
btcVol = request.security(btcSymbol, btcTimeframe, volume)

// === Pattern Detection ===
// Detecting the "{best_params.get('use_pattern', 'btc_bullish_momentum')}" pattern
btcReturns = (btcClose - btcClose[1]) / btcClose[1] * 100
btcVolRatio = btcVol / ta.sma(btcVol, 20)

// Pattern definitions
btcBullishMomentum = btcReturns > 0.8 and btcClose > btcOpen and btcVolRatio > 1.2
btcBearishMomentum = btcReturns < -0.8 and btcClose < btcOpen and btcVolRatio > 1.2
btcSidewaysAction = math.abs(btcReturns) < 0.3 and btcVolRatio < 0.8
btcBreakout = btcReturns > 1.5 and btcClose > btcClose[1] and btcVolRatio > 1.5

// === Strategy Logic ===
// Entry condition based on optimized pattern and lag
usePattern = "{best_params.get('use_pattern', 'btc_bullish_momentum')}"
patternSignal = false

if (usePattern == "strong_up")
    patternSignal := btcBullishMomentum
else if (usePattern == "strong_down")
    patternSignal := btcBearishMomentum
else if (usePattern == "steady_climb" or usePattern == "steady_decline")
    patternSignal := btcSidewaysAction
else if (usePattern == "volatility_breakout")
    patternSignal := btcBreakout
else
    patternSignal := btcBullishMomentum // default

// Signal with lag
entryCondition = patternSignal[patternLag]

// === Position Management ===
if (entryCondition and strategy.position_size == 0)
    strategy.entry("Long", strategy.long, qty=strategy.equity * positionSizePct / 100 / close)
    
    if (stopLossPct > 0)
        strategy.exit("SL/TP", "Long", 
            stop=strategy.position_avg_price * (1 - stopLossPct/100), 
            limit=strategy.position_avg_price * (1 + takeProfitPct/100))

// === Plot Signals ===
plotshape(entryCondition, title="Entry Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)

// === Strategy Notes ===
// This strategy was optimized using machine learning
// It detects specific Bitcoin patterns that precede altcoin moves
// The parameters have been optimized for maximum returns with acceptable risk
// Always monitor the market and adjust parameters if needed
"""
    
    return script

def get_available_files(directory):
    """Get available data files with date ranges."""
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract symbol from filename
            symbol = os.path.splitext(filename)[0]
            filepath = os.path.join(directory, filename)
            
            try:
                # Read first and last rows to get date range
                df = pd.read_csv(filepath)
                if len(df) > 0:
                    # Format min and max timestamps with proper dates
                    first_date = pd.to_datetime(df['timestamp'].min()).strftime('%Y-%m-%d %H:%M')
                    last_date = pd.to_datetime(df['timestamp'].max()).strftime('%Y-%m-%d %H:%M')
                    
                    # Create display name with full date range
                    display_name = f"{symbol} ({first_date} to {last_date})"
                    
                    # Calculate file size in MB
                    size = f"{os.path.getsize(filepath)/1024/1024:.1f} MB"
                    
                    files.append({
                        "filename": filename,
                        "filepath": filepath,
                        "symbol": symbol,
                        "display_name": display_name,
                        "size": size,
                        "first_date": first_date,
                        "last_date": last_date
                    })
            except Exception as e:
                # Fall back to simple format without dates
                files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "symbol": symbol,
                    "display_name": symbol
                })
    
    return files

# Helper function to read file from bottom
def reversed_lines(file):
    """Generate file lines in reverse order."""
    part = ''
    for block in reversed_blocks(file):
        for c in reversed(block):
            if c == '\n' and part:
                yield part[::-1]
                part = ''
            part += c
    if part:
        yield part[::-1]

def reversed_blocks(file, blocksize=4096):
    """Generate blocks of file's contents in reverse order."""
    file.seek(0, os.SEEK_END)
    here = file.tell()
    while (here > 0):
        delta = min(blocksize, here)
        here -= delta
        file.seek(here, os.SEEK_SET)
        yield file.read(delta)

@app.route('/get_available_files', methods=['GET'])
def available_files():
    """Get available data files for the UI."""
    file_type = request.args.get('type', 'btc')
    
    if file_type == 'btc':
        directory = os.path.join(app.config['DATA_DIR'], 'btc')
    else:
        directory = os.path.join(app.config['DATA_DIR'], 'altcoins')
    
    files = get_available_files(directory)
    return jsonify(files)

@app.route('/available_conditions', methods=['GET'])
def available_conditions():
    """Get available condition filters for the current dataset."""
    try:
        # Load the most recent analysis results
        result_dirs = [d for d in os.listdir(app.config['RESULTS_DIR']) if os.path.isdir(os.path.join(app.config['RESULTS_DIR'], d))]
        if not result_dirs:
            return jsonify({'error': 'No analysis results found'}), 404
            
        # Sort by creation time (newest first)
        result_dirs.sort(key=lambda d: os.path.getctime(os.path.join(app.config['RESULTS_DIR'], d)), reverse=True)
        latest_dir = result_dirs[0]
        
        # Look for filter config
        filter_config_path = os.path.join(app.config['RESULTS_DIR'], latest_dir, 'reports', 'filter_config.json')
        if os.path.exists(filter_config_path):
            with open(filter_config_path, 'r') as f:
                filter_config = json.load(f)
                
            return jsonify({
                'status': 'success',
                'conditions': filter_config.get('conditions', []),
                'condition_type': filter_config.get('condition_type', 'any')
            })
        
        # If no filter config found, return standard conditions
        return jsonify({
            'status': 'success',
            'conditions': [
                'btc_strong_up', 'btc_medium_up', 'btc_small_up',
                'btc_strong_down', 'btc_medium_down', 'btc_small_down',
                'btc_sideways', 'btc_high_volatility', 'btc_low_volatility'
            ],
            'condition_type': 'any'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_conditions', methods=['POST'])
def apply_conditions():
    """Apply condition filters to the analysis."""
    try:
        data = request.json
        conditions = data.get('conditions', [])
        condition_type = data.get('condition_type', 'any')
        
        # Get latest BTC and altcoin files
        btc_file = None
        alt_file = None
        
        # Scan data directory for files
        for file in os.listdir('data'):
            if file.lower().startswith('btc') and file.endswith('.csv'):
                btc_file = os.path.join('data', file)
            elif not file.lower().startswith('btc') and file.endswith('.csv'):
                alt_file = os.path.join('data', file)
        
        if not btc_file or not alt_file:
            return jsonify({'error': 'BTC or altcoin file not found'}), 400
            
        # Create logging stream to capture output
        class LoggingStream(io.StringIO):
            def write(self, text):
                if text.strip():  # Only process non-empty lines
                    add_log(text.strip())
                
            def flush(self):
                pass
        
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Redirect to logging stream
        logging_stream = LoggingStream()
        sys.stdout = logging_stream
        sys.stderr = logging_stream
        
        try:
            # Import the run_analysis module
            from src import run_analysis as run_analysis_module
            
            # Call main with the conditions
            result_dir = run_analysis_module.main(
                btc_file=btc_file,
                alt_file=alt_file,
                use_ml=True,
                optimize_strategy=True,
                conditions=conditions,
                condition_type=condition_type,
                return_results_dir=True
            )
            
            # Set the last result directory
            app.config['LAST_RESULT_DIR'] = result_dir
            add_log("ANALYSIS_COMPLETE")
            
            return jsonify({'status': 'complete', 'result_dir': result_dir})
            
        finally:
            # Always restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    except Exception as e:
        add_log(f"ANALYSIS_ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)