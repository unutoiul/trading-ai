"""Web application for crypto data selection and analysis."""

from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory
import os
from datetime import datetime
import io
import json
from src.data_fetch import available_pairs as get_available_pairs
from src.data_fetch import fetch_data as fetch_crypto_data
import queue
import threading
import time
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
print('ANTHROPIC_API_KEY: ',os.environ.get("ANTHROPIC_API_KEY"))

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['FETCH_RESULTS'] = None  # Initialize the config variable

# Create a global log queue
log_queue = queue.Queue()

# Add after your existing log_queue
fetch_log_queue = queue.Queue()

def add_log(message):
    """Add a log message to the queue."""
    log_queue.put(f"{time.strftime('%H:%M:%S')} - {message}")

def add_fetch_log(message):
    """Add a fetch log message to the queue."""
    fetch_log_queue.put(f"{time.strftime('%H:%M:%S')} - {message}")

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

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
            from contextlib import redirect_stdout
            import io
            
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

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Run pattern analysis on selected pairs."""
    try:
        data = request.json
        btc_file = data.get('btc_file')
        alt_file = data.get('alt_file')
        
        if not btc_file or not alt_file:
            return jsonify({'error': 'Both BTC and altcoin files are required'}), 400
        
        # Add initial log
        add_log(f"Starting analysis with {btc_file} and {alt_file}")
        
        # Create a wrapper function to capture logs
        def run_with_logging():
            import sys
            from src.run_analysis import main
            import io
            from contextlib import redirect_stdout
            
            # Create a custom stdout to capture prints
            class LoggingStream(io.StringIO):
                def write(self, text):
                    if text.strip():  # Only process non-empty lines
                        add_log(text.strip())
                    super().write(text)
            
            # Redirect stdout to our custom stream
            with redirect_stdout(LoggingStream()):
                try:
                    # Override sys.argvf
                    sys.argv = ['run.py', '--btc', btc_file, '--doge', alt_file, '--use-ml', '--optimize-strategy']
                    
                    # Run analysis
                    result_dir = main(return_results_dir=True)
                    
                    # Final log
                    add_log(f"Analysis complete! Results in {result_dir}")
                    
                    return result_dir
                except Exception as e:
                    add_log(f"Error: {str(e)}")
                    raise
        
        # Start analysis in a separate thread to not block the response
        def analysis_thread():
            try:
                result_dir = run_with_logging()
                # Store the result for the client to fetch later
                app.config['LAST_RESULT_DIR'] = result_dir
                add_log("ANALYSIS_COMPLETE")
            except Exception as e:
                add_log(f"ANALYSIS_ERROR: {str(e)}")
        
        threading.Thread(target=analysis_thread).start()
        
        return jsonify({
            'status': 'started',
            'message': 'Analysis started, connect to log stream for updates'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analysis_logs')
def analysis_logs():
    """Stream analysis logs using Server-Sent Events."""
    def generate():
        yield "data: Connected to log stream\n\n"
        
        while True:
            try:
                # Non-blocking queue get with timeout
                message = log_queue.get(timeout=1)
                yield f"data: {message}\n\n"
            except queue.Empty:
                # Send a heartbeat to keep the connection open
                yield "data: heartbeat\n\n"
            except GeneratorExit:
                # Client disconnected
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

@app.route('/generate_strategy', methods=['POST'])
def generate_strategy():
    """Generate a Pine Script strategy based on selected report."""
    try:
        data = request.json
        
        # Extract parameters
        report_path = data.get('report_path')
        strategy_type = data.get('strategy_type', 'momentum')
        risk_per_trade = float(data.get('risk_per_trade', 1.0))
        use_stop_loss = data.get('use_stop_loss', True)
        stop_loss_percent = float(data.get('stop_loss_percent', 5.0))
        use_take_profit = data.get('use_take_profit', True)
        take_profit_percent = float(data.get('take_profit_percent', 10.0))
        
        # Check for Claude API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return jsonify({'error': 'Claude API key is required. Please set the ANTHROPIC_API_KEY environment variable.'}), 400
        
        # Check if report exists
        if not report_path:
            return jsonify({'error': 'Please select an analysis report.'}), 400
            
        full_report_path = os.path.join(app.root_path, 'results', report_path)
        if not os.path.exists(full_report_path):
            return jsonify({'error': f'Report file not found: {report_path}'}), 404
        
        # Read the report content
        with open(full_report_path, 'r') as f:
            report_content = f.read()
        
        # Use Claude AI to generate strategy
        from src.ai_analysis import ClaudeAnalyzer
        
        try:
            # Create strategies directory if it doesn't exist
            strategies_dir = os.path.join(app.root_path, 'results', 'strategies')
            if not os.path.exists(strategies_dir):
                os.makedirs(strategies_dir)
                
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate Pine Script from report
            analyzer = ClaudeAnalyzer()
            
            # Generate prompt for Claude based on the report
            prompt = f"""
# Generate Pine Script Strategy from Analysis Report

Below is an analysis report of Bitcoin-Dogecoin trading patterns. Use this report to create a Pine Script strategy for TradingView.

## Strategy Parameters:
- Strategy Type: {strategy_type}
- Risk Per Trade: {risk_per_trade}%
- Use Stop Loss: {'Yes' if use_stop_loss else 'No'}
- Stop Loss Percentage: {stop_loss_percent}%
- Use Take Profit: {'Yes' if use_take_profit else 'No'}
- Take Profit Percentage: {take_profit_percent}%

## Analysis Report:
{report_content}

## Instructions:
1. Create a Pine Script v5 strategy based on the patterns and insights in the report
2. Implement the specific strategy type ({strategy_type}) requested
3. Use the risk parameters provided
4. Focus on generating actionable trading signals based on BTC patterns affecting DOGE
5. Include appropriate comments explaining the strategy logic

Respond with just the complete Pine Script code in a code block.
"""
            
            # Call Claude API directly for this custom prompt
            response = analyzer.client.messages.create(
                model=analyzer.model,
                max_tokens=4000,
                temperature=0.1,
                system="You are an expert Pine Script developer who specializes in creating algorithmic trading strategies for TradingView. You write clean, optimized, and well-commented code that follows Pine Script best practices.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract code from response
            full_response = response.content[0].text
            
            # Try to extract just the Pine Script code between ```pine and ``` if it exists
            import re
            code_match = re.search(r'```pine\n(.*?)```', full_response, re.DOTALL)
            if code_match:
                pine_script = code_match.group(1)
            else:
                # If no code block, just use the whole response
                pine_script = full_response
            
            # Save the strategy to file
            strategy_filename = f"claude_strategy_{strategy_type}_{timestamp}.pine"
            strategy_path = os.path.join(strategies_dir, strategy_filename)
            with open(strategy_path, 'w') as f:
                f.write(pine_script)
                
            # Create a plain text version for viewing in browser
            view_filename = f"{strategy_filename}.txt"
            view_path = os.path.join(strategies_dir, view_filename)
            with open(view_path, 'w') as f:
                f.write(pine_script)
                
            return jsonify({
                'status': 'success',
                'message': 'Strategy generated successfully',
                'download_url': f'/results/strategies/{strategy_filename}',
                'view_url': f'/results/strategies/{view_filename}',
                'filename': strategy_filename
            })
            
        except Exception as e:
            return jsonify({'error': f"Error generating strategy: {str(e)}"}), 500
            
    except Exception as e:
        return jsonify({'error': f'Strategy generation failed: {str(e)}'}), 500

@app.route('/download_strategy/<path:filename>')
def download_strategy(filename):
    """Download a generated Pine Script strategy."""
    try:
        return send_from_directory('results/strategies', filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Error downloading strategy: {str(e)}'}), 404

@app.route('/check_claude_api')
def check_claude_api():
    """Check if Claude API key is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    return jsonify({
        'available': bool(api_key)
    })

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)