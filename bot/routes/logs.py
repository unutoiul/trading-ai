from flask import Blueprint, render_template_string, request, redirect, url_for, flash
import os
import json
import datetime
import glob

# Create a Blueprint for the logs route
logs_bp = Blueprint('logs', __name__)

# Ensure logs directory exists
def ensure_logs_dir():
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir

def create_logs_route():
    @logs_bp.route('/logs')
    def view_logs():
        logs_dir = ensure_logs_dir()
        
        # Get all log files, sorted by date (newest first)
        log_files = sorted(glob.glob(os.path.join(logs_dir, "webhook_logs_*.txt")), reverse=True)
        
        # Get optional date parameter from query string
        date_param = request.args.get('date', '')
        if date_param:
            # Filter logs for specific date
            log_files = [f for f in log_files if date_param in f]
        
        logs = []
        
        # Limit to most recent X files to avoid performance issues
        for log_file in log_files[:5]:  # Display 5 most recent log files
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    content = f.read()
                    entries = content.strip().split("\n\n")
                    for entry in entries:
                        if entry.strip():
                            try:
                                log_entry = json.loads(entry)
                                logs.append(log_entry)
                            except json.JSONDecodeError:
                                continue
        
        # Sort logs by timestamp (newest first)
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Get unique dates from the log files for the date selector
        available_dates = sorted(set([os.path.basename(f).replace('webhook_logs_', '').replace('.txt', '') 
                              for f in glob.glob(os.path.join(logs_dir, "webhook_logs_*.txt"))]), reverse=True)
        
        # Create an HTML template to display the logs
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Webhook Logs</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .log-entry { border: 1px solid #ddd; margin-bottom: 10px; padding: 10px; border-radius: 5px; }
                .success { border-left: 5px solid green; }
                .error { border-left: 5px solid red; }
                .received { border-left: 5px solid blue; }
                .info { border-left: 5px solid orange; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
                .timestamp { color: #666; font-size: 0.8em; }
                .filter-form { margin-bottom: 20px; }
                .filter-form input, .filter-form button, .filter-form select { padding: 5px; margin-right: 10px; }
                .btn { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
                .btn-danger { background-color: #f44336; }
                .btn:hover { opacity: 0.8; }
                .clear-section { margin-top: 20px; border-top: 1px solid #ddd; padding-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Webhook Logs</h1>
            
            <div class="filter-form">
                <form action="/logs" method="GET">
                    <label for="date">Filter by date:</label>
                    <select id="date" name="date">
                        <option value="">All dates</option>
                        {% for date in available_dates %}
                            <option value="{{ date }}" {% if date == date_param %}selected{% endif %}>{{ date }}</option>
                        {% endfor %}
                    </select>
                    <button type="submit" class="btn">Filter</button>
                </form>
            </div>
            
            <div class="clear-section">
                <h3>Clear Logs</h3>
                <p>Use the buttons below to clear log files. This action cannot be undone.</p>
                
                <form action="/logs/clear" method="POST" onsubmit="return confirm('Are you sure you want to clear these logs? This action cannot be undone.');">
                    {% if date_param %}
                        <input type="hidden" name="date" value="{{ date_param }}">
                        <button type="submit" name="clear_type" value="selected_date" class="btn btn-danger">
                            Clear Logs for {{ date_param }}
                        </button>
                    {% endif %}
                    
                    <button type="submit" name="clear_type" value="all" class="btn btn-danger">
                        Clear All Logs
                    </button>
                </form>
            </div>
            
            <h2>Log Entries {% if date_param %}for {{ date_param }}{% endif %}</h2>
            
            {% if logs %}
                {% for log in logs %}
                    <div class="log-entry {{ log.status }}">
                        <div class="timestamp">{{ log.timestamp }}</div>
                        <h3>{{ log.status.upper() }}: {{ log.message }}</h3>
                        
                        <h4>Request Data:</h4>
                        <pre>{{ json.dumps(log.request_data, indent=2) }}</pre>
                        
                        {% if log.get('order') %}
                            <h4>Order Details:</h4>
                            <pre>{{ json.dumps(log.order, indent=2) }}</pre>
                        {% endif %}
                    </div>
                {% endfor %}
            {% else %}
                <p>No logs found.</p>
            {% endif %}
        </body>
        </html>
        """
        
        # Render the template with the logs data
        return render_template_string(
            html_template, 
            logs=logs, 
            date_param=date_param,
            available_dates=available_dates,
            json=json  # Pass json module to the template
        )
    
    @logs_bp.route('/logs/clear', methods=['POST'])
    def clear_logs():
        logs_dir = ensure_logs_dir()
        clear_type = request.form.get('clear_type', '')
        date_to_clear = request.form.get('date', '')
        
        if clear_type == 'all':
            # Clear all log files
            log_files = glob.glob(os.path.join(logs_dir, "webhook_logs_*.txt"))
            for file in log_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error removing log file {file}: {e}")
        
        elif clear_type == 'selected_date' and date_to_clear:
            # Clear logs for a specific date
            log_file = os.path.join(logs_dir, f"webhook_logs_{date_to_clear}.txt")
            if os.path.exists(log_file):
                try:
                    os.remove(log_file)
                except Exception as e:
                    print(f"Error removing log file {log_file}: {e}")
        
        # Redirect back to the logs page
        if date_to_clear and clear_type == 'selected_date':
            return redirect(url_for('logs.view_logs'))
        else:
            return redirect(url_for('logs.view_logs'))