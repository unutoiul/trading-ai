"""Simple web server to host the analysis results."""

import os
import http.server
import socketserver
import webbrowser
import threading
import socket

def get_free_port():
    """Find a free port to use for the server."""
    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]

def start_server(results_dir, port=None):
    """Start a simple HTTP server to host the results."""
    if port is None:
        port = get_free_port()
    
    # Change to the results directory
    os.chdir(results_dir)
    
    handler = http.server.SimpleHTTPRequestHandler
    
    class QuietHTTPRequestHandler(handler):
        def log_message(self, format, *args):
            # Suppress log messages
            pass
    
    # Create the server
    with socketserver.TCPServer(("", port), QuietHTTPRequestHandler) as httpd:
        url = f"http://localhost:{port}"
        print(f"Server started at {url}")
        print(f"View analysis results at {url}/index.html")
        
        # Open browser automatically
        threading.Timer(1.0, lambda: webbrowser.open(url + "/index.html")).start()
        
        try:
            # Serve until interrupted
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")

def run_server_in_background(results_dir):
    """Run the server in a background thread."""
    port = get_free_port()
    server_thread = threading.Thread(
        target=start_server, 
        args=(results_dir, port),
        daemon=True  # This makes the thread exit when the main program exits
    )
    server_thread.start()
    return port