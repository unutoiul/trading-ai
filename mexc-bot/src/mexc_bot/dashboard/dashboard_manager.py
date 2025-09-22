"""Web dashboard for monitoring and controlling the trading bot."""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn


class DashboardManager:
    """Manages the web dashboard for the trading bot."""
    
    def __init__(self, config: Dict[str, Any], bot_instance=None):
        """Initialize dashboard manager."""
        self.config = config
        self.bot = bot_instance
        self.logger = logging.getLogger(__name__)
        
        # Dashboard configuration
        self.host = config.get('dashboard', {}).get('host', '127.0.0.1')
        self.port = config.get('dashboard', {}).get('port', 8080)
        self.debug = config.get('dashboard', {}).get('debug', False)
        
        # FastAPI app
        self.app = FastAPI(title="MEXC Trading Bot Dashboard", version="1.0.0")
        
        # WebSocket connections
        self.active_connections: list[WebSocket] = []
        
        # Setup routes and middleware
        self.setup_routes()
        self.setup_websocket()
        
        # Dashboard state
        self.dashboard_data = {
            'bot_status': 'stopped',
            'account_balance': 0.0,
            'positions': [],
            'recent_trades': [],
            'performance': {},
            'risk_metrics': {},
            'last_update': datetime.now().isoformat()
        }
    
    def setup_routes(self) -> None:
        """Setup HTTP routes for the dashboard."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return self.render_dashboard()
        
        @self.app.get("/api/status")
        async def get_bot_status():
            """Get current bot status."""
            return await self.get_status_data()
        
        @self.app.post("/api/start")
        async def start_bot():
            """Start the trading bot."""
            return await self.start_bot_endpoint()
        
        @self.app.post("/api/stop")
        async def stop_bot():
            """Stop the trading bot."""
            return await self.stop_bot_endpoint()
        
        @self.app.post("/api/emergency-stop")
        async def emergency_stop():
            """Emergency stop all trading."""
            return await self.emergency_stop_endpoint()
        
        @self.app.get("/api/positions")
        async def get_positions():
            """Get current positions."""
            return await self.get_positions_data()
        
        @self.app.get("/api/trades")
        async def get_trades():
            """Get recent trades."""
            return await self.get_trades_data()
        
        @self.app.get("/api/performance")
        async def get_performance():
            """Get performance metrics."""
            return await self.get_performance_data()
        
        @self.app.get("/api/risk")
        async def get_risk():
            """Get risk metrics."""
            return await self.get_risk_data()
        
        @self.app.post("/api/close-position/{position_id}")
        async def close_position(position_id: str):
            """Close a specific position."""
            return await self.close_position_endpoint(position_id)
    
    def setup_websocket(self) -> None:
        """Setup WebSocket for real-time updates."""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_handler(websocket)
    
    async def websocket_handler(self, websocket: WebSocket) -> None:
        """Handle WebSocket connections."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Send periodic updates
                await websocket.receive_text()
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
            self.logger.info("🔌 WebSocket client disconnected")
    
    async def broadcast_update(self, data: Dict[str, Any]) -> None:
        """Broadcast updates to all connected WebSocket clients."""
        if not self.active_connections:
            return
        
        message = json.dumps(data)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    def render_dashboard(self) -> str:
        """Render the main dashboard HTML."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>MEXC Trading Bot Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .header h1 { margin: 0; }
        .status { display: inline-block; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
        .status.running { background: #27ae60; color: white; }
        .status.stopped { background: #e74c3c; color: white; }
        .status.error { background: #f39c12; color: white; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h3 { margin-top: 0; color: #2c3e50; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-value { font-weight: bold; }
        .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .btn-primary { background: #3498db; color: white; }
        .btn-success { background: #27ae60; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-warning { background: #f39c12; color: white; }
        .btn:hover { opacity: 0.8; }
        .table { width: 100%; border-collapse: collapse; }
        .table th, .table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .table th { background: #f8f9fa; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        #log { max-height: 300px; overflow-y: auto; background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 MEXC Trading Bot Dashboard</h1>
            <span id="bot-status" class="status stopped">STOPPED</span>
            <span style="float: right;">Balance: $<span id="balance">0.00</span></span>
        </div>
        
        <div class="grid">
            <!-- Controls -->
            <div class="card">
                <h3>🎮 Bot Controls</h3>
                <button id="start-btn" class="btn btn-success" onclick="startBot()">Start Bot</button>
                <button id="stop-btn" class="btn btn-warning" onclick="stopBot()">Stop Bot</button>
                <button id="emergency-btn" class="btn btn-danger" onclick="emergencyStop()">Emergency Stop</button>
                <div style="margin-top: 15px;">
                    <small>Last update: <span id="last-update">Never</span></small>
                </div>
            </div>
            
            <!-- Performance -->
            <div class="card">
                <h3>📊 Performance</h3>
                <div class="metric">
                    <span>Daily P&L:</span>
                    <span id="daily-pnl" class="metric-value">$0.00</span>
                </div>
                <div class="metric">
                    <span>Total P&L:</span>
                    <span id="total-pnl" class="metric-value">$0.00</span>
                </div>
                <div class="metric">
                    <span>Win Rate:</span>
                    <span id="win-rate" class="metric-value">0%</span>
                </div>
                <div class="metric">
                    <span>Total Trades:</span>
                    <span id="total-trades" class="metric-value">0</span>
                </div>
            </div>
            
            <!-- Risk Metrics -->
            <div class="card">
                <h3>🛡️ Risk Metrics</h3>
                <div class="metric">
                    <span>Risk Score:</span>
                    <span id="risk-score" class="metric-value">0%</span>
                </div>
                <div class="metric">
                    <span>Exposure:</span>
                    <span id="exposure" class="metric-value">$0.00</span>
                </div>
                <div class="metric">
                    <span>Open Positions:</span>
                    <span id="open-positions" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span>Unrealized P&L:</span>
                    <span id="unrealized-pnl" class="metric-value">$0.00</span>
                </div>
            </div>
        </div>
        
        <!-- Positions Table -->
        <div class="card" style="margin-top: 20px;">
            <h3>📈 Open Positions</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Size</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>P&L</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="positions-table">
                    <tr><td colspan="7" style="text-align: center;">No open positions</td></tr>
                </tbody>
            </table>
        </div>
        
        <!-- Activity Log -->
        <div class="card" style="margin-top: 20px;">
            <h3>📝 Activity Log</h3>
            <div id="log">Waiting for updates...</div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        // Control functions
        async function startBot() {
            const response = await fetch('/api/start', { method: 'POST' });
            const result = await response.json();
            addLog(`Start command: ${result.message}`);
        }
        
        async function stopBot() {
            const response = await fetch('/api/stop', { method: 'POST' });
            const result = await response.json();
            addLog(`Stop command: ${result.message}`);
        }
        
        async function emergencyStop() {
            if (confirm('Are you sure you want to trigger emergency stop?')) {
                const response = await fetch('/api/emergency-stop', { method: 'POST' });
                const result = await response.json();
                addLog(`Emergency stop: ${result.message}`);
            }
        }
        
        async function closePosition(positionId) {
            if (confirm(`Close position ${positionId}?`)) {
                const response = await fetch(`/api/close-position/${positionId}`, { method: 'POST' });
                const result = await response.json();
                addLog(`Close position: ${result.message}`);
            }
        }
        
        // Update dashboard
        function updateDashboard(data) {
            // Status
            const statusEl = document.getElementById('bot-status');
            statusEl.textContent = data.bot_status.toUpperCase();
            statusEl.className = `status ${data.bot_status}`;
            
            // Balance
            document.getElementById('balance').textContent = data.account_balance.toFixed(2);
            
            // Performance
            document.getElementById('daily-pnl').textContent = `$${data.performance.daily_pnl || 0}`;
            document.getElementById('total-pnl').textContent = `$${data.performance.total_pnl || 0}`;
            document.getElementById('win-rate').textContent = `${(data.performance.win_rate || 0) * 100}%`;
            document.getElementById('total-trades').textContent = data.performance.total_trades || 0;
            
            // Risk
            document.getElementById('risk-score').textContent = `${(data.risk_metrics.risk_score || 0) * 100}%`;
            document.getElementById('exposure').textContent = `$${data.risk_metrics.total_exposure || 0}`;
            document.getElementById('open-positions').textContent = data.risk_metrics.open_positions || 0;
            document.getElementById('unrealized-pnl').textContent = `$${data.risk_metrics.unrealized_pnl || 0}`;
            
            // Last update
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            
            // Update positions table
            updatePositionsTable(data.positions || []);
        }
        
        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positions-table');
            
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center;">No open positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(pos => `
                <tr>
                    <td>${pos.symbol}</td>
                    <td>${pos.side}</td>
                    <td>${pos.size}</td>
                    <td>$${pos.entry_price}</td>
                    <td>$${pos.current_price}</td>
                    <td class="${pos.pnl >= 0 ? 'positive' : 'negative'}">$${pos.pnl}</td>
                    <td><button class="btn btn-danger" onclick="closePosition('${pos.id}')">Close</button></td>
                </tr>
            `).join('');
        }
        
        function addLog(message) {
            const log = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            log.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            log.scrollTop = log.scrollHeight;
        }
        
        // Load initial data
        async function loadInitialData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                addLog(`Error loading data: ${error.message}`);
            }
        }
        
        // Load data on page load
        loadInitialData();
        
        // Periodic refresh
        setInterval(loadInitialData, 5000);
    </script>
</body>
</html>
        """
        return html_content
    
    async def start(self) -> None:
        """Start the dashboard server."""
        self.logger.info(f"🌐 Starting dashboard server on {self.host}:{self.port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info" if self.debug else "warning"
        )
        
        self.server = uvicorn.Server(config)
        await self.server.serve()
    
    async def stop(self) -> None:
        """Stop the dashboard server."""
        if hasattr(self, 'server'):
            self.logger.info("🛑 Stopping dashboard server")
            self.server.should_exit = True
    
    # API endpoint implementations
    async def get_status_data(self) -> Dict[str, Any]:
        """Get current status data."""
        if self.bot:
            # Get real data from bot
            return await self.bot.get_dashboard_data()
        else:
            # Return mock data for development
            return self.dashboard_data
    
    async def start_bot_endpoint(self) -> Dict[str, Any]:
        """Start bot endpoint."""
        if self.bot and hasattr(self.bot, 'start'):
            try:
                await self.bot.start()
                return {"success": True, "message": "Bot started successfully"}
            except Exception as e:
                return {"success": False, "message": f"Failed to start bot: {str(e)}"}
        return {"success": False, "message": "Bot instance not available"}
    
    async def stop_bot_endpoint(self) -> Dict[str, Any]:
        """Stop bot endpoint."""
        if self.bot and hasattr(self.bot, 'stop'):
            try:
                await self.bot.stop()
                return {"success": True, "message": "Bot stopped successfully"}
            except Exception as e:
                return {"success": False, "message": f"Failed to stop bot: {str(e)}"}
        return {"success": False, "message": "Bot instance not available"}
    
    async def emergency_stop_endpoint(self) -> Dict[str, Any]:
        """Emergency stop endpoint."""
        if self.bot and hasattr(self.bot, 'emergency_stop'):
            try:
                await self.bot.emergency_stop()
                return {"success": True, "message": "Emergency stop activated"}
            except Exception as e:
                return {"success": False, "message": f"Emergency stop failed: {str(e)}"}
        return {"success": False, "message": "Bot instance not available"}
    
    async def get_positions_data(self) -> Dict[str, Any]:
        """Get positions data."""
        if self.bot and hasattr(self.bot, 'get_positions'):
            return {"positions": await self.bot.get_positions()}
        return {"positions": []}
    
    async def get_trades_data(self) -> Dict[str, Any]:
        """Get trades data."""
        if self.bot and hasattr(self.bot, 'get_recent_trades'):
            return {"trades": await self.bot.get_recent_trades()}
        return {"trades": []}
    
    async def get_performance_data(self) -> Dict[str, Any]:
        """Get performance data."""
        if self.bot and hasattr(self.bot, 'get_performance'):
            return await self.bot.get_performance()
        return {"daily_pnl": 0, "total_pnl": 0, "win_rate": 0, "total_trades": 0}
    
    async def get_risk_data(self) -> Dict[str, Any]:
        """Get risk data."""
        if self.bot and hasattr(self.bot, 'get_risk_metrics'):
            return await self.bot.get_risk_metrics()
        return {"risk_score": 0, "total_exposure": 0, "open_positions": 0, "unrealized_pnl": 0}
    
    async def close_position_endpoint(self, position_id: str) -> Dict[str, Any]:
        """Close position endpoint."""
        if self.bot and hasattr(self.bot, 'close_position'):
            try:
                result = await self.bot.close_position(position_id)
                return {"success": True, "message": f"Position {position_id} closed", "result": result}
            except Exception as e:
                return {"success": False, "message": f"Failed to close position: {str(e)}"}
        return {"success": False, "message": "Bot instance not available"}
    
    async def update_dashboard_data(self, data: Dict[str, Any]) -> None:
        """Update dashboard data and broadcast to clients."""
        self.dashboard_data.update(data)
        self.dashboard_data['last_update'] = datetime.now().isoformat()
        
        # Broadcast to WebSocket clients
        await self.broadcast_update(self.dashboard_data)
