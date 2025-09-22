"""Risk management for trading operations."""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RiskMetrics:
    """Risk metrics for monitoring."""
    total_exposure: float = 0.0
    used_margin: float = 0.0
    available_margin: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    open_positions: int = 0
    risk_score: float = 0.0


class RiskManager:
    """Manages trading risk and position sizing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize risk manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_total_exposure = config.get('max_total_exposure', 10000)  # USD
        self.max_positions = config.get('max_positions', 5)
        self.max_leverage = config.get('max_leverage', 20)
        self.max_daily_loss = config.get('max_daily_loss', 500)  # USD
        self.position_size_pct = config.get('position_size_pct', 0.02)  # 2% of account per trade
        
        # Emergency stop settings
        self.emergency_stop_loss = config.get('emergency_stop_loss', 0.10)  # 10% account loss
        self.max_drawdown = config.get('max_drawdown', 0.15)  # 15% max drawdown
        
        # Risk tracking
        self.daily_start_balance: float = 0.0
        self.peak_balance: float = 0.0
        self.risk_metrics = RiskMetrics()
        self.emergency_stop_triggered: bool = False
        
        # Position tracking
        self.position_history: List[Dict[str, Any]] = []
        self.risk_violations: List[Dict[str, Any]] = []
    
    async def initialize(self, initial_balance: float) -> None:
        """Initialize risk manager with account balance."""
        self.logger.info("🛡️ Initializing Risk Manager...")
        
        self.daily_start_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Calculate dynamic risk parameters based on account size
        self.adjust_risk_parameters(initial_balance)
        
        self.logger.info(f"✅ Risk Manager initialized - Account: ${initial_balance:.2f}")
        self.logger.info(f"📊 Max exposure: ${self.max_total_exposure:.2f}, Max positions: {self.max_positions}")
    
    def adjust_risk_parameters(self, account_balance: float) -> None:
        """Adjust risk parameters based on account size."""
        # Scale exposure based on account size
        base_exposure_ratio = 0.8  # Use 80% of account for trading
        self.max_total_exposure = account_balance * base_exposure_ratio
        
        # Scale daily loss limit
        daily_loss_ratio = 0.05  # Max 5% daily loss
        self.max_daily_loss = account_balance * daily_loss_ratio
        
        self.logger.info(f"📈 Risk parameters adjusted for ${account_balance:.2f} account")
    
    async def validate_new_position(self, signal: Dict[str, Any], current_positions: List[Dict[str, Any]], 
                                  account_balance: float) -> Dict[str, Any]:
        """Validate if a new position can be opened safely."""
        validation_result = {
            'allowed': False,
            'reason': '',
            'adjusted_quantity': 0.0,
            'adjusted_leverage': 1
        }
        
        try:
            # Check emergency stop
            if self.emergency_stop_triggered:
                validation_result['reason'] = 'Emergency stop active'
                return validation_result
            
            # Check maximum positions
            if len(current_positions) >= self.max_positions:
                validation_result['reason'] = f'Max positions reached ({self.max_positions})'
                return validation_result
            
            # Check daily loss limit
            current_daily_pnl = self.calculate_daily_pnl(account_balance)
            if current_daily_pnl <= -self.max_daily_loss:
                validation_result['reason'] = f'Daily loss limit reached (${abs(current_daily_pnl):.2f})'
                return validation_result
            
            # Calculate current exposure
            current_exposure = sum(abs(pos.get('notional', 0)) for pos in current_positions)
            
            # Calculate position size
            proposed_quantity = signal.get('quantity', 0)
            proposed_price = signal.get('price', 0)
            proposed_leverage = min(signal.get('leverage', 1), self.max_leverage)
            
            # Adjust position size based on risk limits
            max_position_value = account_balance * self.position_size_pct
            proposed_notional = proposed_quantity * proposed_price
            
            if proposed_notional > max_position_value:
                # Reduce position size
                adjusted_quantity = max_position_value / proposed_price
                validation_result['adjusted_quantity'] = round(adjusted_quantity, 6)
                proposed_notional = max_position_value
            else:
                validation_result['adjusted_quantity'] = proposed_quantity
            
            # Check total exposure limit
            total_exposure_after = current_exposure + proposed_notional
            if total_exposure_after > self.max_total_exposure:
                remaining_exposure = self.max_total_exposure - current_exposure
                if remaining_exposure > 0:
                    # Reduce position to fit within exposure limit
                    adjusted_quantity = remaining_exposure / proposed_price
                    validation_result['adjusted_quantity'] = round(adjusted_quantity, 6)
                else:
                    validation_result['reason'] = 'Total exposure limit reached'
                    return validation_result
            
            # Check leverage limits
            validation_result['adjusted_leverage'] = proposed_leverage
            
            # All checks passed
            validation_result['allowed'] = True
            validation_result['reason'] = 'Position validated'
            
            self.logger.info(f"✅ Position validated: {signal.get('symbol')} "
                           f"Qty: {validation_result['adjusted_quantity']:.6f}")
        
        except Exception as e:
            self.logger.error(f"❌ Error validating position: {e}")
            validation_result['reason'] = f'Validation error: {str(e)}'
        
        return validation_result
    
    async def check_emergency_conditions(self, account_balance: float, positions: List[Dict[str, Any]]) -> bool:
        """Check if emergency stop conditions are met."""
        try:
            # Check account drawdown
            if self.peak_balance > 0:
                current_drawdown = (self.peak_balance - account_balance) / self.peak_balance
                
                if current_drawdown >= self.max_drawdown:
                    self.logger.critical(f"🚨 EMERGENCY STOP: Max drawdown reached ({current_drawdown:.1%})")
                    await self.trigger_emergency_stop()
                    return True
            
            # Check emergency stop loss
            if self.daily_start_balance > 0:
                daily_loss_pct = (self.daily_start_balance - account_balance) / self.daily_start_balance
                
                if daily_loss_pct >= self.emergency_stop_loss:
                    self.logger.critical(f"🚨 EMERGENCY STOP: Daily loss limit reached ({daily_loss_pct:.1%})")
                    await self.trigger_emergency_stop()
                    return True
            
            # Update peak balance
            if account_balance > self.peak_balance:
                self.peak_balance = account_balance
        
        except Exception as e:
            self.logger.error(f"❌ Error checking emergency conditions: {e}")
        
        return self.emergency_stop_triggered
    
    async def trigger_emergency_stop(self) -> None:
        """Trigger emergency stop procedures."""
        if not self.emergency_stop_triggered:
            self.emergency_stop_triggered = True
            
            self.logger.critical("🚨 EMERGENCY STOP TRIGGERED")
            self.logger.critical("🚨 All new positions will be blocked")
            self.logger.critical("🚨 Consider closing existing positions")
            
            # Record the emergency stop event
            self.risk_violations.append({
                'type': 'emergency_stop',
                'timestamp': datetime.now().isoformat(),
                'reason': 'Risk limits exceeded'
            })
    
    def calculate_daily_pnl(self, current_balance: float) -> float:
        """Calculate daily P&L."""
        return current_balance - self.daily_start_balance
    
    def calculate_position_risk(self, position: Dict[str, Any]) -> float:
        """Calculate risk score for a position."""
        try:
            notional = abs(position.get('notional', 0))
            leverage = position.get('leverage', 1)
            unrealized_pnl = position.get('unrealized_pnl', 0)
            
            # Risk factors
            size_risk = min(notional / self.max_total_exposure, 1.0)
            leverage_risk = min(leverage / self.max_leverage, 1.0)
            pnl_risk = abs(unrealized_pnl) / notional if notional > 0 else 0
            
            # Combined risk score (0-1)
            risk_score = (size_risk * 0.4 + leverage_risk * 0.3 + pnl_risk * 0.3)
            
            return min(risk_score, 1.0)
        
        except Exception as e:
            self.logger.error(f"❌ Error calculating position risk: {e}")
            return 0.5  # Medium risk if calculation fails
    
    def update_risk_metrics(self, account_balance: float, positions: List[Dict[str, Any]]) -> RiskMetrics:
        """Update and return current risk metrics."""
        try:
            # Calculate metrics
            total_exposure = sum(abs(pos.get('notional', 0)) for pos in positions)
            unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
            daily_pnl = self.calculate_daily_pnl(account_balance)
            
            # Calculate overall risk score
            position_risks = [self.calculate_position_risk(pos) for pos in positions]
            avg_position_risk = sum(position_risks) / len(position_risks) if position_risks else 0
            
            exposure_risk = min(total_exposure / self.max_total_exposure, 1.0)
            overall_risk = (avg_position_risk * 0.6 + exposure_risk * 0.4)
            
            # Update metrics
            self.risk_metrics = RiskMetrics(
                total_exposure=total_exposure,
                used_margin=total_exposure,  # Simplified
                available_margin=max(0, self.max_total_exposure - total_exposure),
                unrealized_pnl=unrealized_pnl,
                daily_pnl=daily_pnl,
                open_positions=len(positions),
                risk_score=overall_risk
            )
        
        except Exception as e:
            self.logger.error(f"❌ Error updating risk metrics: {e}")
        
        return self.risk_metrics
    
    def should_reduce_exposure(self) -> bool:
        """Determine if exposure should be reduced."""
        return (self.risk_metrics.risk_score > 0.8 or
                self.risk_metrics.total_exposure > self.max_total_exposure * 0.9 or
                self.risk_metrics.daily_pnl < -self.max_daily_loss * 0.8)
    
    def reset_daily_tracking(self, current_balance: float) -> None:
        """Reset daily tracking metrics (call at start of new trading day)."""
        self.daily_start_balance = current_balance
        self.risk_metrics.daily_pnl = 0.0
        
        # Reset emergency stop if conditions improve
        if (self.emergency_stop_triggered and 
            current_balance > self.daily_start_balance * (1 - self.emergency_stop_loss / 2)):
            self.emergency_stop_triggered = False
            self.logger.info("✅ Emergency stop reset - conditions improved")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            'emergency_stop_active': self.emergency_stop_triggered,
            'risk_metrics': {
                'total_exposure': self.risk_metrics.total_exposure,
                'exposure_limit': self.max_total_exposure,
                'exposure_utilization': (self.risk_metrics.total_exposure / self.max_total_exposure) * 100,
                'daily_pnl': self.risk_metrics.daily_pnl,
                'daily_loss_limit': self.max_daily_loss,
                'open_positions': self.risk_metrics.open_positions,
                'max_positions': self.max_positions,
                'risk_score': self.risk_metrics.risk_score * 100,
                'unrealized_pnl': self.risk_metrics.unrealized_pnl
            },
            'risk_violations': self.risk_violations[-10:],  # Last 10 violations
            'recommendations': self.get_risk_recommendations()
        }
    
    def get_risk_recommendations(self) -> List[str]:
        """Get risk management recommendations."""
        recommendations = []
        
        if self.risk_metrics.risk_score > 0.8:
            recommendations.append("⚠️ High risk score - consider reducing position sizes")
        
        if self.risk_metrics.total_exposure > self.max_total_exposure * 0.9:
            recommendations.append("⚠️ High exposure - avoid opening new positions")
        
        if self.risk_metrics.daily_pnl < -self.max_daily_loss * 0.8:
            recommendations.append("⚠️ Approaching daily loss limit - be cautious")
        
        if self.risk_metrics.open_positions >= self.max_positions * 0.8:
            recommendations.append("⚠️ High number of positions - consider consolidation")
        
        if not recommendations:
            recommendations.append("✅ Risk levels are within acceptable ranges")
        
        return recommendations
