"""Use Claude 3.7 to analyze trading patterns and generate insights."""

import os
import json
import time
from datetime import datetime
import anthropic
import pandas as pd

class ClaudeAnalyzer:
    """Use Claude 3.7 to analyze trading patterns and generate insights."""
    
    def __init__(self, api_key=None):
        """Initialize Claude client with API key."""
        # Get API key from environment or parameter
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-3-7-sonnet"  # Use Claude 3.7 Sonnet
    
    def analyze_patterns(self, pattern_stats, ml_results=None, combined_data=None):
        """
        Send pattern analysis results to Claude 3.7 for deeper insights.
        
        Args:
            pattern_stats: Pattern analysis results
            ml_results: Optional ML analysis results
            combined_data: Optional DataFrame with combined data
            
        Returns:
            dict: Claude's analysis results
        """
        # Format data for Claude
        prompt = self._create_analysis_prompt(pattern_stats, ml_results, combined_data)
        
        # Call Claude API
        print("Sending data to Claude 3.7 for analysis...")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.0,  # Use deterministic output for consistency
                system="You are an expert cryptocurrency trading analyst who specializes in technical analysis, pattern recognition, and algorithmic trading strategy development.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and parse response
            analysis = response.content[0].text
            
            return {
                "raw_response": analysis,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "model": self.model
            }
        
        except Exception as e:
            print(f"Error calling Claude API: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_pine_script(self, pattern_stats, ml_results=None, **kwargs):
        """
        Generate a TradingView Pine Script based on pattern analysis.
        
        Args:
            pattern_stats: Pattern analysis results
            ml_results: Optional ML analysis results
            **kwargs: Additional strategy parameters
            
        Returns:
            str: Generated Pine Script
        """
        # Extract strategy parameters
        strategy_type = kwargs.get('strategy_type', 'momentum')
        risk_per_trade = kwargs.get('risk_per_trade', 1.0)
        use_stop_loss = kwargs.get('use_stop_loss', True)
        stop_loss_percent = kwargs.get('stop_loss_percent', 5.0)
        use_take_profit = kwargs.get('use_take_profit', True)
        take_profit_percent = kwargs.get('take_profit_percent', 10.0)
        
        prompt = self._create_pine_script_prompt(pattern_stats, ml_results, **kwargs)
        
        print("Requesting Pine Script strategy from Claude 3.7...")
        try:
            response = self.client.messages.create(
                model=self.model,
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
            
            return {
                "pine_script": pine_script,
                "full_response": full_response,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"Error generating Pine Script: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_analysis_prompt(self, pattern_stats, ml_results=None, combined_data=None):
        """Create a detailed prompt for pattern analysis."""
        prompt = """
# Trading Pattern Analysis Results

I need you to analyze these cryptocurrency pattern findings and provide strategic insights. The data shows how BTC price patterns affect altcoin (DOGE) price movements.

## Pattern Statistics Summary:
"""
        
        # Add pattern stats summary
        for pattern, lags in pattern_stats.items():
            if pattern.startswith('btc_pattern_'):
                pattern_name = pattern.replace('btc_pattern_', '')
                prompt += f"\n### {pattern_name.replace('_', ' ').title()}\n"
                
                # Add most significant lags
                sorted_lags = sorted(lags.items(), key=lambda x: abs(float(x[1]['mean_response'])), reverse=True)
                for lag, stats in sorted_lags[:3]:  # Top 3 most significant lags
                    prompt += f"- Lag {lag}: Mean Response: {stats['mean_response']:.4f}%, Confidence: {stats['confidence']:.2f}%\n"
        
        # Add ML results if available
        if ml_results and 'feature_importance' in ml_results:
            prompt += "\n## ML Feature Importance:\n"
            for feature, importance in ml_results['feature_importance'][:10]:  # Top 10 features
                prompt += f"- {feature}: {importance:.6f}\n"
            
            if 'metrics' in ml_results:
                prompt += f"\nModel Performance:\n"
                prompt += f"- Directional Accuracy: {ml_results['metrics']['directional_accuracy']:.2%}\n"
                prompt += f"- R² Score: {ml_results['metrics']['r2']:.4f}\n"
        
        # Add your requests for analysis
        prompt += """
## Analysis Requests:

1. **Key Pattern Insights**: What are the 3-5 most significant BTC patterns that affect DOGE, and how should traders interpret them?

2. **Trading Strategy Recommendations**: Based on these patterns, develop 2-3 concrete trading strategies for DOGE that leverage BTC movements. Include:
   - Entry and exit criteria
   - Optimal time frames
   - Risk management suggestions
   - Expected performance characteristics

3. **Pattern Implementation Guide**: How can traders actually identify these patterns in real-time? Provide specific indicators or visual cues.

4. **Risk Assessment**: What are the potential failure modes of these pattern-based strategies? When might these correlations break down?

5. **Optimization Suggestions**: How could we improve this analysis to get more reliable trading signals?

Keep your analysis focused on practical, actionable trading insights. Use clear, concise language that an intermediate trader would understand.
"""
        return prompt
    
    def _create_pine_script_prompt(self, pattern_stats, ml_results=None):
        """Create a prompt for Pine Script generation."""
        prompt = """
# Create a TradingView Pine Script Strategy

Based on our analysis of Bitcoin's impact on DOGE price movements, I need you to create a complete Pine Script strategy that implements these findings for TradingView.

## Key Pattern Findings:
"""
        
        # Add most significant patterns
        significant_patterns = {}
        for pattern, lags in pattern_stats.items():
            if pattern.startswith('btc_pattern_'):
                pattern_name = pattern.replace('btc_pattern_', '')
                
                # Find most significant lag
                most_sig_lag = max(lags.items(), key=lambda x: abs(float(x[1]['mean_response'])))
                lag_num = most_sig_lag[0]
                response = most_sig_lag[1]['mean_response']
                confidence = most_sig_lag[1]['confidence']
                
                significant_patterns[pattern_name] = {
                    'lag': lag_num,
                    'response': response,
                    'confidence': confidence
                }
        
        # Sort patterns by absolute response value
        sorted_patterns = sorted(significant_patterns.items(), 
                                key=lambda x: abs(x[1]['response']), 
                                reverse=True)
        
        # Add top patterns to prompt
        for pattern_name, stats in sorted_patterns[:5]:  # Top 5 patterns
            prompt += f"- Pattern: {pattern_name.replace('_', ' ').title()}\n"
            prompt += f"  - Best lag: {stats['lag']} bars\n"
            prompt += f"  - Expected DOGE response: {stats['response']:.4f}%\n"
            prompt += f"  - Confidence: {stats['confidence']:.2f}%\n\n"
        
        # Add ML feature importance if available
        if ml_results and 'feature_importance' in ml_results:
            prompt += "\n## ML Feature Importance (Top 5):\n"
            for feature, importance in ml_results['feature_importance'][:5]:
                prompt += f"- {feature}: {importance:.6f}\n"
        
        prompt += """
## Pine Script Requirements:

1. Create a version 5 Pine Script strategy that trades DOGE based on BTC patterns
2. Include the following components:
   - Use request.security() to get BTC price data
   - Implement detection functions for the key patterns identified above
   - Create entry signals based on the most significant patterns with their optimal lag
   - Implement appropriate exit conditions
   - Add risk management with stop loss and take profit
   - Include position sizing based on risk percentage
   - Add appropriate strategy settings via inputs

3. Make the strategy flexible with customizable:
   - Risk per trade (default 1%)
   - Stop loss percentage (appropriate for DOGE volatility)
   - Take profit targets
   - Maximum concurrent positions
   
4. Add relevant comments explaining the logic and how it implements our pattern findings

5. The script should be well-organized, efficient, and ready to load into TradingView.

Respond with just the complete Pine Script code in a code block.
"""
        return prompt

def generate_claude_reports(pattern_stats, ml_results, output_dirs):
    """Generate Claude 3.7 analysis reports."""
    try:
        analyzer = ClaudeAnalyzer()
        
        # Generate pattern analysis from Claude
        analysis_results = analyzer.analyze_patterns(pattern_stats, ml_results)
        
        # Save raw response
        report_path = os.path.join(output_dirs['reports'], 'claude_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write("CLAUDE 3.7 TRADING ANALYSIS\n")
            f.write("==========================\n\n")
            if 'error' in analysis_results:
                f.write(f"ERROR: {analysis_results['error']}\n")
            else:
                f.write(analysis_results['raw_response'])
        
        # Generate Pine Script
        pine_results = analyzer.generate_pine_script(pattern_stats, ml_results)
        
        # Save Pine Script
        script_path = os.path.join(output_dirs['reports'], 'claude_pine_script_strategy.pine')
        if 'error' in pine_results:
            with open(script_path, 'w') as f:
                f.write(f"// ERROR: {pine_results['error']}\n")
        else:
            with open(script_path, 'w') as f:
                f.write(pine_results['pine_script'])
        
        print(f"Claude analysis saved to {report_path}")
        print(f"Pine Script strategy saved to {script_path}")
        
        return {
            'analysis_report': report_path,
            'pine_script': script_path
        }
    
    except Exception as e:
        print(f"Error generating Claude reports: {str(e)}")
        error_path = os.path.join(output_dirs['reports'], 'claude_error_report.txt')
        with open(error_path, 'w') as f:
            f.write(f"ERROR GENERATING CLAUDE REPORTS: {str(e)}\n")
            f.write("\nPlease ensure you have set the ANTHROPIC_API_KEY environment variable")
            f.write("\nand have installed the anthropic package (pip install anthropic)")
        
        return {
            'error_report': error_path
        }