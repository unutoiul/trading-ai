"""
BTC-DOGE Momentum Pattern Analysis
---------------------------------
This script analyzes the lead-lag relationship between BTC and DOGE,
identifying momentum patterns in BTC that may predict DOGE price movements.

Usage:
    python run.py --btc [BTC_CSV_PATH] --doge [DOGE_CSV_PATH] --serve
    python run.py --use-ml  # Include XGBoost machine learning prediction
"""

import os
import sys
import argparse

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Run the analysis
from src.run_analysis import main

if __name__ == "__main__":
    main()