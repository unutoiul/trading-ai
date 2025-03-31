import streamlit as st
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import main function
from src.run_analysis import main

# Page setup
st.set_page_config(page_title="Trading AI")

# Simple title
st.title("BTC-Altcoin Pattern Analyzer")

# Basic sidebar with file selection
with st.sidebar:
    st.header("Select Data Files")
    
    # Get available CSV files
    files = [f for f in os.listdir('data') if f.endswith('.csv')]
    
    # Simple file selection
    btc_file = st.selectbox("BTC Data File", files)
    alt_file = st.selectbox("Altcoin Data File", files)
    
    # Basic options
    use_ml = st.checkbox("Use Machine Learning", value=True)
    optimize = st.checkbox("Optimize Strategy", value=True)

# Simple run button
if st.button("Run Analysis", type="primary"):
    # Show a status message
    status = st.empty()
    status.info("Running analysis... This may take a few minutes.")
    
    try:
        # Prepare command arguments
        sys.argv = ['run.py']
        sys.argv.extend(['--btc', os.path.join('data', btc_file)])
        sys.argv.extend(['--doge', os.path.join('data', alt_file)])
        
        if use_ml:
            sys.argv.append('--use-ml')
        
        if optimize:
            sys.argv.append('--optimize-strategy')
        
        # Run analysis
        result_dir = main(return_results_dir=True)
        
        # Show success message
        status.success(f"Analysis complete!")
        
        # Simply show link to the results folder
        st.markdown(f"### Results")
        st.markdown(f"Results saved to: `{result_dir}`")
        
        # Add a link to the index.html if it exists
        index_path = os.path.join(result_dir, 'index.html')
        if os.path.exists(index_path):
            st.markdown(f"[View Analysis Results]({index_path})")
        
    except Exception as e:
        status.error(f"Error: {str(e)}")
else:
    # Simple instructions
    st.info("Select files and click 'Run Analysis' to begin")