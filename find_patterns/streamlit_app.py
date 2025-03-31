import streamlit as st
import os
import sys
import threading
import time

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import main function
from src.run_analysis import main

# Page setup
st.set_page_config(page_title="Trading AI", layout="wide")

# Session state for progress tracking
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'progress_message' not in st.session_state:
    st.session_state.progress_message = ""
if 'result_dir' not in st.session_state:
    st.session_state.result_dir = None

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
    
    # Run button (disabled during processing)
    run_button = st.button("Run Analysis", type="primary", 
                          disabled=st.session_state.processing)

# Create a status area
status_area = st.empty()

# Handle run button click
if run_button:
    st.session_state.processing = True
    st.session_state.progress_message = "Starting analysis..."
    st.session_state.result_dir = None
    
    # Function to run in background thread
    def run_analysis_thread():
        try:
            # Prepare command arguments
            sys.argv = ['run.py']
            sys.argv.extend(['--btc', os.path.join('data', btc_file)])
            sys.argv.extend(['--doge', os.path.join('data', alt_file)])
            
            if use_ml:
                sys.argv.append('--use-ml')
                st.session_state.progress_message = "Running with ML enabled..."
            
            if optimize:
                sys.argv.append('--optimize-strategy')
                st.session_state.progress_message = "Strategy optimization enabled..."
            
            # Simulate progress updates
            def update_status():
                stages = [
                    "Loading data files...", 
                    "Analyzing BTC-Altcoin patterns...",
                    "Calculating correlations...",
                    "Running machine learning..." if use_ml else None,
                    "Optimizing trading strategy..." if optimize else None,
                    "Generating visualizations...",
                    "Creating reports..."
                ]
                stages = [s for s in stages if s]  # Remove None items
                
                for stage in stages:
                    st.session_state.progress_message = stage
                    time.sleep(2)  # Wait a bit between updates
            
            # Start status updater thread
            status_thread = threading.Thread(target=update_status)
            status_thread.start()
            
            # Run the actual analysis
            result_dir = main(return_results_dir=True)
            
            # Save the result directory
            st.session_state.result_dir = result_dir
            st.session_state.progress_message = "Analysis complete!"
            
        except Exception as e:
            st.session_state.progress_message = f"Error: {str(e)}"
        finally:
            st.session_state.processing = False
    
    # Start the analysis thread
    threading.Thread(target=run_analysis_thread).start()

# Show current status
if st.session_state.processing:
    status_area.info(st.session_state.progress_message)
elif st.session_state.result_dir:
    # Show success and results
    status_area.success(f"Analysis complete!")
    
    # Show the results
    st.header("Analysis Results")
    
    # Display path
    st.code(f"Results saved to: {st.session_state.result_dir}")
    
    # Create columns to display results
    col1, col2 = st.columns(2)
    
    with col1:
        # Show main reports
        st.subheader("Reports")
        reports_dir = os.path.join(st.session_state.result_dir, 'reports')
        if os.path.exists(reports_dir):
            for file in os.listdir(reports_dir):
                file_path = os.path.join(reports_dir, file)
                st.download_button(f"📊 {file}", 
                                  open(file_path, 'rb').read(),
                                  file_name=file)
    
    with col2:
        # Show charts
        st.subheader("Charts")
        charts_dir = os.path.join(st.session_state.result_dir, 'charts')
        if os.path.exists(charts_dir):
            for file in os.listdir(charts_dir):
                if file.endswith(('.png', '.jpg')):
                    file_path = os.path.join(charts_dir, file)
                    st.image(file_path, caption=file, use_column_width=True)
    
    # Link to HTML results
    st.subheader("Interactive Results")
    index_path = os.path.join(st.session_state.result_dir, 'index.html')
    if os.path.exists(index_path):
        html_content = open(index_path, 'r').read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    
        st.markdown("**Note**: For best experience, open the full report:")
        st.markdown(f"[Open Full Analysis Results](file://{index_path})")
    
else:
    # Show instructions
    status_area.info("Select files and click 'Run Analysis' to begin")