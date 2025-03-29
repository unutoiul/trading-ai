"""
Helper script to set the Anthropic API key for Claude analysis.
"""

import os
import sys
import argparse

def set_api_key(key):
    """Set the ANTHROPIC_API_KEY environment variable."""
    # For Windows
    os.system(f'setx ANTHROPIC_API_KEY "{key}"')
    print("API key set! Please restart your terminal for changes to take effect.")
    print("You can now run analysis with Claude 3.7 integration.")

def main():
    parser = argparse.ArgumentParser(description="Set Anthropic API key for Claude analysis")
    parser.add_argument('key', help='Your Anthropic API key')
    args = parser.parse_args()
    
    set_api_key(args.key)

if __name__ == "__main__":
    main()