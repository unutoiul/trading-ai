# filepath: c:\projects\trading-ai\platform_check.py

import os
import sys
import platform

def check_platform_compatibility():
    """Check for potential cross-platform issues."""
    print(f"Running on: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    
    # Check file paths
    for root, dirs, files in os.walk('.'):
        # Skip version control and virtual environments
        if '.git' in root or 'venv' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    try:
                        content = f.read()
                        # Check for Windows-specific path separators
                        if '\\\\' in content and platform.system() != 'Windows':
                            print(f"WARNING: Windows path separator found in {filepath}")
                        # Check for f-string issues (simplified check)
                        if '{"' in content or '"}' in content:
                            print(f"POTENTIAL ISSUE: Check f-strings in {filepath}")
                    except UnicodeDecodeError:
                        print(f"ERROR: Encoding issue in {filepath}")

if __name__ == "__main__":
    check_platform_compatibility()