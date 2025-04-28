# filepath: c:\projects\trading-ai\platform_check.py

import os
import sys
import platform
import re

def check_platform_compatibility():
    """Check for potential cross-platform issues."""
    print(f"Running on: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    
    # Check file paths
    for root, dirs, files in os.walk('.'):
        # Skip version control and virtual environments
        if '.git' in root or 'venv' in root or '__pycache__' in root or 'node_modules' in root:
            continue
               
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    try:
                        content = f.read()
                        line_num = 1
                        for line in content.split('\n'):
                            # Check for Windows-specific path separators
                            if '\\\\' in line and platform.system() != 'Windows':
                                print(f"WARNING: Windows path separator in {filepath}:{line_num}")
                                
                            # Check for f-string issues - looking for problematic patterns
                            if "{'" in line or "'}" in line:
                                print(f"POTENTIAL ISSUE: Suspicious f-string pattern in {filepath}:{line_num}")
                                
                            # Check for nested quotes
                            if re.search(r'"[^"]*\'[^\']*\'[^"]*"', line) or re.search(r"'[^']*\"[^\"]*\"[^']*'", line):
                                print(f"POTENTIAL ISSUE: Nested quotes in {filepath}:{line_num}")
                                
                            # Check for HTML class attributes in f-strings
                            if re.search(r'class="{\w+', line) or re.search(r"class='{\w+", line):
                                print(f"POTENTIAL ISSUE: HTML class in f-string in {filepath}:{line_num}")
                                
                            line_num += 1
                    except UnicodeDecodeError:
                        print(f"ERROR: Encoding issue in {filepath}")
                    except Exception as e:
                        print(f"ERROR checking {filepath}: {str(e)}")
    
    # Print fix suggestions
    print("\n============ SUGGESTED FIXES ============")
    print("1. Use os.path.join() for paths instead of hardcoded separators")
    print("2. For HTML class attributes in f-strings, use helper functions:")
    print("   def get_class_name(value):")
    print("       return 'up-value' if value > 0 else 'down-value'")
    print("   Then: f'<div class=\"{get_class_name(value)}\">'")
    print("3. Break complex f-strings into smaller chunks")
    print("4. Use separate string formatting for complex HTML templates")

if __name__ == "__main__":
    check_platform_compatibility()