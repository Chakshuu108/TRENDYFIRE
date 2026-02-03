#!/usr/bin/env python3
"""
Quick Start Script
Launches the Streamlit application
"""

import subprocess
import sys
import os

def main():
    print("\n" + "="*60)
    print("LAUNCHING DEMAND FORECASTING APPLICATION")
    print("="*60 + "\n")
    
    # Check if in project directory
    if not os.path.exists('streamlit_app/app.py'):
        print("❌ Error: Please run this script from the project root directory")
        print("   Current directory:", os.getcwd())
        sys.exit(1)
    
    print("Starting Streamlit app...")
    print("The app will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the application")
    print("="*60 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py"])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure streamlit is installed: pip install streamlit")
        print("2. Check that you're in the project directory")
        print("3. See HOW_TO_RUN.md for detailed instructions")

if __name__ == "__main__":
    main()
