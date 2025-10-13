"""
Run script for Safety Helmet and Vest Detection System
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import streamlit
        import cv2
        import ultralytics
        import numpy
        import matplotlib
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def main():
    """Main run function"""
    print("🦺 Safety Helmet and Vest Detection System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run from the project root directory.")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("🚀 Starting Streamlit application...")
    print("📱 The dashboard will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
