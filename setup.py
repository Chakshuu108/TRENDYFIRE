"""
Setup Script for Demand Forecasting Project
Automates the installation and setup process
"""

import subprocess
import sys
import os

def print_step(step_num, message):
    """Print formatted step message"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {message}")
    print(f"{'='*60}\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        return False

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("DEMAND FORECASTING PROJECT - AUTOMATED SETUP")
    print("="*60)
    
    # Step 1: Check Python version
    print_step(1, "Checking Python Version")
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print("Please upgrade Python and try again.")
        sys.exit(1)
    else:
        print("✅ Python version is compatible")
    
    # Step 2: Install requirements
    print_step(2, "Installing Python Dependencies")
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing packages from requirements.txt"
    )
    
    if not success:
        print("\n⚠️  Some packages may have failed to install.")
        print("You can continue, but some features might not work.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 3: Download NLTK data
    print_step(3, "Downloading NLTK Data")
    nltk_commands = [
        "import nltk; nltk.download('punkt')",
        "import nltk; nltk.download('stopwords')",
        "import nltk; nltk.download('wordnet')",
        "import nltk; nltk.download('averaged_perceptron_tagger')"
    ]
    
    for cmd in nltk_commands:
        run_command(f"{sys.executable} -c \"{cmd}\"", f"Downloading NLTK data")
    
    # Step 4: Create directories
    print_step(4, "Creating Project Directories")
    directories = [
        'data/raw',
        'data/processed',
        'data/models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    # Step 5: Generate sample data
    print_step(5, "Generating Sample Data")
    success = run_command(
        f"{sys.executable} src/utils/sample_data_generator.py",
        "Generating sample demand and review data"
    )
    
    # Final message
    print("\n" + "="*60)
    print("SETUP COMPLETE! 🎉")
    print("="*60)
    
    print("\nNext Steps:")
    print("1. Activate your virtual environment (if using one)")
    print("2. Run: streamlit run streamlit_app/app.py")
    print("3. The app will open in your browser")
    print("\nFor detailed instructions, see HOW_TO_RUN.md")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
