#!/usr/bin/env python3
"""
Installation script for expense categorizer dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def install_dependencies():
    """Install all required dependencies"""
    print("Installing Expense Categorizer Dependencies")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or python_version.minor < 8:
        print("Warning: Python 3.8+ is recommended")
    
    # Install core dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch (CPU version)"),
        ("pip install transformers==4.30.0", "Installing Transformers"),
        ("pip install accelerate==0.20.3", "Installing Accelerate"),
        ("pip install pandas numpy scikit-learn", "Installing data processing libraries"),
        ("pip install datasets tokenizers", "Installing additional ML libraries"),
        ("pip install spacy", "Installing spaCy"),
        ("python -m spacy download en_core_web_sm", "Downloading spaCy English model"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    print(f"\n\nInstallation Summary:")
    print(f"✓ {success_count}/{len(commands)} components installed successfully")
    
    if success_count == len(commands):
        print("\n🎉 All dependencies installed successfully!")
        print("You can now run the expense categorizer.")
    else:
        print(f"\n⚠️  {len(commands) - success_count} components failed to install.")
        print("The script may still work with reduced functionality.")
    
    # Verify installations
    print("\nVerifying installations...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not available")
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not available")
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError:
        print("✗ Pandas not available")
    
    try:
        import spacy
        print(f"✓ spaCy {spacy.__version__}") # type: ignore
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy English model available")
        except OSError:
            print("✗ spaCy English model not available")
    except ImportError:
        print("✗ spaCy not available")

if __name__ == "__main__":
    install_dependencies()