#!/usr/bin/env python
"""
Quick start script for Multilingual Safety Evaluation Framework.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)


def create_virtual_env():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("Virtual environment created.")
    else:
        print("Virtual environment already exists.")


def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = ["venv\\Scripts\\pip.exe"]
    else:  # Unix/Linux/MacOS
        pip_cmd = ["venv/bin/pip"]
    
    # Upgrade pip
    subprocess.run(pip_cmd + ["install", "--upgrade", "pip"])
    
    # Install requirements
    subprocess.run(pip_cmd + ["install", "-r", "requirements.txt"])
    print("Dependencies installed.")


def setup_config():
    """Setup configuration file if needed."""
    config_path = Path("configs/config.yaml")
    example_path = Path("configs/config.example.yaml")
    
    if not config_path.exists() and example_path.exists():
        print("\nSetting up configuration...")
        import shutil
        shutil.copy(example_path, config_path)
        print("Configuration file created. Please edit configs/config.yaml with your API keys.")
    elif config_path.exists():
        print("\nConfiguration file already exists.")
    else:
        print("\nWarning: No example configuration found.")


def create_directories():
    """Create necessary directories."""
    dirs = [
        "data/cache",
        "data/datasets", 
        "data/results",
        "logs",
        "reports"
    ]
    
    print("\nCreating directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("Directories created.")


def run_tests():
    """Run basic tests to verify installation."""
    print("\nRunning basic tests...")
    
    # Determine python command based on OS
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python.exe"
    else:  # Unix/Linux/MacOS
        python_cmd = "venv/bin/python"
    
    # Try importing main modules
    test_code = """
import sys
sys.path.insert(0, '.')

try:
    from src.core import SafetyEvaluator
    from src.models import DummyModel
    from src.evaluation import HarmfulContentEvaluator
    print("✓ Core modules imported successfully")
    
    # Test dummy model
    model = DummyModel()
    response = model.generate("test")
    print("✓ Dummy model working")
    
    # Test evaluator
    evaluator = HarmfulContentEvaluator()
    result = evaluator.evaluate("test", "response")
    print("✓ Evaluator working")
    
    print("\\nAll tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
"""
    
    result = subprocess.run([python_cmd, "-c", test_code], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*50)
    print("Setup completed successfully!")
    print("="*50)
    
    print("\nNext steps:")
    print("1. Edit configs/config.yaml with your API keys")
    print("2. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n3. Try the examples:")
    print("   python examples/basic_usage.py")
    print("   python -m src.cli evaluate -p 'Tell me about AI safety'")
    
    print("\n4. Start the API server:")
    print("   python -m src.api.app")
    
    print("\n5. Read the documentation:")
    print("   - README.md")
    print("   - docs/API_GUIDE.md")
    print("   - docs/CONFIGURATION.md")
    
    print("\nFor more help, see: https://github.com/ml-safety-framework/multilingual-safety-evaluation")


def main():
    """Main setup function."""
    print("Multilingual Safety Evaluation Framework - Quick Setup")
    print("=" * 55)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_env()
    
    # Install dependencies
    install_dependencies()
    
    # Setup configuration
    setup_config()
    
    # Create directories
    create_directories()
    
    # Run tests
    run_tests()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()