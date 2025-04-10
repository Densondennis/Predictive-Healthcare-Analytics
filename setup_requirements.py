import subprocess
import sys

# List of required libraries
required_libraries = [
    'pandas',
    'numpy',
    'scikit-learn',
    'xgboost',
    'matplotlib',
    'seaborn',
    'imbalanced-learn',
    'flask',
    'jupyter'  # optional, if you're using Jupyter notebooks
]

# Function to install libraries

def  install_libraries(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
        sys.exit(1)

# Check for missing libraries and install them
for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        print(f"{lib} not found. Installing...")
        install_libraries(lib)

print("All required libraries are installed and ready!")
