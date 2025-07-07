#!/bin/bash

# CausalHES Environment Setup Script
# This script sets up the complete development environment for CausalHES

set -e  # Exit on any error

echo "🚀 Setting up CausalHES development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
        
        # Check if version is >= 3.8
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python version is compatible"
        else
            print_error "Python 3.8+ required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Check if Git is installed
check_git() {
    print_status "Checking Git installation..."
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        print_status "Git $GIT_VERSION found"
    else
        print_error "Git not found. Please install Git"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_status "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "Requirements installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Install PyTorch
    print_status "Installing PyTorch..."
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected, installing PyTorch with CUDA support"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "No NVIDIA GPU detected, installing CPU-only PyTorch"
        pip install torch torchvision torchaudio
    fi
    
    # Install CausalHES in development mode
    pip install -e .
    print_status "CausalHES installed in development mode"
}

# Install development tools
install_dev_tools() {
    print_status "Installing development tools..."
    pip install black flake8 pytest pytest-cov pre-commit
    
    # Setup pre-commit hooks
    pre-commit install
    print_status "Development tools installed"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    python run_tests.py
    if [ $? -eq 0 ]; then
        print_status "All tests passed!"
    else
        print_warning "Some tests failed. Check the output above."
    fi
}

# Create directories
create_directories() {
    print_status "Creating project directories..."
    mkdir -p data outputs checkpoints logs results
    print_status "Directories created"
}

# Main setup function
main() {
    echo "🎯 CausalHES Development Environment Setup"
    echo "=========================================="
    
    check_python
    check_git
    create_venv
    activate_venv
    install_dependencies
    install_dev_tools
    create_directories
    run_tests
    
    echo ""
    echo "✅ Setup completed successfully!"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Activate the environment: source venv/bin/activate"
    echo "   2. Run the demo: python examples/complete_causal_hes_demo.py"
    echo "   3. Read the documentation: docs/"
    echo "   4. Start developing!"
    echo ""
    echo "📚 Useful commands:"
    echo "   - Run tests: python run_tests.py"
    echo "   - Format code: black src/ tests/ examples/"
    echo "   - Check style: flake8 src/ tests/ examples/"
    echo "   - Run demo: python examples/complete_causal_hes_demo.py"
}

# Run main function
main "$@"