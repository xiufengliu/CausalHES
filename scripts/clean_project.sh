#!/bin/bash

# CausalHES Project Cleanup Script
# This script cleans up the project for GitHub publication

set -e

echo "🧹 Cleaning CausalHES project for GitHub publication..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[CLEAN]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Remove Python cache files
clean_python_cache() {
    print_status "Removing Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type f -name "*.pyd" -delete 2>/dev/null || true
    find . -type f -name ".coverage" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
}

# Remove temporary files
clean_temp_files() {
    print_status "Removing temporary files..."
    find . -type f -name "tmp_*" -delete 2>/dev/null || true
    find . -type f -name "temp_*" -delete 2>/dev/null || true
    find . -type f -name "*.tmp" -delete 2>/dev/null || true
    find . -type f -name "*.temp" -delete 2>/dev/null || true
    find . -type f -name "*~" -delete 2>/dev/null || true
    find . -type f -name "*.swp" -delete 2>/dev/null || true
    find . -type f -name "*.swo" -delete 2>/dev/null || true
}

# Remove build artifacts
clean_build_artifacts() {
    print_status "Removing build artifacts..."
    rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
    rm -rf .tox/ .nox/ .pytest_cache/ 2>/dev/null || true
    rm -rf htmlcov/ .coverage.* 2>/dev/null || true
}

# Remove IDE files
clean_ide_files() {
    print_status "Removing IDE files..."
    rm -rf .vscode/ .idea/ 2>/dev/null || true
    find . -type f -name ".DS_Store" -delete 2>/dev/null || true
    find . -type f -name "Thumbs.db" -delete 2>/dev/null || true
}

# Remove LaTeX compilation files
clean_latex_files() {
    print_status "Removing LaTeX compilation files..."
    find paper/ -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.blg" \
        -o -name "*.log" -o -name "*.out" -o -name "*.synctex.gz" \
        -o -name "*.fdb_latexmk" -o -name "*.fls" -o -name "*.toc" \
        -o -name "*.lof" -o -name "*.lot" \) -delete 2>/dev/null || true
}

# Remove large data files (keep structure)
clean_data_files() {
    print_status "Cleaning data directories..."
    # Remove large data files but keep directory structure
    find data/ -type f -name "*.csv" -size +10M -delete 2>/dev/null || true
    find data/ -type f -name "*.npz" -size +10M -delete 2>/dev/null || true
    find data/ -type f -name "*.npy" -size +10M -delete 2>/dev/null || true
    
    # Create .gitkeep files in empty directories
    find data/ -type d -empty -exec touch {}/.gitkeep \; 2>/dev/null || true
}

# Remove output files
clean_outputs() {
    print_status "Cleaning output directories..."
    rm -rf outputs/ results/ checkpoints/ logs/ figures/ plots/ 2>/dev/null || true
    
    # Create output directories with .gitkeep
    mkdir -p outputs/{figures,results,checkpoints,logs}
    touch outputs/figures/.gitkeep
    touch outputs/results/.gitkeep
    touch outputs/checkpoints/.gitkeep
    touch outputs/logs/.gitkeep
}

# Remove model checkpoints
clean_checkpoints() {
    print_status "Removing model checkpoints..."
    find . -type f -name "*.pth" -delete 2>/dev/null || true
    find . -type f -name "*.pt" -delete 2>/dev/null || true
    find . -type f -name "*.ckpt" -delete 2>/dev/null || true
}

# Validate project structure
validate_structure() {
    print_status "Validating project structure..."
    
    # Check required files
    required_files=(
        "README.md"
        "requirements.txt"
        "setup.py"
        "LICENSE"
        ".gitignore"
        "CONTRIBUTING.md"
        "CHANGELOG.md"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_warning "Missing required file: $file"
        fi
    done
    
    # Check required directories
    required_dirs=(
        "src"
        "tests"
        "examples"
        "docs"
        "configs"
        "scripts"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            print_warning "Missing required directory: $dir"
        fi
    done
}

# Check file sizes
check_file_sizes() {
    print_status "Checking for large files..."
    large_files=$(find . -type f -size +50M 2>/dev/null || true)
    if [ -n "$large_files" ]; then
        print_warning "Large files found (>50MB):"
        echo "$large_files"
        print_warning "Consider adding these to .gitignore or using Git LFS"
    fi
}

# Format Python code
format_code() {
    print_status "Formatting Python code..."
    if command -v black &> /dev/null; then
        black src/ tests/ examples/ --line-length 88 --quiet || true
    else
        print_warning "Black not installed, skipping code formatting"
    fi
}

# Run linting
run_linting() {
    print_status "Running code linting..."
    if command -v flake8 &> /dev/null; then
        flake8 src/ tests/ examples/ --max-line-length=88 --extend-ignore=E203,W503 || true
    else
        print_warning "Flake8 not installed, skipping linting"
    fi
}

# Generate file tree
generate_file_tree() {
    print_status "Generating project structure..."
    if command -v tree &> /dev/null; then
        tree -I '__pycache__|*.pyc|.git|venv|env|.pytest_cache|.coverage|htmlcov' > PROJECT_STRUCTURE.txt
    else
        print_warning "Tree command not available, skipping structure generation"
    fi
}

# Main cleanup function
main() {
    echo "🎯 CausalHES Project Cleanup"
    echo "============================"
    
    clean_python_cache
    clean_temp_files
    clean_build_artifacts
    clean_ide_files
    clean_latex_files
    clean_data_files
    clean_outputs
    clean_checkpoints
    
    format_code
    run_linting
    
    validate_structure
    check_file_sizes
    generate_file_tree
    
    echo ""
    echo "✅ Project cleanup completed!"
    echo ""
    echo "📋 Pre-publication checklist:"
    echo "   [ ] Update version numbers"
    echo "   [ ] Update CHANGELOG.md"
    echo "   [ ] Review README.md"
    echo "   [ ] Check .gitignore completeness"
    echo "   [ ] Run tests: python run_tests.py"
    echo "   [ ] Test installation: pip install -e ."
    echo "   [ ] Review documentation"
    echo "   [ ] Add GitHub repository URL"
    echo ""
    echo "🚀 Ready for GitHub publication!"
}

# Run main function
main "$@"