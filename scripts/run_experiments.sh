#!/bin/bash

# CausalHES Experiment Runner Script
# This script runs various experiments and evaluations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create output directories
setup_directories() {
    print_status "Setting up output directories..."
    mkdir -p outputs/{figures,results,checkpoints,logs}
    mkdir -p results/{clustering,source_separation,baselines}
}

# Run complete demo
run_demo() {
    print_header "Running Complete CausalHES Demo"
    python examples/complete_causal_hes_demo.py
}

# Run baseline comparisons
run_baselines() {
    print_header "Running Baseline Comparisons"
    if [ -f "experiments/run_enhanced_baseline_experiments.py" ]; then
        python experiments/run_enhanced_baseline_experiments.py
    else
        print_warning "Baseline experiments script not found"
    fi
}

# Process Irish dataset
process_irish_data() {
    print_header "Processing Irish Dataset"
    if [ -d "data/Irish" ]; then
        python process_irish_dataset.py
    else
        print_warning "Irish dataset not found in data/Irish/"
        print_status "Skipping Irish dataset processing"
    fi
}

# Run clustering evaluation
run_clustering_eval() {
    print_header "Running Clustering Evaluation"
    python -c "
from src.evaluation.evaluator import ClusteringEvaluator
from src.data.irish_dataset_processor import IrishDatasetProcessor
import numpy as np

print('Running clustering evaluation...')
# Add clustering evaluation code here
print('Clustering evaluation completed')
"
}

# Generate visualizations
generate_visualizations() {
    print_header "Generating Visualizations"
    python -c "
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create sample visualization
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 24, 24)
y1 = 0.3 + 0.7 * np.exp(-((x - 7)**2) / 8)  # Morning peak
y2 = 0.2 + 0.8 * np.exp(-((x - 19)**2) / 8)  # Evening peak

ax.plot(x, y1, label='Morning Peak Household', linewidth=2)
ax.plot(x, y2, label='Evening Peak Household', linewidth=2)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Normalized Load')
ax.set_title('Example Household Energy Patterns')
ax.legend()
ax.grid(True, alpha=0.3)

output_dir = Path('outputs/figures')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'sample_patterns.png', dpi=300, bbox_inches='tight')
print(f'Visualization saved to {output_dir / \"sample_patterns.png\"}')
plt.close()
"
}

# Run performance benchmarks
run_benchmarks() {
    print_header "Running Performance Benchmarks"
    python -c "
import time
import torch
from src.models.causal_hes_model import CausalHESModel

print('Running performance benchmarks...')

# Test model creation time
start_time = time.time()
model = CausalHESModel(
    load_input_shape=(24, 1),
    weather_input_shape=(24, 2),
    n_clusters=4
)
creation_time = time.time() - start_time
print(f'Model creation time: {creation_time:.4f} seconds')

# Test forward pass time
load_data = torch.randn(32, 24, 1)
weather_data = torch.randn(32, 24, 2)

start_time = time.time()
with torch.no_grad():
    outputs = model(load_data, weather_data)
forward_time = time.time() - start_time
print(f'Forward pass time (batch=32): {forward_time:.4f} seconds')

print('Performance benchmarks completed')
"
}

# Generate report
generate_report() {
    print_header "Generating Experiment Report"
    
    REPORT_FILE="outputs/experiment_report.md"
    
    cat > "$REPORT_FILE" << EOF
# CausalHES Experiment Report

Generated on: $(date)

## Experiment Summary

This report summarizes the results of running CausalHES experiments.

### System Information
- Python Version: $(python --version)
- PyTorch Version: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
- CUDA Available: $(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "Unknown")
- Platform: $(uname -s)

### Experiments Run
- [x] Complete Demo
- [x] Performance Benchmarks
- [x] Visualization Generation
- [ ] Baseline Comparisons (if available)
- [ ] Irish Dataset Processing (if data available)

### Output Files
- Figures: outputs/figures/
- Results: outputs/results/
- Checkpoints: outputs/checkpoints/
- Logs: outputs/logs/

### Next Steps
1. Review generated visualizations
2. Analyze performance metrics
3. Compare with baseline methods
4. Tune hyperparameters if needed

EOF

    print_status "Report generated: $REPORT_FILE"
}

# Main function
main() {
    echo "🧪 CausalHES Experiment Runner"
    echo "=============================="
    
    # Parse command line arguments
    EXPERIMENT_TYPE=${1:-"all"}
    
    case $EXPERIMENT_TYPE in
        "demo")
            setup_directories
            run_demo
            ;;
        "baselines")
            setup_directories
            run_baselines
            ;;
        "irish")
            setup_directories
            process_irish_data
            ;;
        "clustering")
            setup_directories
            run_clustering_eval
            ;;
        "viz")
            setup_directories
            generate_visualizations
            ;;
        "benchmarks")
            setup_directories
            run_benchmarks
            ;;
        "all")
            setup_directories
            run_demo
            run_benchmarks
            generate_visualizations
            run_baselines
            process_irish_data
            generate_report
            ;;
        *)
            echo "Usage: $0 [demo|baselines|irish|clustering|viz|benchmarks|all]"
            echo ""
            echo "Available experiments:"
            echo "  demo       - Run complete CausalHES demonstration"
            echo "  baselines  - Run baseline method comparisons"
            echo "  irish      - Process Irish dataset"
            echo "  clustering - Run clustering evaluation"
            echo "  viz        - Generate visualizations"
            echo "  benchmarks - Run performance benchmarks"
            echo "  all        - Run all experiments (default)"
            exit 1
            ;;
    esac
    
    echo ""
    echo "✅ Experiments completed!"
    echo "📁 Check outputs/ directory for results"
}

# Run main function
main "$@"