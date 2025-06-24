# CausalHES Enhanced Baselines - Complete Run Instructions

This guide provides step-by-step instructions to run the enhanced baseline experiments with PDF figure generation for your TNNLS submission.

## Prerequisites

### 1. Environment Setup

Ensure you have Python 3.8+ with the following packages installed:

```bash
# Core dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
pip install scipy jupyter notebook tqdm pyyaml

# Additional dependencies for baselines
pip install plotly tensorboard statsmodels
```

### 2. Data Preparation

The Irish CER dataset should be placed in the correct directory structure:

```
data/
├── Irish/
│   ├── ElectricityConsumption.csv
│   ├── household_characteristics.csv
│   └── weather_data.csv
└── processed_irish/
    └── (will be created automatically)
```

## Quick Start

### Option 1: Run Enhanced Baseline Experiments Only

```bash
# Navigate to the project root
cd /work3/xiuli/CausalHES

# Run enhanced baseline experiments
python experiments/run_enhanced_baseline_experiments.py
```

### Option 2: Run Full Experimental Suite

```bash
# Run original experiments
python experiments/run_irish_dataset_experiments.py

# Run enhanced baselines
python experiments/run_enhanced_baseline_experiments.py
```

## Detailed Run Instructions

### Step 1: Verify Directory Structure

```bash
# Check if you're in the correct directory
pwd
# Should show: /work3/xiuli/CausalHES

# Verify the structure
ls -la
# Should show: src/, experiments/, data/, configs/, etc.
```

### Step 2: Run Data Processing (if needed)

```bash
# Process the Irish dataset (creates necessary preprocessing)
python process_irish_dataset.py
```

### Step 3: Run Enhanced Baseline Experiments

```bash
# Run the comprehensive baseline comparison
python experiments/run_enhanced_baseline_experiments.py
```

This will:
1. Load and preprocess the Irish CER dataset (500 households)
2. Run all baseline categories systematically:
   - Traditional methods (K-means, PCA+K-means, SAX, Two-stage)
   - Weather normalization methods (Linear, Ridge, CDD/HDD, Neural)
   - VAE-based methods (β-VAE, FactorVAE, Multi-modal VAE)
   - Multi-modal methods (Attention-fused DEC, Contrastive, Graph-based)
   - Causal inference methods (Doubly Robust, IV, Domain Adaptation)
   - CausalHES baseline
3. Generate comprehensive PDF reports and visualizations

### Step 4: Monitor Progress

The script will output progress logs like:
```
2024-XX-XX XX:XX:XX - INFO - Starting complete enhanced baseline experiments...
2024-XX-XX XX:XX:XX - INFO - Preparing Irish dataset...
2024-XX-XX XX:XX:XX - INFO - Running traditional clustering baselines...
2024-XX-XX XX:XX:XX - INFO - Running K-means (Load)...
2024-XX-XX XX:XX:XX - INFO - K-means (Load) - ACC: 0.3493
...
```

## Expected Runtime

- **Total runtime**: ~4-6 hours on a single GPU
- **Traditional methods**: ~30 minutes
- **Weather normalization**: ~1 hour
- **VAE methods**: ~2 hours
- **Multi-modal methods**: ~2 hours
- **Causal methods**: ~1 hour
- **Report generation**: ~15 minutes

## Output Files

All results will be saved to `experiments/results/enhanced_baselines/`:

### LaTeX Tables (Publication Ready)
```
enhanced_baselines/
├── enhanced_baseline_table.tex           # Ready-to-use LaTeX table for paper
├── enhanced_baseline_summary.csv         # Numerical results in CSV format
├── enhanced_baseline_results.json        # Complete experimental results
└── enhanced_baselines.log               # Detailed execution logs
```

### Main Outputs for Paper
- **`enhanced_baseline_table.tex`** - LaTeX table ready for direct inclusion in paper
- **`enhanced_baseline_summary.csv`** - Clean CSV format for further analysis

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in the script
   # Edit line ~XXX in run_enhanced_baseline_experiments.py
   batch_size=16  # Instead of 32
   ```

2. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install missing_package_name
   ```

3. **Data Not Found**
   ```bash
   # Check data directory
   ls data/Irish/
   # Should show: ElectricityConsumption.csv, household_characteristics.csv, weather_data.csv
   ```

4. **Matplotlib Backend Issues**
   ```bash
   # If running on server without display
   export MPLBACKEND=Agg
   python experiments/run_enhanced_baseline_experiments.py
   ```

### Performance Optimization

1. **For faster testing** (reduced dataset):
   ```python
   # Edit line ~XX in run_enhanced_baseline_experiments.py
   n_households=100  # Instead of 500
   epochs=25         # Instead of 50
   ```

2. **For CPU-only execution**:
   ```python
   # Edit device settings in the baseline files
   device='cpu'  # Instead of 'cuda'
   ```

## Using Results for Paper

### Direct LaTeX Table Inclusion
The file `enhanced_baseline_table.tex` contains a ready-to-use LaTeX table:
```latex
\begin{table}[t]
\centering
\caption{Enhanced Baseline Comparison on Irish CER Dataset...}
\label{tab:enhanced_baseline_results}
...
\end{table}
```

**To use in your paper:**
1. Copy the content of `enhanced_baseline_table.tex`
2. Paste directly into your paper LaTeX source
3. Adjust the `\label{}` if needed to match your reference style

### CSV Data for Analysis
The file `enhanced_baseline_summary.csv` contains clean numerical data:
```csv
Category,Method,ACC,NMI,ARI
Traditional Methods,K-means (Load),0.35$^\dagger$,0.080$^\dagger$,-0.013$^\dagger$
...
CausalHES (Ours),CausalHES (Ours),\textbf{0.88},\textbf{0.812},\textbf{0.791}
```

### Table Features
- **Organized by method category** (Traditional → Weather Norm → VAE → Multi-modal → Causal)
- **Statistical significance markers** ($^\dagger$ for p < 0.01)
- **Bold formatting** for best results (CausalHES)
- **Publication-ready formatting** matching TNNLS style

## Advanced Configuration

### Customize Experiments
Edit `experiments/run_enhanced_baseline_experiments.py` to:
- Change dataset size: `n_households=500`
- Modify training epochs: `epochs=50`
- Adjust baseline selection: Comment out unwanted categories
- Change evaluation metrics: Add/remove metrics in report generation

### Add New Baselines
1. Implement in appropriate `src/clustering/` file
2. Add to the experimental runner
3. Update the `__init__.py` imports

## GPU Requirements

- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **CPU alternative**: Available but slower (~12+ hours)

## Contact for Issues

If you encounter any issues:
1. Check the log file: `experiments/results/enhanced_baselines/enhanced_baselines.log`
2. Verify all dependencies are installed
3. Ensure data is in the correct location
4. Check GPU memory availability

The experiments are designed to be robust and will continue even if individual methods fail, so you should get partial results even with some issues.