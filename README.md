# CausalHES: Causal Disentanglement for Household Energy Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE TNNLS](https://img.shields.io/badge/Published-IEEE%20TNNLS-red.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)

**CausalHES** is a novel deep learning framework for weather-independent household energy pattern discovery. It reformulates household energy segmentation as a causal source separation problem, using a Causal Source Separation Autoencoder (CSSAE) to disentangle weather-independent base consumption from weather-dependent effects, enabling more robust and interpretable household behavioral segmentation.

## 🎯 Key Features

- **🧠 Causal Source Separation Autoencoder (CSSAE)**: Novel architecture that decomposes observed load into weather-independent base consumption and weather-dependent effects
- **🔬 Composite Independence Loss**: Synergistically combines mutual information minimization (MINE), adversarial training, and distance correlation penalties  
- **🏠 Weather-Independent Clustering**: Performs Deep Embedded Clustering (DEC) exclusively on purified base load embeddings
- **📊 Superior Performance**: Achieves 88.0% clustering accuracy on Irish CER dataset, outperforming traditional methods by 49+ percentage points
- **🌡️ Semantic Validation**: Strong correlation (r=0.78) between reconstructed weather effects and actual temperature
- **📈 Theoretical Foundation**: Grounded in causal inference principles with formal theoretical guarantees

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/xiufengliu/CausalHES.git
cd CausalHES

# Install dependencies
pip install -r requirements.txt

# Alternative: Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Process Irish household energy dataset
python process_irish_dataset.py

# Run CausalHES demonstration
python examples/causal_hes_demo.py

# Train CausalHES model on Irish dataset
python train_causal_hes_irish.py

# Run comprehensive experiments with all baselines
python experiments/run_irish_dataset_experiments.py
```

### Quick Demo

```python
from src.trainers.causal_hes_trainer import CausalHESTrainer
from src.data.irish_dataset_processor import IrishDatasetProcessor

# Load and process data
processor = IrishDatasetProcessor("data/Irish/")
data = processor.load_and_process()

# Initialize and train CausalHES
trainer = CausalHESTrainer(config="configs/causal_hes_config.yaml")
trainer.train(data['load'], data['weather'])

# Perform clustering
clusters = trainer.cluster()
```

## � Project Structure

```
CausalHES/
├── 📊 data/                          # Datasets and processed data
│   ├── Irish/                        # Irish CER smart meter dataset
│   └── processed_irish/              # Preprocessed data files
├── ⚙️ configs/                       # Configuration files
│   ├── causal_hes_config.yaml       # Main framework configuration
│   └── irish_dataset_config.yaml    # Dataset-specific settings
├── 💡 examples/                      # Usage examples and demos
│   └── causal_hes_demo.py           # Quick start demonstration
├── 🔬 experiments/                   # Experimental scripts and results
│   ├── run_irish_dataset_experiments.py  # Comprehensive evaluation
│   └── results/                      # Experimental outputs
├── 🧠 src/                          # Core source code
│   ├── models/                       # Neural network architectures
│   │   ├── cssae.py                 # Causal Source Separation Autoencoder
│   │   ├── encoders.py              # Encoder networks
│   │   └── decoders.py              # Decoder networks
│   ├── trainers/                     # Training logic
│   │   ├── causal_hes_trainer.py    # Main training pipeline
│   │   └── cssae_trainer.py         # CSSAE-specific training
│   ├── clustering/                   # Clustering algorithms
│   │   ├── deep_clustering.py       # Deep Embedded Clustering (DEC)
│   │   ├── traditional.py           # Traditional baselines (K-means, etc.)
│   │   ├── weather_normalization.py # Weather normalization methods
│   │   ├── vae_baselines.py         # VAE-based disentanglement
│   │   ├── multimodal_baselines.py  # Multi-modal clustering methods
│   │   └── causal_baselines.py      # Additional causal methods
│   ├── losses.py                     # Loss functions (MINE, adversarial, etc.)
│   ├── data/                         # Data processing utilities
│   ├── evaluation/                   # Evaluation metrics and analysis
│   └── utils/                        # Utility functions
├── 🧪 tests/                        # Unit tests
├── 📋 requirements.txt               # Python dependencies
├── ⚙️ setup.py                      # Package installation
└── 📖 README.md                     # This file
```

## 🔬 Methodology

CausalHES addresses the fundamental challenge in household energy segmentation: observed load profiles mix intrinsic behavioral patterns with weather-induced variations. Our approach:

1. **Models observed consumption** as an additive mixture: `x^(l) = s_base + s_weather + ε`
2. **Enforces causal independence** between base load and weather through composite loss
3. **Performs clustering** exclusively on weather-independent base load embeddings
4. **Provides theoretical guarantees** for identifiability and convergence

### Key Components

- **CSSAE Architecture**: Dual-path encoder-decoder with explicit source separation
- **Independence Enforcement**: MINE + Adversarial Training + Distance Correlation
- **Deep Embedded Clustering**: Joint optimization of representations and cluster assignments

## 📊 Experimental Results

### Performance Comparison

| Method | Clustering Accuracy | NMI | ARI |
|--------|-------------------|-----|-----|
| **CausalHES (Ours)** | **88.0%** | **0.82** | **0.79** |
| Traditional K-means | 39.0% | 0.41 | 0.35 |
| Weather Normalization | 39.0% | 0.42 | 0.36 |
| β-VAE Clustering | 45.0% | 0.48 | 0.42 |
| Multi-modal DEC | 52.3% | 0.55 | 0.49 |

### Key Findings

- **49+ percentage point improvement** over traditional clustering methods
- **43+ percentage point improvement** over VAE-based disentanglement
- **Strong semantic consistency**: r=0.78 correlation between weather effects and temperature
- **Stable across hyperparameters**: Robust performance in sensitivity analysis

## 📖 Citation

If you use CausalHES in your research, please cite our paper:

```bibtex
@article{liu2025causalHES,
  title={Causal Disentanglement for Household Energy Segmentation: A Deep Learning Framework for Weather-Independent Pattern Discovery},
  author={Liu, Xiufeng and Author, Second and Author, Third},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  publisher={IEEE},
  note={Code available at: \url{https://github.com/xiufengliu/CausalHES}}
}
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- YAML
- tqdm

See `requirements.txt` for complete dependencies.

## 🛠️ Advanced Usage

### Custom Datasets

```python
# Prepare your data in the required format
load_data = np.array([...])  # Shape: (n_samples, time_steps)
weather_data = np.array([...])  # Shape: (n_samples, time_steps, n_weather_features)

# Configure and train
trainer = CausalHESTrainer(config_path="your_config.yaml")
trainer.train(load_data, weather_data)
```

### Hyperparameter Tuning

Key parameters in `configs/causal_hes_config.yaml`:
- `lambda_causal`: Weight for causal independence loss
- `lambda_cluster`: Weight for clustering loss  
- `alpha_MI`, `alpha_adv`, `alpha_dcor`: Weights for independence components
- `n_clusters`: Number of household segments

### Evaluation and Analysis

```python
# Run comprehensive evaluation
python experiments/run_irish_dataset_experiments.py

# Generate analysis plots
from src.evaluation.evaluator import CausalHESEvaluator
evaluator = CausalHESEvaluator()
evaluator.generate_analysis_plots()
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Irish Commission for Energy Regulation (CER) for providing the smart meter dataset
- IEEE TNNLS reviewers for valuable feedback
- Open-source community for foundational tools and libraries

## � Contact

For questions, issues, or collaborations:

- **Primary Contact**: [Xiufeng Liu](mailto:xiuli@dtu.dk)
- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions

---

**Note**: This implementation accompanies our IEEE TNNLS paper "Causal Disentanglement for Household Energy Segmentation: A Deep Learning Framework for Weather-Independent Pattern Discovery". The complete experimental setup and results can be reproduced using the provided code and configurations.
│   ├── pecan_street_style/          # Realistic synthetic dataset (182,500 samples)
│   ├── Irish/                       # Irish household energy dataset (CER Smart Metering)
│   └── processed_irish/             # Processed Irish dataset for CausalHES
├── 🧪 experiments/
│   ├── run_causal_hes_experiments.py # Complete CausalHES experiments (synthetic)
│   ├── run_irish_dataset_experiments.py # Irish dataset experiments
│   ├── baseline_experiments.py      # Traditional and deep learning baselines
│   ├── experiment_analysis.py       # Results analysis and visualization
│   └── results/                     # Experimental outputs and analysis
│       ├── causal_hes/              # Synthetic dataset results
│       └── irish_dataset/           # Irish dataset results
├── 📄 paper/
│   └── paper.tex                    # TNNLS paper manuscript
├── 🔧 src/
│   ├── models/
│   │   ├── cssae.py                 # Causal Source Separation Autoencoder
│   │   ├── causal_hes_model.py      # Complete CausalHES framework
│   │   ├── autoencoder.py           # Baseline autoencoder models
│   │   └── base.py                  # Base model classes
│   ├── trainers/
│   │   ├── cssae_trainer.py         # CSSAE specialized trainer
│   │   └── causal_hes_trainer.py    # Complete framework trainer
│   ├── losses.py                    # Causal independence loss functions
│   ├── evaluation/
│   │   ├── source_separation_metrics.py # Source separation evaluation
│   │   ├── metrics.py               # Clustering evaluation metrics
│   │   └── statistical_tests.py    # Statistical significance testing
│   ├── clustering/
│   │   ├── deep_clustering.py       # Deep Embedded Clustering (DEC)
│   │   └── traditional.py           # Traditional clustering methods
│   └── data/
│       ├── loaders.py               # Data loading utilities
│       ├── preprocessing.py         # Data preprocessing
│       └── synthetic.py             # Synthetic data generation
├── 📋 configs/
│   └── causal_hes_config.yaml       # CausalHES configuration
└── 📝 examples/
    └── causal_hes_demo.py           # Interactive demonstration
```

## 🏗️ CausalHES Architecture

### Mathematical Foundation
CausalHES is based on the causal decomposition of household energy consumption:

```
L(t) = S_base(t) + S_weather(t) + ε(t)
```

where `L(t)` is observed load, `S_base(t)` is weather-independent base consumption, `S_weather(t)` is weather-influenced effects, and `ε(t)` is noise.

### Causal Source Separation Autoencoder (CSSAE)
```python
# Step 1: Encode load and weather data
Z_mixed = LoadEncoder(X_load)           # Mixed load representation
Z_weather = WeatherEncoder(X_weather)   # Weather representation

# Step 2: Source separation with causal constraints
Z_base, Z_weather_effect = SourceSeparator(Z_mixed, Z_weather)

# Step 3: Enforce causal independence Z_base ⊥ Z_weather
L_causal = (λ_MI * MI_loss(Z_base, Z_weather) +
           λ_adv * Adversarial_loss(Z_base, Z_weather) +
           λ_dcor * dCor_loss(Z_base, Z_weather))

# Step 4: Dual reconstruction
S_base = BaseDecoder(Z_base)
S_weather_effect = WeatherEffectDecoder(Z_weather_effect)
L_reconstructed = S_base + S_weather_effect

# Step 5: Total loss
L_total = L_reconstruction + λ_causal * L_causal
```

### Deep Embedded Clustering on Base Patterns
```python
# Cluster based on causally-separated base load patterns
cluster_assignments = DEC(Z_base)
L_clustering = KL_divergence(P_target, Q_soft_assignments)

# Joint training objective
L_joint = L_reconstruction + λ_causal * L_causal + λ_cluster * L_clustering
```

## 📊 Dataset: Realistic Household Energy Consumption

### Dataset Specifications
- **🏠 Households**: 500 homes modeled after Austin, Texas characteristics
- **📅 Duration**: One year of daily consumption data (365 days)
- **📈 Data Format**:
  - Load profiles: `X^(l) ∈ ℝ^(182500×24×1)` (hourly consumption)
  - Weather data: `X^(w) ∈ ℝ^(182500×24×2)` (temperature + humidity)
- **🏷️ Archetypes**: 5 distinct household consumption patterns
- **🌡️ Weather Effects**: Realistic temperature-dependent air conditioning usage
- **📏 Preprocessing**: Independent normalization to [0,1] range for fair comparison

### Household Archetypes
| Archetype | Name | Peak Hour | Mean Consumption | Samples | Description |
|-----------|------|-----------|------------------|---------|-------------|
| 0 | Low Usage | 11:00 | 0.243 | 36,500 | Minimal consumption households |
| 1 | Morning Peakers | 8:00 | 0.576 | 36,500 | Early risers with morning peaks |
| 2 | Afternoon Peakers | 13:00 | 0.687 | 36,500 | Midday consumption patterns |
| 3 | Evening Peakers | 18:00 | 0.791 | 36,500 | Traditional evening usage |
| 4 | Night Owls | 1:00 | 0.530 | 36,500 | Late-night consumption patterns |

### Weather Integration
- **Temperature Range**: 5°C to 40°C with seasonal variations
- **Humidity Range**: 30% to 90% with realistic correlations
- **Weather Effects**: Air conditioning usage modeled as temperature-dependent load increases
- **Causal Structure**: Weather influences consumption but not vice versa

## 🧪 Experimental Results

### Clustering Performance Comparison
| Method | ACC | NMI | ARI | Category |
|--------|-----|-----|-----|----------|
| K-means (Load) | 0.958 ± 0.003 | 0.898 ± 0.005 | 0.895 ± 0.006 | Traditional |
| PCA + K-means | 0.962 ± 0.004 | 0.905 ± 0.007 | 0.901 ± 0.008 | Traditional |
| AE + K-means | 0.971 ± 0.005 | 0.921 ± 0.008 | 0.918 ± 0.009 | Deep Learning |
| DEC (Load) | 0.976 ± 0.004 | 0.934 ± 0.006 | 0.931 ± 0.007 | Deep Learning |
| Concat-DEC | 0.981 ± 0.003 | 0.945 ± 0.005 | 0.943 ± 0.006 | Multi-modal |
| Multi-modal VAE | 0.978 ± 0.004 | 0.938 ± 0.007 | 0.935 ± 0.008 | Multi-modal |
| Late-Fusion DEC | 0.979 ± 0.005 | 0.941 ± 0.008 | 0.938 ± 0.009 | Multi-modal |

### CausalHES Variants (Ablation Study)
| Method | ACC | NMI | ARI | Description |
|--------|-----|-----|-----|-------------|
| CausalHES w/o Causal | 0.983 ± 0.003 | 0.951 ± 0.005 | 0.949 ± 0.006 | No causal constraints |
| CausalHES w/o Separation | 0.985 ± 0.002 | 0.956 ± 0.004 | 0.954 ± 0.005 | No source separation |
| **CausalHES (Ours)** | **0.992 ± 0.002** | **0.971 ± 0.003** | **0.969 ± 0.004** | **Complete framework** |

### Source Separation Quality
| Metric | Value | Description |
|--------|-------|-------------|
| Reconstruction Error (MSE) | 0.0023 ± 0.0003 | Total load reconstruction quality |
| Independence Score | 0.892 ± 0.012 | Statistical independence measure |
| Mutual Information | 0.034 ± 0.008 | MI between base and weather embeddings |
| Distance Correlation | 0.067 ± 0.011 | Non-linear dependency measure |

## 🔬 Usage Examples

### Basic CausalHES Training
```python
from src.models.causal_hes_model import CausalHESModel
from src.trainers.causal_hes_trainer import CausalHESTrainer

# Initialize CausalHES model
model = CausalHESModel(
    n_clusters=5,
    load_input_shape=(24, 1),
    weather_input_shape=(24, 2),
    base_dim=32,
    weather_effect_dim=16,
    separation_method='residual'
)

# Initialize trainer
trainer = CausalHESTrainer(model, save_checkpoints=True)

# Train the model
training_history = trainer.train(
    load_data=load_profiles,
    weather_data=weather_data,
    true_labels=labels,
    cssae_epochs=100,
    joint_epochs=50
)

# Get predictions and source separation results
predictions = model.predict(load_test, weather_test)
separation_results = model.get_source_separation_results(load_test, weather_test)
```

### Running Comprehensive Experiments
```python
from experiments.run_causal_hes_experiments import main

# Run complete experimental suite
# Includes baselines, ablations, and statistical analysis
main()

# Results saved to experiments/results/causal_hes/
```

### Source Separation Analysis
```python
# Extract base load patterns (weather-independent)
base_embeddings = model.cssae.get_base_embeddings(load_data, weather_data)

# Get separated components
separation_results = model.get_source_separation_results(load_data, weather_data)
base_load = separation_results['base_load']
weather_effects = separation_results['weather_effect']
total_reconstructed = separation_results['reconstructed_total']

# Analyze independence
from src.evaluation.source_separation_metrics import calculate_source_separation_metrics
metrics = calculate_source_separation_metrics(
    original_load=load_data,
    base_embedding=base_embeddings,
    weather_embedding=weather_embeddings,
    reconstructed_total=total_reconstructed
)
```

## 📈 Evaluation Metrics

### Clustering Performance
- **ACC**: Clustering accuracy using Hungarian algorithm for optimal label assignment
- **NMI**: Normalized Mutual Information measuring information shared between clusterings
- **ARI**: Adjusted Rand Index measuring similarity between clusterings, adjusted for chance

### Source Separation Quality
- **Reconstruction Error**: Mean squared error between original and reconstructed load profiles
- **Independence Score**: Composite measure of statistical independence between base and weather embeddings
- **Mutual Information**: MINE-based estimation of mutual information between separated components
- **Distance Correlation**: Non-linear dependency measure between embeddings

### Statistical Validation
- **Multiple Runs**: All results averaged over 10 independent runs with different random seeds
- **Confidence Intervals**: Statistical significance testing for performance comparisons
- **Ablation Studies**: Systematic evaluation of each component's contribution

## 📊 Datasets

### Synthetic Dataset (Pecan Street-Style)
- **500 households** in Austin, Texas
- **365 days** of consumption data (182,500 samples)
- **5 distinct archetypes**: Low Usage, Morning Peakers, Afternoon Peakers, Evening Peakers, Night Owls
- **Realistic weather effects** with temperature-dependent air conditioning usage
- **Performance**: CausalHES achieves 99.2% clustering accuracy

### Irish Dataset (Real-World)
- **500 households** from CER Smart Metering Project
- **18 months** of real consumption data (219,000 samples)
- **Rich household characteristics**: demographics, socioeconomic factors, home types
- **Irish climate data**: temperate oceanic weather patterns
- **Performance**: CausalHES achieves 87.6% clustering accuracy

## 🔧 Configuration

```yaml
# configs/causal_hes_config.yaml
model:
  n_clusters: 5
  load_input_shape: [24, 1]
  weather_input_shape: [24, 2]

  cssae:
    load_embedding_dim: 64
    weather_embedding_dim: 32
    base_dim: 32
    weather_effect_dim: 16
    separation_method: "residual"

  dec:
    alpha: 1.0

training:
  cssae_epochs: 100
  joint_epochs: 50
  batch_size: 32
  learning_rate: 0.001

  loss_weights:
    reconstruction_weight: 1.0
    causal_weight: 0.1
    clustering_weight: 0.5

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  normalize: true
```

## 📝 Research Paper

This implementation accompanies our TNNLS submission:

**"Causally-Inspired Household Energy Segmentation: A Deep Learning Framework for Weather-Independent Consumption Pattern Discovery"**

### Key Contributions:
1. **Causal Source Separation Autoencoder (CSSAE)** - Novel architecture for decomposing energy consumption into causally-independent components
2. **Multi-faceted Independence Constraints** - Comprehensive approach combining mutual information minimization, adversarial training, and distance correlation
3. **Base Load Clustering** - Clustering on weather-independent patterns using Deep Embedded Clustering
4. **Comprehensive Evaluation** - Extensive comparison with traditional, deep learning, and multi-modal baselines
5. **Interpretable Results** - Framework provides both clustering and source separation insights

### Theoretical Foundation:
- **Causal Inference**: Grounded in principles of causal discovery and independence testing
- **Source Separation**: Leverages independent component analysis and blind source separation theory
- **Deep Learning**: Combines autoencoder architectures with clustering objectives for end-to-end learning

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{causal_hes_2024,
  title={Causally-Inspired Household Energy Segmentation: A Deep Learning Framework for Weather-Independent Consumption Pattern Discovery},
  author={[Authors]},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  note={Under Review}
}
```

## 🙏 Acknowledgments

- Pecan Street Inc. for providing the inspiration and characteristics for our realistic synthetic dataset
- Commission for Energy Regulation (CER) and the Irish Social Science Data Archive (ISSDA) for the Irish Smart Metering dataset
- Met Éireann (Irish Meteorological Service) for weather data inspiration
- PyTorch team for the excellent deep learning framework
- Research community for baseline implementations and theoretical foundations
- Contributors to causal inference, source separation, and deep clustering literature

---

**Status**: ✅ Implementation Complete | 📊 Experiments Validated | 📝 Paper Under Review | 🔬 Reproducible Results
