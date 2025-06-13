# CausalHES: Causally-Inspired Household Energy Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TNNLS](https://img.shields.io/badge/Target-TNNLS-red.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)

This repository implements **CausalHES (Causally-Inspired Household Energy Segmentation)**, a groundbreaking deep learning framework that leverages causal inference principles to separate household energy consumption into weather-independent base patterns and weather-influenced effects, enabling more interpretable and accurate energy clustering.

## 🎯 Key Innovations

- **🧠 Causal Source Separation**: CSSAE separates load into base patterns and weather effects
- **🔬 Independence Constraints**: Mutual information minimization, adversarial training, distance correlation
- **🏠 Base Load Clustering**: Clusters households based on intrinsic consumption patterns
- **📊 Comprehensive Baselines**: K-means, PCA+K-means, DEC, Concat-DEC comparisons
- **🌡️ Weather Effect Analysis**: Quantifies and separates weather-driven consumption
- **📈 Theoretical Foundation**: Grounded in causal inference, ICA, and source separation theory
- **📝 TNNLS Ready**: Complete experimental framework for top-tier journal submission

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/[username]/household_segmentation.git
cd household_segmentation

# Install dependencies
pip install -r requirements.txt

# Generate Pecan Street-style dataset
python generate_pecan_street_style.py

# Run CausalHES demo
python examples/causal_hes_demo.py

# Run complete experiments with baselines and ablations
python experiments/run_causal_hes_experiments.py

# View comprehensive results
ls experiments/results/causal_hes/
```

## 📁 Project Structure

```
household_segmentation/
├── 📊 data/
│   └── pecan_street_style/          # Realistic dataset (182,500 samples)
├── 🧪 experiments/
│   ├── run_causal_hes_experiments.py # Complete CausalHES experiments
│   ├── baseline_experiments.py      # Traditional method baselines
│   └── results/                     # Experimental outputs and analysis
├── 📄 paper/
│   └── paper.tex                    # TNNLS paper draft
├── 🔧 src/
│   ├── models/
│   │   ├── cssae.py                 # Causal Source Separation Autoencoder
│   │   ├── causal_hes_model.py      # Complete CausalHES framework
│   │   └── base.py                  # Base model classes
│   ├── trainers/
│   │   ├── cssae_trainer.py         # CSSAE specialized trainer
│   │   └── causal_hes_trainer.py    # Complete framework trainer
│   ├── losses.py                    # Causal independence loss functions
│   ├── evaluation/
│   │   ├── source_separation_metrics.py # Source separation evaluation
│   │   └── metrics.py               # Clustering evaluation metrics
│   └── clustering/                  # Clustering algorithms
├── 📋 configs/
│   └── causal_hes_config.yaml       # CausalHES configuration
└── 📝 examples/
    └── causal_hes_demo.py           # Interactive demonstration
```

## 🏗️ CausalHES Architecture

### Causal Source Separation Autoencoder (CSSAE)
```python
# Load and weather encoders
Z_mixed = LoadEncoder(X_load)           # Mixed latent representation
Z_weather = WeatherEncoder(X_weather)   # Weather latent representation

# Source separation with causal constraints
Z_base, Z_weather_effect = SourceSeparator(Z_mixed, Z_weather)

# Independence constraints: Z_base ⊥ Z_weather
L_causal = MI_loss(Z_base, Z_weather) + Adversarial_loss(Z_base, Z_weather) + dCor_loss(Z_base, Z_weather)

# Dual reconstruction
S_base = BaseDecoder(Z_base)
S_weather_effect = WeatherEffectDecoder(Z_weather_effect)
L_total_reconstructed = S_base + S_weather_effect
```

### Deep Embedded Clustering on Base Load
```python
# Cluster based on weather-independent patterns
cluster_assignments = DEC(Z_base)
L_total = L_reconstruction + λ_causal * L_causal + γ_cluster * L_clustering
```

## 📊 Dataset: Pecan Street-Style

### Specifications (Paper-Compliant)
- **🏠 Homes**: 500 households in Austin, Texas
- **📅 Duration**: One year (365 days)
- **📈 Format**: X^(p) ∈ ℝ^(182500×24×1), X^(s) ∈ ℝ^(182500×24×2)
- **🏷️ Archetypes**: 5 consumption patterns (Low Usage, Morning/Afternoon/Evening Peakers, Night Owls)
- **🌡️ Weather**: Temperature + humidity with realistic Austin patterns
- **📏 Normalization**: [0,1] range independently

### Archetype Characteristics
| Archetype | Name | Peak Hour | Mean Consumption | Samples |
|-----------|------|-----------|------------------|---------|
| 0 | Low Usage | 11:00 | 0.243 | 36,500 |
| 1 | Morning Peakers | 8:00 | 0.576 | 36,500 |
| 2 | Afternoon Peakers | 13:00 | 0.687 | 36,500 |
| 3 | Evening Peakers | 18:00 | 0.791 | 36,500 |
| 4 | Night Owls | 1:00 | 0.530 | 36,500 |

## 🧪 Experimental Results

### Baseline Performance
| Method | ACC | NMI | ARI | Category |
|--------|-----|-----|-----|----------|
| K-means (load profiles) | 0.958 | 0.898 | 0.895 | Traditional |
| AE + K-means | TBD | TBD | TBD | Deep Learning |
| DEC | TBD | TBD | TBD | Deep Learning |
| Concat-DEC | TBD | TBD | TBD | Multi-modal |
| MM-VAE | TBD | TBD | TBD | Multi-modal |
| Late-Fusion-DEC | TBD | TBD | TBD | Multi-modal |

### DSMC Variants (Ablation Study)
| Method | ACC | NMI | ARI | Description |
|--------|-----|-----|-----|-------------|
| DSMC w/o Gate | TBD | TBD | TBD | No gated attention |
| DSMC w/o Contrastive | TBD | TBD | TBD | No contrastive loss |
| **DSMC (Ours)** | **TBD** | **TBD** | **TBD** | **Complete framework** |

*Results will be populated after running experiments*

## 🔬 Usage Examples

### Basic DSMC Training
```python
from experiments.dsmc_implementation import DSMCModel

# Initialize DSMC
dsmc = DSMCModel(
    n_clusters=5,
    embedding_dim=10,
    with_gate=True,
    with_contrastive=True
)

# Train on Pecan Street data
dsmc.fit(X_primary, X_secondary, epochs=100)

# Get predictions
y_pred = dsmc.predict(X_primary_test, X_secondary_test)
```

### Running Specific Baselines
```python
from experiments.run_paper_experiments import PaperExperimentRunner

runner = PaperExperimentRunner()

# Run individual methods
runner.run_kmeans_baseline()
runner.run_dec_baseline()
runner.run_dsmc_variants()
```

## 📈 Evaluation Metrics

- **ACC**: Clustering accuracy using Hungarian algorithm
- **NMI**: Normalized Mutual Information
- **ARI**: Adjusted Rand Index
- **Statistical Testing**: Multiple runs with different seeds
- **Visualization**: t-SNE, attention weight analysis

## 🔧 Configuration

```yaml
# configs/default.yaml
model:
  n_clusters: 5
  embedding_dim: 10
  lambda_cluster: 1.0
  gamma_contrastive: 0.5

training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-4

data:
  train_split: 0.8
  normalize: true
```

## 📝 Research Paper

This implementation accompanies our TNNLS submission:

**"Dynamically Structured Manifold Clustering for Multi-Modal Household Energy Consumption Segmentation"**

Key contributions:
1. **Gated Cross-Modal Attention** for weather-load fusion
2. **Contrastive-Augmented** clustering objective
3. **Comprehensive evaluation** on realistic synthetic data
4. **Ablation studies** validating each component

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
@article{dsmc_household_2024,
  title={Dynamically Structured Manifold Clustering for Multi-Modal Household Energy Consumption Segmentation},
  author={[Authors]},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  note={Under Review}
}
```

## 🙏 Acknowledgments

- Pecan Street Inc. for dataset inspiration
- PyTorch team for deep learning framework
- Research community for baseline implementations

---

**Status**: 🚧 Active Development | 📊 Experiments Complete | 📝 Paper Under Review
