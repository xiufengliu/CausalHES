# CausalHES: Causal Disentanglement for Household Energy Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](./tests/)

**CausalHES** is a state-of-the-art deep learning framework for weather-independent household energy pattern discovery. It reformulates household energy segmentation as a causal source separation problem, using a novel Causal Source Separation Autoencoder (CSSAE) to disentangle weather-independent base consumption from weather-dependent effects.

## 🎯 Key Features

- **🧠 Causal Source Separation Autoencoder (CSSAE)**: Novel architecture that decomposes observed load into weather-independent base consumption and weather-dependent effects
- **🔬 Composite Independence Loss**: Combines mutual information minimization (MINE), adversarial training, and distance correlation penalties  
- **🏠 Weather-Independent Clustering**: Performs Deep Embedded Clustering (DEC) exclusively on purified base load embeddings
- **🌡️ Semantic Validation**: Correlation between reconstructed weather effects and actual temperature
- **📈 Theoretical Foundation**: Grounded in causal inference principles with theoretical guarantees

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/CausalHES.git
cd CausalHES

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (if not already installed)
pip install torch torchvision torchaudio
```

### Basic Usage

```python
from src.models.causal_hes_model import CausalHESModel
from src.losses.cssae_loss import CSSAELoss
from src.trainers.causal_hes_complete_trainer import CausalHESTrainer

# Initialize model
model = CausalHESModel(
    load_input_shape=(24, 1),      # 24-hour load profiles
    weather_input_shape=(24, 2),   # Temperature + humidity
    n_clusters=4,
    base_dim=32,
    weather_effect_dim=16,
    embedding_dim=64
)

# Initialize loss function
loss_fn = CSSAELoss(
    base_dim=32,
    weather_dim=16,
    lambda_causal=0.1,
    lambda_cluster=0.5
)

# Train with two-stage strategy
trainer = CausalHESTrainer(model, loss_fn)

# Stage 1: CSSAE Pre-training
pretrain_results = trainer.stage1_pretrain(
    train_loader=train_loader,
    epochs=50
)

# Stage 2: Joint Training with Clustering
joint_results = trainer.stage2_joint_training(
    train_loader=train_loader,
    epochs=30
)
```

### Run Complete Demo

```bash
# Run the complete demonstration
python examples/complete_causal_hes_demo.py
```

This will:
1. Generate synthetic household energy data
2. Train the CausalHES model using the two-stage strategy
3. Evaluate clustering performance
4. Visualize source separation results

## 📊 Methodology

### Causal Decomposition

CausalHES models observed household energy consumption as:

```
x^(l) = s_base + s_weather + ε
```

Where:
- `x^(l)`: Observed load profile
- `s_base`: Weather-independent base consumption (target for clustering)
- `s_weather`: Weather-dependent effects
- `ε`: Residual noise

### Key Innovation: Weather Independence

The framework enforces the causal assumption `s_base ⊥ x^(w)` through a composite independence loss:

```
L_causal = α_MI * L_MI + α_adv * L_adv + α_dcor * L_dcor
```

Combining:
- **MINE**: Mutual information minimization
- **Adversarial**: GAN-style independence training  
- **Distance Correlation**: Statistical dependence penalty

### Two-Stage Training

1. **Stage 1**: CSSAE pre-training for source separation
2. **Stage 2**: Joint optimization with clustering objective

## 🏗️ Architecture

```
Input: Load Profiles + Weather Data
         ↓
    CNN Encoders (1D Convolution)
         ↓
   Source Separation Module
    ↙              ↘
Base Load      Weather Effect
Embedding      Embedding
    ↓              ↓
DEC Clustering  Reconstruction
```

## 📁 Project Structure

```
CausalHES/
├── src/
│   ├── models/              # Neural network models
│   │   ├── causal_hes_model.py    # Main CausalHES model
│   │   ├── base.py               # Base autoencoder
│   │   └── clustering_layers.py  # DEC clustering layer
│   ├── losses/              # Loss functions
│   │   ├── cssae_loss.py         # Complete CSSAE loss
│   │   └── composite_causal_loss.py  # Causal independence loss
│   ├── trainers/            # Training algorithms
│   │   └── causal_hes_complete_trainer.py  # Two-stage trainer
│   ├── data/                # Data processing
│   │   └── irish_dataset_processor.py  # Irish dataset processor
│   ├── evaluation/          # Evaluation metrics
│   ├── clustering/          # Baseline clustering methods
│   └── utils/               # Utilities
├── examples/                # Example scripts
│   └── complete_causal_hes_demo.py  # Complete demonstration
├── tests/                   # Unit tests
├── configs/                 # Configuration files
├── paper/                   # Research paper and figures
└── docs/                    # Documentation
```

## 🧪 Testing

```bash
# Run all tests
python run_tests.py

# Run specific test
python -m pytest tests/test_models.py -v

# Run with coverage
python run_tests.py --coverage
```

## 📈 Performance

CausalHES achieves superior clustering performance by:

- **Weather Independence**: Removes weather confounding effects
- **Temporal Modeling**: CNN-based encoders capture temporal patterns
- **Multi-objective Optimization**: Balances reconstruction, independence, and clustering
- **Semantic Validation**: Weather effects correlate with actual temperature

### Baseline Comparisons

The framework includes implementations of 15+ baseline methods:
- Traditional: K-means, SAX K-means, Two-stage K-means
- Deep Learning: DEC, VAE-based clustering
- Multi-modal: Attention-fused, contrastive learning
- Causal: Doubly robust, instrumental variables

## 🔬 Research Paper

This implementation accompanies the research paper:

**"CausalHES: Causal Disentanglement for Household Energy Segmentation"**

Key contributions:
1. Novel causal source separation architecture
2. Composite independence loss for weather disentanglement
3. Theoretical analysis of identifiability conditions
4. Comprehensive empirical evaluation

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Training Guide](docs/training.md)
- [Configuration](docs/configuration.md)
- [Examples](docs/examples.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/CausalHES.git
cd CausalHES

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
python run_tests.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Authors**: CausalHES Research Team
- **Email**: research@causalhes.org
- **Paper**: [arXiv:2024.xxxxx](https://arxiv.org/abs/2024.xxxxx)
- **Issues**: [GitHub Issues](https://github.com/your-username/CausalHES/issues)

## 🙏 Acknowledgments

- Irish Commission for Energy Regulation (CER) for the Smart Metering dataset
- PyTorch team for the deep learning framework
- scikit-learn for machine learning utilities

## 📊 Citation

If you use CausalHES in your research, please cite:

```bibtex
@article{causalhes2024,
  title={CausalHES: Causal Disentanglement for Household Energy Segmentation},
  author={CausalHES Research Team},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```

---

**⭐ Star this repository if you find it useful!**