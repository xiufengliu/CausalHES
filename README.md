# CausalHES: Causal Disentanglement for Household Energy Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CausalHES** is an open-source deep learning framework for weather-independent household energy pattern discovery. It reformulates household energy segmentation as a causal source separation problem, using a Causal Source Separation Autoencoder (CSSAE) to disentangle weather-independent base consumption from weather-dependent effects, enabling more robust and interpretable household behavioral clustering.

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

## 📁 Project Structure

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
│   │   ├── causal_hes_model.py      # Complete CausalHES framework
│   │   └── base.py                  # Base model classes
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

## 🔧 Configuration

Key configuration options in `configs/causal_hes_config.yaml`:

```yaml
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

training:
  cssae_epochs: 100
  joint_epochs: 50
  batch_size: 32
  learning_rate: 0.001

  loss_weights:
    reconstruction_weight: 1.0
    causal_weight: 0.1
    clustering_weight: 0.5
```

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
    cssae_epochs=100,
    joint_epochs=50
)

# Get predictions and source separation results
predictions = model.predict(load_test, weather_test)
separation_results = model.get_source_separation_results(load_test, weather_test)
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

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Irish Commission for Energy Regulation (CER) for providing the smart meter dataset
- Open-source community for foundational tools and libraries

## 📧 Contact

For questions, issues, or collaborations:

- **Primary Contact**: [Xiufeng Liu](mailto:xiuli@dtu.dk)
- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
