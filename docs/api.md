# API Reference

This document provides detailed API documentation for the CausalHES framework.

## Core Models

### CausalHESModel

The main model implementing the Causal Source Separation Autoencoder.

```python
from src.models.causal_hes_model import CausalHESModel

model = CausalHESModel(
    load_input_shape=(24, 1),
    weather_input_shape=(24, 2),
    n_clusters=4,
    base_dim=32,
    weather_effect_dim=16,
    embedding_dim=64
)
```

#### Parameters
- `load_input_shape` (tuple): Shape of load profile input (timesteps, features)
- `weather_input_shape` (tuple): Shape of weather input (timesteps, features)
- `n_clusters` (int): Number of clusters for segmentation
- `base_dim` (int): Dimension of base load embedding
- `weather_effect_dim` (int): Dimension of weather effect embedding
- `embedding_dim` (int): Total embedding dimension

#### Methods

##### `forward(x_load, x_weather=None)`
Forward pass through the model.

**Parameters:**
- `x_load` (torch.Tensor): Load profile data [batch_size, timesteps, features]
- `x_weather` (torch.Tensor, optional): Weather data [batch_size, timesteps, features]

**Returns:**
Dictionary containing:
- `load_reconstruction`: Reconstructed load profiles
- `base_embedding`: Weather-independent base embeddings
- `weather_effect_pred`: Predicted weather effects
- `weather_embedding`: Weather embeddings (if weather data provided)
- `cluster_probs`: Cluster assignment probabilities
- `mixed_embedding`: Mixed load embeddings

## Loss Functions

### CSSAELoss

Complete loss function for CSSAE training.

```python
from src.losses.cssae_loss import CSSAELoss

loss_fn = CSSAELoss(
    base_dim=32,
    weather_dim=16,
    lambda_causal=0.1,
    lambda_cluster=0.5
)
```

#### Parameters
- `base_dim` (int): Dimension of base load embeddings
- `weather_dim` (int): Dimension of weather embeddings
- `lambda_causal` (float): Weight for causal independence loss
- `lambda_cluster` (float): Weight for clustering loss

#### Methods

##### `forward(y_true, model_outputs, stage="pretrain", target_distribution=None)`
Compute complete CSSAE loss.

**Parameters:**
- `y_true` (torch.Tensor): Original load profiles
- `model_outputs` (dict): Dictionary from CausalHESModel.forward()
- `stage` (str): Training stage ("pretrain" or "joint")
- `target_distribution` (torch.Tensor, optional): Target distribution for clustering

**Returns:**
Dictionary containing all loss components.

### CompositeCausalLoss

Multi-component causal independence loss.

```python
from src.losses.composite_causal_loss import CompositeCausalLoss

causal_loss = CompositeCausalLoss(
    base_dim=32,
    weather_dim=16,
    alpha_mi=1.0,
    alpha_adv=0.5,
    alpha_dcor=0.3
)
```

## Training

### CausalHESTrainer

Complete trainer implementing the two-stage training strategy.

```python
from src.trainers.causal_hes_complete_trainer import CausalHESTrainer

trainer = CausalHESTrainer(
    model=model,
    loss_fn=loss_fn,
    device='cuda',
    learning_rate=1e-3
)
```

#### Methods

##### `stage1_pretrain(train_loader, val_loader=None, epochs=50, patience=10)`
Stage 1: CSSAE pre-training for causal source separation.

**Parameters:**
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader, optional): Validation data loader
- `epochs` (int): Number of training epochs
- `patience` (int): Early stopping patience

**Returns:**
Dictionary with training results.

##### `stage2_joint_training(train_loader, val_loader=None, epochs=30, patience=10)`
Stage 2: Joint training for causal clustering.

**Parameters:**
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader, optional): Validation data loader
- `epochs` (int): Number of training epochs
- `patience` (int): Early stopping patience

**Returns:**
Dictionary with training results and cluster assignments.

## Data Processing

### IrishDatasetProcessor

Processor for Irish CER Smart Metering dataset.

```python
from src.data.irish_dataset_processor import IrishDatasetProcessor

processor = IrishDatasetProcessor(
    data_dir="data/Irish",
    random_state=42
)
```

#### Methods

##### `process_irish_dataset(n_households=1000, n_clusters=4, normalize=True)`
Process the complete Irish dataset for CausalHES experiments.

**Parameters:**
- `n_households` (int): Number of households to process
- `n_clusters` (int): Number of clusters for ground truth
- `normalize` (bool): Whether to normalize the data

**Returns:**
Dictionary containing processed data and metadata.

## Evaluation

### ClusteringMetrics

Comprehensive clustering evaluation metrics.

```python
from src.evaluation.metrics import ClusteringMetrics

metrics = ClusteringMetrics()
results = metrics.evaluate_clustering(true_labels, predicted_labels, embeddings)
```

### ClusteringEvaluator

Complete evaluation framework for clustering experiments.

```python
from src.evaluation.evaluator import ClusteringEvaluator

evaluator = ClusteringEvaluator()
results = evaluator.evaluate_method(method, data, true_labels)
```

## Utilities

### Configuration

```python
from src.utils.config import load_config

config = load_config("configs/causal_hes_config.yaml")
```

### Logging

```python
from src.utils.logging import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)
```

### Preprocessing

```python
from src.utils.preprocessing import normalize_data

normalized_data = normalize_data(data, method='minmax')
```

## Examples

### Basic Usage

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.causal_hes_model import CausalHESModel
from src.losses.cssae_loss import CSSAELoss
from src.trainers.causal_hes_complete_trainer import CausalHESTrainer

# Create synthetic data
load_data = torch.randn(1000, 24, 1)
weather_data = torch.randn(1000, 24, 2)

# Create data loader
dataset = TensorDataset(load_data, weather_data)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and loss
model = CausalHESModel(
    load_input_shape=(24, 1),
    weather_input_shape=(24, 2),
    n_clusters=4
)

loss_fn = CSSAELoss(base_dim=32, weather_dim=16)

# Train model
trainer = CausalHESTrainer(model, loss_fn)
pretrain_results = trainer.stage1_pretrain(train_loader, epochs=30)
joint_results = trainer.stage2_joint_training(train_loader, epochs=20)
```

### Advanced Configuration

```python
# Custom model configuration
model = CausalHESModel(
    load_input_shape=(24, 1),
    weather_input_shape=(24, 2),
    n_clusters=5,
    base_dim=64,
    weather_effect_dim=32,
    embedding_dim=128
)

# Custom loss configuration
loss_fn = CSSAELoss(
    base_dim=64,
    weather_dim=32,
    lambda_causal=0.2,
    lambda_cluster=0.8,
    alpha_mi=1.5,
    alpha_adv=0.8,
    alpha_dcor=0.5
)

# Custom training configuration
trainer = CausalHESTrainer(
    model=model,
    loss_fn=loss_fn,
    device='cuda',
    learning_rate=5e-4,
    weight_decay=1e-4
)
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameter values
- `RuntimeError`: Model training errors
- `ImportError`: Missing dependencies
- `FileNotFoundError`: Missing data files

### Example Error Handling

```python
try:
    model = CausalHESModel(
        load_input_shape=(24, 1),
        weather_input_shape=(24, 2),
        n_clusters=4
    )
    outputs = model(load_data, weather_data)
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

## Performance Tips

### GPU Usage

```python
# Check GPU availability
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move model to GPU
model = model.to(device)
data = data.to(device)
```

### Memory Optimization

```python
# Use smaller batch sizes
train_loader = DataLoader(dataset, batch_size=16)

# Enable gradient checkpointing (if implemented)
model.gradient_checkpointing = True

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(data)
```

### Training Optimization

```python
# Use learning rate scheduling
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Enable early stopping
trainer.stage1_pretrain(train_loader, patience=10)
```