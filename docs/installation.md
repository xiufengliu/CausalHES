# Installation Guide

This guide provides detailed instructions for installing CausalHES on different platforms.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

### Recommended Requirements
- **Python**: 3.9 or 3.10
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 5GB free space

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/CausalHES.git
cd CausalHES

# Create virtual environment
python -m venv causalhes_env
source causalhes_env/bin/activate  # On Windows: causalhes_env\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# For GPU support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CausalHES in development mode
pip install -e .
```

### Method 2: Conda Installation

```bash
# Create conda environment
conda create -n causalhes python=3.9
conda activate causalhes

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Clone and install CausalHES
git clone https://github.com/your-username/CausalHES.git
cd CausalHES
pip install -r requirements.txt
pip install -e .
```

### Method 3: Docker Installation

```bash
# Pull the Docker image (when available)
docker pull causalhes/causalhes:latest

# Or build from source
git clone https://github.com/your-username/CausalHES.git
cd CausalHES
docker build -t causalhes .

# Run container
docker run -it --gpus all -v $(pwd):/workspace causalhes
```

## Platform-Specific Instructions

### Windows

1. **Install Python**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure "Add Python to PATH" is checked

2. **Install Git**
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **Install Visual Studio Build Tools** (if needed)
   - Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

4. **Follow Quick Install steps above**

### macOS

1. **Install Homebrew** (if not installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python and Git**
   ```bash
   brew install python git
   ```

3. **Follow Quick Install steps above**

### Linux (Ubuntu/Debian)

1. **Update system packages**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv git
   ```

2. **Install build essentials** (if needed)
   ```bash
   sudo apt install build-essential
   ```

3. **Follow Quick Install steps above**

### Linux (CentOS/RHEL)

1. **Install Python and Git**
   ```bash
   sudo yum install python3 python3-pip git
   # or for newer versions
   sudo dnf install python3 python3-pip git
   ```

2. **Follow Quick Install steps above**

## GPU Support

### NVIDIA GPU Setup

1. **Install NVIDIA drivers**
   - Download from [NVIDIA website](https://www.nvidia.com/drivers/)

2. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

3. **Verify installation**
   ```bash
   nvidia-smi
   nvcc --version
   ```

4. **Install PyTorch with CUDA**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Test GPU availability**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   ```

## Verification

### Quick Test

```bash
# Run tests
python run_tests.py

# Run demo
python examples/complete_causal_hes_demo.py
```

### Import Test

```python
# Test imports
import torch
from src.models.causal_hes_model import CausalHESModel
from src.losses.cssae_loss import CSSAELoss

print("✅ All imports successful!")
```

### Model Test

```python
# Test model creation
model = CausalHESModel(
    load_input_shape=(24, 1),
    weather_input_shape=(24, 2),
    n_clusters=4
)

# Test forward pass
import torch
load_data = torch.randn(8, 24, 1)
weather_data = torch.randn(8, 24, 2)
outputs = model(load_data, weather_data)

print("✅ Model test successful!")
```

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation Issues

**Problem**: PyTorch not installing or CUDA not detected

**Solutions**:
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Install specific version
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# For CUDA issues, check compatibility
python -c "import torch; print(torch.version.cuda)"
```

#### 2. Import Errors

**Problem**: `ModuleNotFoundError` when importing CausalHES modules

**Solutions**:
```bash
# Ensure you're in the right directory
cd /path/to/CausalHES

# Install in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 3. Memory Issues

**Problem**: Out of memory errors during training

**Solutions**:
- Reduce batch size in configuration
- Use gradient accumulation
- Enable mixed precision training
- Use CPU training for small datasets

#### 4. Permission Issues (Linux/macOS)

**Problem**: Permission denied errors

**Solutions**:
```bash
# Fix permissions
chmod +x scripts/*.sh

# Use user installation
pip install --user -r requirements.txt
```

### Getting Help

1. **Check existing issues**: [GitHub Issues](https://github.com/your-username/CausalHES/issues)
2. **Create new issue**: Include system info and error messages
3. **Contact maintainers**: For urgent issues

### System Information

To help with troubleshooting, collect system information:

```python
import sys
import torch
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Next Steps

After successful installation:

1. **Read the documentation**: [API Reference](api.md)
2. **Try examples**: [Examples Guide](examples.md)
3. **Configure settings**: [Configuration Guide](configuration.md)
4. **Start training**: [Training Guide](training.md)

## Development Installation

For contributors:

```bash
# Clone repository
git clone https://github.com/your-username/CausalHES.git
cd CausalHES

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
python run_tests.py
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed development guidelines.