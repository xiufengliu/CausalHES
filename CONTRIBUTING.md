# Contributing to CausalHES

We welcome contributions to the CausalHES project! This document provides guidelines for contributing to the codebase.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Git

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/CausalHES.git
   cd CausalHES
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install development tools**
   ```bash
   pip install black flake8 pytest pytest-cov pre-commit
   pre-commit install
   ```

5. **Run tests to ensure everything works**
   ```bash
   python run_tests.py
   ```

## 🔧 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 2. Make Changes

- Follow the existing code style and conventions
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Code Quality

We use several tools to maintain code quality:

#### Code Formatting
```bash
# Format code with black
black src/ tests/ examples/

# Check formatting
black --check src/ tests/ examples/
```

#### Linting
```bash
# Run flake8 for style checking
flake8 src/ tests/ examples/
```

#### Testing
```bash
# Run all tests
python run_tests.py

# Run specific test file
python -m pytest tests/test_models.py -v

# Run with coverage
python run_tests.py --coverage
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
# or
git commit -m "fix: resolve issue description"
```

**Commit Message Convention:**
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `style:` Code style changes
- `chore:` Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📝 Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write clear, descriptive variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular

### Documentation

- Update README.md if adding new features
- Add docstrings following Google/NumPy style
- Include examples in docstrings when helpful
- Update configuration documentation for new parameters

### Testing

- Write unit tests for new functionality
- Ensure tests cover edge cases
- Mock external dependencies appropriately
- Aim for high test coverage (>80%)

### Example Test Structure

```python
import pytest
import torch
from src.models.causal_hes_model import CausalHESModel

class TestCausalHESModel:
    def test_initialization(self):
        """Test model initialization."""
        model = CausalHESModel(
            load_input_shape=(24, 1),
            weather_input_shape=(24, 2),
            n_clusters=4
        )
        assert model.n_clusters == 4
        
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        model = CausalHESModel(
            load_input_shape=(24, 1),
            weather_input_shape=(24, 2),
            n_clusters=4
        )
        
        load_data = torch.randn(8, 24, 1)
        weather_data = torch.randn(8, 24, 2)
        
        outputs = model(load_data, weather_data)
        
        assert 'base_embedding' in outputs
        assert 'weather_embedding' in outputs
        assert outputs['base_embedding'].shape == (8, 32)
```

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment information**
   - Python version
   - PyTorch version
   - Operating system
   - Hardware (CPU/GPU)

2. **Steps to reproduce**
   - Minimal code example
   - Expected behavior
   - Actual behavior
   - Error messages/stack traces

3. **Additional context**
   - Screenshots if applicable
   - Related issues or discussions

### Feature Requests

When requesting features:

1. **Describe the motivation**
   - What problem does it solve?
   - How does it improve the project?

2. **Provide implementation details**
   - Proposed API changes
   - Examples of usage
   - Potential challenges

3. **Consider alternatives**
   - Other approaches considered
   - Why this approach is preferred

## 🔬 Research Contributions

### Algorithmic Improvements

- Provide theoretical justification
- Include experimental validation
- Compare against existing baselines
- Document computational complexity

### New Baseline Methods

- Implement following the base class interface
- Add comprehensive tests
- Include in evaluation framework
- Update documentation

### Dataset Integration

- Follow existing data processor patterns
- Include data validation
- Add preprocessing utilities
- Provide usage examples

## 📊 Performance Considerations

### Optimization Guidelines

- Profile code for bottlenecks
- Use vectorized operations when possible
- Consider memory usage for large datasets
- Implement efficient data loading

### GPU Compatibility

- Ensure CUDA compatibility
- Handle device placement correctly
- Test on both CPU and GPU
- Document hardware requirements

## 🧪 Experimental Features

### Adding New Models

1. Inherit from appropriate base class
2. Implement required abstract methods
3. Add comprehensive tests
4. Include usage examples
5. Update model registry

### New Loss Functions

1. Follow existing loss function patterns
2. Implement both forward and backward passes
3. Add gradient checking tests
4. Document mathematical formulation

## 📚 Documentation Standards

### Docstring Format

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    algorithm, or important implementation details.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        RuntimeError: When computation fails
        
    Example:
        >>> result = example_function(5, "test")
        >>> print(result)
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    
    # Implementation here
    return True
```

### README Updates

When adding features that affect user experience:

- Update installation instructions if needed
- Add usage examples
- Update feature list
- Include performance notes

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn
- Focus on technical merit

### Communication

- Use clear, professional language
- Provide context for discussions
- Reference relevant issues/PRs
- Be patient with response times

## 🏆 Recognition

Contributors will be acknowledged in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- Research paper acknowledgments (for algorithmic contributions)

## ❓ Questions?

- Open an issue for technical questions
- Use discussions for general questions
- Contact maintainers for sensitive issues

Thank you for contributing to CausalHES! 🎉