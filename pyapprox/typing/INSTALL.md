# Installation Instructions for PyApprox Typing Module

## Quick Start

### Using pip

```bash
cd pyapprox/typing
pip install -r requirements.txt
```

### Using conda

```bash
cd pyapprox/typing
conda env create -f environment.yml
conda activate pyapprox-typing
```

## Dependencies

### Core Runtime Dependencies

- **Python** >= 3.9
- **NumPy** >= 1.20.0 - NumPy backend support
- **SciPy** >= 1.7.0 - Optimization routines
- **PyTorch** >= 2.0.0 - PyTorch backend support

### Development Dependencies

- **pytest** >= 7.0.0 - Testing framework
- **pytest-cov** >= 3.0.0 - Code coverage
- **mypy** >= 1.0.0 - Static type checking
- **numpy-stubs** - Type stubs for NumPy

### Documentation Dependencies

- **jupyter-cache** >= 1.0.0 - Caching for Quarto
- **quartodoc** >= 0.6.0 - API documentation generation
- **Quarto** >= 1.4.0 - Documentation rendering (separate install)

## Backend-Specific Setup

### NumPy Backend Only

If you only need NumPy backend support:

```bash
pip install numpy>=1.20.0 scipy>=1.7.0
```

### PyTorch Backend Only

If you only need PyTorch backend support:

```bash
pip install torch>=2.0.0 scipy>=1.7.0
```

## Quarto Documentation

To build the Quarto documentation:

1. Install Quarto: https://quarto.org/docs/get-started/

2. Install Python dependencies:
```bash
pip install jupyter-cache quartodoc
```

3. Render documentation:
```bash
cd pyapprox/typing/quarto-docs
quarto render
```

4. Preview documentation:
```bash
quarto preview
```

## Verification

### Test Installation

```bash
python -c "from pyapprox.typing.util.backends.numpy import NumpyBkd; print('NumPy backend OK')"
python -c "from pyapprox.typing.util.backends.torch import TorchBkd; print('PyTorch backend OK')"
```

### Run Tests

```bash
# From pyapprox root
python -m pytest pyapprox/typing/surrogates/kernels/tests/
python -m pytest pyapprox/typing/surrogates/gaussianprocess/tests/
```

Or with conda:

```bash
conda run -n pyapprox-typing python -m pytest pyapprox/typing/surrogates/kernels/tests/
conda run -n pyapprox-typing python -m pytest pyapprox/typing/surrogates/gaussianprocess/tests/
```

### Type Checking

```bash
mypy pyapprox/typing/surrogates/kernels/
mypy pyapprox/typing/surrogates/gaussianprocess/
```

## Troubleshooting

### PyTorch Installation Issues

If you have GPU support requirements, install PyTorch separately:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

See https://pytorch.org/get-started/locally/ for platform-specific instructions.

### Jupyter Cache Issues

If Quarto freezing/caching fails:

```bash
pip install --upgrade jupyter-cache nbclient nbformat
```

### MyPy Issues

If mypy can't find numpy stubs:

```bash
pip install --upgrade numpy-stubs types-setuptools
```

## Development Setup

For development work on the typing module:

```bash
# Create environment
conda env create -f environment.yml
conda activate pyapprox-typing

# Install pyapprox in editable mode
cd pyapprox  # root directory
pip install -e .

# Verify
python -m pytest pyapprox/typing/surrogates/kernels/tests/ -v
python -m pytest pyapprox/typing/surrogates/gaussianprocess/tests/ -v
```

## Minimal Installation

For the absolute minimal installation (NumPy backend only, no tests, no docs):

```bash
pip install numpy>=1.20.0 scipy>=1.7.0
```

This provides:
- All kernels (Matern, composition, multi-output, etc.)
- Gaussian process regression with NumPy backend
- Hyperparameter optimization
- HVP computations

PyTorch backend will not be available with this setup.
