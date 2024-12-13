# DUAL_AUTODIFF_PACKAGE

A dual number implementation for automatic differentiation, available in both Pure Python and Cython packages. This repository provides optimized implementations suitable for both educational purposes and production environments.

## Repository Structure

```
DUAL_AUTODIFF_PACKAGE/
├── cython_package/            # Cython implementation
│   ├── dual_autodiff_x/      # Source code
│   ├── wheelhouse/           # Built wheels
│   ├── pyproject.toml       
│   ├── setup.py
│   └── README.md
│
├── Docker_testing/           # Docker configurations
│   ├── notebooks/           # Jupyter notebooks
│   ├── wheels/              # Wheel files
│   ├── Dockerfile
│   └── README.md
│
├── python_package/          # Python implementation
│   ├── .pytest_cache/      # Test cache
│   ├── build/              # Build artifacts
│   ├── dist/               # Distribution files
│   ├── docs/               # Documentation
│   ├── dual_autodiff/      # Source code
│   ├── tests/              # Test files
│   ├── pyproject.toml
│   ├── setup.py
│   └── README.md
│
└── README.md               # This file
```

## Documentation

The documentation can be found in the prebuilt HTML files in the `python_package/docs/` directory. 

To rebuild the documentation:
1. Navigate to the `python_package/docs/` directory
2. Run the following commands:
```bash
make.bat clean   # Clean existing build
make.bat html    # Build new HTML documentation
```
The newly built documentation will be available in `python_package/docs/build/html/`.

## Choosing an Implementation

### Python Package (`python_package/`)
- Pure Python implementation
- Maximum compatibility and readability
- Ideal for:
  - Learning and educational purposes
  - Small to medium-scale computations
  - Environments where installation simplicity is priority
  - Projects prioritizing code readability

### Cython Package (`cython_package/`)
- High-performance implementation
- Optimized for production use
- Ideal for:
  - Large-scale numerical computations
  - Production environments
  - Performance-critical applications
  - Scientific computing projects

## Quick Start

### Using Python Implementation

```bash
cd python_package
pip install -e .
```

### Using Cython Implementation

```bash
cd cython_package
pip install -e .
```

### Using Docker

The `Docker_testing/` directory contains all necessary files for running the package in a containerized environment:
```bash
cd Docker_testing
docker build -t dual-autodiff .
docker run -it dual-autodiff
```

### Running Tests
There is a pytest suite available for the pure python package. 
```bash
# Test Python implementation
cd python_package
pytest tests/


## License

MIT License - see LICENSE file for details
