# DUAL_AUTODIFF_PACKAGE

A dual number implementation for automatic differentiation, available in both Pure Python and Cython packages. This repository provides optimised implementations suitable for both educational purposes and production environments.

## LLM Usage

This project was developed with supportive use of Large Language Models (LLMs):

- **Models Used**: Claude 3.5 Sonnet and ChatGPT 4.0

### Development Process
All algorithms were implemented independently. LLMs were primarily utilised for:
- Plotting assistance 
- Bug fixing support
- Code commenting and docstrings

In particular, I used Claude to static type my .pyx files when Cythonising the package and add dosctrings to the original package. Claude was also used to make the README.md template.

### Example Prompts Used
- "ERROR MESSAGE What is wrong with my code?"
- "What tests should I implement for my dual class?"
- "Add docstrings to this function."
- "Please add static typing so we can Cythonise the package."

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
├── requirements/          # Requirements file
│   ├── requirements_dev.txt      # External dependencies used in development
│   ├── requirements.txt          # Minimal external dependencies
│   └── requirements_full.txt     # Comprehensive dependencies
│
└── README.md               # This file
```

## Choosing an Implementation

### Python Package (`python_package/`)
- Pure Python implementation
- Maximum compatibility and readability
- Ideal for:
  - Learning and educational purposes
  - Small to medium-scale computations
  - Environments where installation simplicity is priority
  - Projects prioritising code readability

### Cython Package (`cython_package/`)
- High-performance implementation
- Optimised for production use
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

## Requirements 
The requirement files are located in the `requirements/` directory. If you want to contribute to the development of the package, please run 

```bash
cd requirements
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

in your virtual environment to download the required dependencies.
If you want to rebuild Sphinx documentation, please download Pandoc, as this cannot be downloaded through pip. 
For a `conda` virtual environment

```bash

conda install -c conda-forge pandoc

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


### Using Docker

The `Docker_testing/` directory contains all necessary files for running the package in a containerised environment:
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
```

## License

MIT License - see LICENSE file for details
