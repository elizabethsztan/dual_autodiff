from setuptools import setup, find_packages

setup(
    # Package discovery
    packages=find_packages(),
    
    # Package data
    package_data={
        'dual_autodiff': ['*.py'],
    },
)