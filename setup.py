"""
Setup script for the household energy segmentation package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="household-energy-segmentation",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Advanced Household Energy Segmentation with Multi-Modal Deep Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/household-energy-segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "optional": [
            "psutil>=5.8.0",
            "umap-learn>=0.5.0",
            "hdbscan>=0.8.0",
            "optuna>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "household-segmentation=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["configs/*.yaml", "configs/*.json"],
    },
    zip_safe=False,
)
