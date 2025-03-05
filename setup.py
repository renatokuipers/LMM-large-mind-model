"""
Setup configuration for NeuralChild package.
"""

import os
from setuptools import setup, find_packages

# Read the content of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Define package requirements
requirements = [
    "numpy>=1.19.0",
    "scipy>=1.5.0",
    "torch>=1.9.0",
    "nltk>=3.6.0",
    "spacy>=3.0.0",
    "gensim>=4.0.0",
    "faiss-cpu>=1.7.0",
    "textblob>=0.15.3",
    "plotly>=5.0.0",
    "dash>=2.0.0",
    "dash-bootstrap-components>=1.0.0",
    "networkx>=2.6.0",
    "pydantic>=1.8.0",
    "joblib>=1.0.0",
    "pandas>=1.3.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.5b2",
        "isort>=5.9.0",
        "flake8>=3.9.0",
        "mypy>=0.812",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "sphinx-autodoc-typehints>=1.12.0",
    ],
    "gpu": [
        "torch>=1.9.0",  # Replace with cuda-specific version if needed
    ],
}

setup(
    name="neuralchild",
    version="0.1.0",
    description="A Psychological Mind Simulation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NeuralChild Development Team",
    author_email="info@neuralchild.example.com",
    url="https://github.com/neuralchild/neuralchild",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "NeuralChild": ["assets/*.css"],
    },
    install_requires=requirements,
    extras_require=extras_require,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "neuralchild=NeuralChild.main:main",
        ],
    },
)