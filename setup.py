from setuptools import setup, find_packages

setup(
    name="lmm_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.5.2",
        "torch>=2.1.1",
        "numpy>=1.26.2",
        "faiss-cpu>=1.7.4",
        "matplotlib>=3.8.2",
        "requests>=2.31.0",
        "soundfile>=0.12.1",
        "sounddevice>=0.4.6",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "nltk>=3.8.1",
        "textblob>=0.17.1",
        "scipy>=1.11.3",
        "tqdm>=4.66.1",
    ],
    author="Renjestoo",
    description="Large Mind Model - A project to create an autonomous developing mind",
    python_requires=">=3.9",
) 