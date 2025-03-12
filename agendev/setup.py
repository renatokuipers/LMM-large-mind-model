from setuptools import setup, find_packages

setup(
    name="agendev",
    version="0.1.0",
    description="An Intelligent Agentic Development System",
    author="RenDev",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests",
        "pydantic",
        "numpy",
        "soundfile",
        "sounddevice",
        # Add dash for the web interface
        "dash",
        "dash-bootstrap-components",
        "plotly",
        "pandas",
    ],
    python_requires=">=3.9",
)