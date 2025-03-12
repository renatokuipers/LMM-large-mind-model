from setuptools import setup, find_packages

setup(
    name="agendev",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Web server dependencies
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "websockets>=11.0.3",
        
        # Local LLM integration
        "requests>=2.28.0",
        "numpy>=1.24.0",
        
        # Audio/TTS dependencies
        "soundfile>=0.12.1",
        "sounddevice>=0.4.6",
    ],
    entry_points={
        "console_scripts": [
            "agendev=agendev.app:main",
        ],
    },
    author="Renjestoo",
    author_email="renjestoo@gmail.com",
    description="An agentive development framework called AgenDev",
    keywords="agent, development, llm",
    python_requires=">=3.9",
)