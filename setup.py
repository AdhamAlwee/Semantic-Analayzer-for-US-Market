"""
Setup configuration for Semantic Analyzer package
"""

from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

_requirements_path = Path(__file__).parent / "requirements.txt"
with open(_requirements_path, "r", encoding="utf-8") as fh:
    install_requires = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="semantic-analyzer",
    version="0.1.0",
    author="Mas Hairul Adham Bin Alwee",
    author_email="mas_22005746@utp.edu.my",
    description="Financial News Sentiment Analysis Using FinBERT for Stock Price Movement Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["notebooks*", "data*", "models*", "results*", "logs*"]),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
)
