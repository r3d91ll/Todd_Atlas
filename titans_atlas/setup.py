#!/usr/bin/env python3
"""Setup script for titans_atlas package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
    ]

setup(
    name="titans_atlas",
    version="0.1.0",
    author="Implementation of Behrouz et al.",
    description="Titans & Atlas: Learning to Memorize at Test Time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/titans_atlas",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "flash": [
            "flash-attn>=2.3.0",
            "triton>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-titans=titans_atlas.examples.train_titans:main",
        ],
    },
)
