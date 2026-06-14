"""
setup.py
────────
Package installation for genetic-quant-trader.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="genetic-quant-trader",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description=(
        "Genetic Algorithm framework for discovering mean-reversion "
        "trading strategies across equity markets."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/genetic-quant-trader",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ga-evolve=scripts.run_evolution:main",
            "ga-backtest=scripts.backtest_strategy:main",
            "ga-validate=scripts.walk_forward:main",
            "ga-data=scripts.download_data:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords=[
        "genetic-algorithm", "pairs-trading", "mean-reversion",
        "quantitative-finance", "algorithmic-trading", "cointegration",
        "backtesting", "sharpe-ratio", "evolutionary-computation",
    ],
)
