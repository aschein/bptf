from setuptools import setup, find_packages

setup(
    name="bptf",
    version="0.1",
    description="A simple Python package for Bayesian Poisson tensor factorization (BPTF).",
    author="Aaron Schein",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "tensorly>=0.8.1",
        "numpy>=2.0.2",
        "scipy>=1.14.1",
        "pandas>=2.2.3",
        "numba>=0.60.0",
        "scikit-learn>=1.5.2",
        "sparse>=0.15.4",
    ],
    extras_require={
        "dev": [
            "ipython",
            "matplotlib",
            "seaborn",
            "tqdm",
            "jedi",
            "pytest",  # For testing, if you plan to add tests
        ],
    },
)
