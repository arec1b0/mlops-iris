"""Setup configuration for MLOps Iris Classification API."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (
    (this_directory / "requirements.txt").read_text(encoding="utf-8").splitlines()
)

setup(
    name="mlops-iris",
    version="1.0.0",
    author="MLOps Team",
    author_email="mlops@example.com",
    description="Production-ready MLOps API for Iris flower classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/mlops-iris",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: AsyncIO",
        "Framework :: FastAPI",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "mlflow": [
            "mlflow>=2.7.0",
        ],
        "docker": [
            "docker>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iris-train=src.train:main",
            "iris-api=src.run_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords=[
        "machine-learning",
        "mlops",
        "iris-classification",
        "fastapi",
        "scikit-learn",
        "onnx",
        "mlflow",
        "api",
        "microservice",
        "docker",
        "kubernetes",
    ],
    project_urls={
        "Documentation": "https://github.com/your-username/mlops-iris#readme",
        "Source": "https://github.com/your-username/mlops-iris",
        "Tracker": "https://github.com/your-username/mlops-iris/issues",
        "CI/CD": "https://github.com/your-username/mlops-iris/actions",
    },
    license="MIT",
    zip_safe=False,
)
