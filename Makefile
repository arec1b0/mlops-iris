# MLOps Iris Classification - Development Makefile

.PHONY: help install test lint format clean build run deploy

# Default target
help: ## Show this help message
	@echo "MLOps Iris Classification - Development Commands"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install: ## Install Python dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black ruff mypy pytest pytest-cov pre-commit

# Development
lint: ## Run linting checks
	ruff check src/ tests/
	black --check --diff src/ tests/

format: ## Format code with Black
	black src/ tests/

type-check: ## Run type checking with MyPy
	mypy src/ --ignore-missing-imports

# Testing
test: ## Run all tests
	pytest tests/

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/ -m "not (integration or e2e or performance)"

test-integration: ## Run integration tests only
	pytest tests/ -m integration

test-e2e: ## Run end-to-end tests only
	pytest tests/ -m e2e

test-performance: ## Run performance tests only
	pytest tests/ -m performance

# Training
train: ## Train the model
	python train.py

train-no-mlflow: ## Train the model without MLflow
	python train.py --no-mlflow

# API
run-api: ## Run the API server
	python run_api.py

run-api-dev: ## Run the API server in development mode
	python run_api.py --reload --host 0.0.0.0 --port 8000

# Docker
build: ## Build Docker image
	docker build -t mlops-iris .

build-no-cache: ## Build Docker image without cache
	docker build --no-cache -t mlops-iris .

run-docker: ## Run Docker container
	docker run -p 8000:80 mlops-iris

run-compose: ## Run with Docker Compose
	docker-compose up --build

run-compose-mlflow: ## Run with Docker Compose including MLflow
	docker-compose --profile mlflow up --build

stop-compose: ## Stop Docker Compose services
	docker-compose down

# Quality Assurance
qa: lint type-check test-cov ## Run full quality assurance suite

security-scan: ## Run security vulnerability scan
	trivy fs .

# Pre-commit
setup-pre-commit: ## Setup pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

run-pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

# Cleanup
clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache
	rm -rf artifacts/*.pkl
	rm -rf mlruns mlflow

clean-all: clean ## Clean all including virtual environment
	rm -rf env venv .venv
	docker system prune -f
	docker volume prune -f

# Deployment (examples)
deploy-staging: ## Deploy to staging (example)
	@echo "🚀 Deploying to staging..."
	@echo "Add your staging deployment commands here"

deploy-production: ## Deploy to production (example)
	@echo "🚀 Deploying to production..."
	@echo "Add your production deployment commands here"

# CI/CD simulation
ci: qa build test-docker ## Simulate CI pipeline locally

# Documentation
docs: ## Generate documentation (if applicable)
	@echo "Documentation generation not implemented yet"

# Development server with auto-reload
dev: ## Start development environment
	@echo "Starting development environment..."
	@echo "API will be available at http://localhost:8000"
	@echo "MLflow UI at http://localhost:5000 (if running)"
	docker-compose --profile mlflow up --build
