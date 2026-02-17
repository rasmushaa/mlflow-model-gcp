# High-Level ML Pipeline Training Framework


![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Polymodel Version](https://img.shields.io/badge/polymodel-v2.0.0-green.svg)
![MLflow Version](https://img.shields.io/badge/mlflow-v3.6.0-red.svg)

A production-grade MLOps pipeline for training machine learning models on Google Cloud Platform, featuring automated CI/CD workflows, experiment tracking with MLflow, and comprehensive traceability.

## Project Overview

### Polymodel: Configurable ML Pipeline Library

This repository includes `polymodel`, a fully configurable machine learning pipeline framework that supports arbitrary multi-model and preprocessing architectures. Key features:

- **Model Agnostic**: Define any combination of models through YAML configuration
- **Flexible Preprocessing**: Chain multiple transformers and feature engineering steps
- **MLflow Integration**: Complete pipeline artifacts (models, preprocessors, metadata) are registered to MLflow
- **Package Distribution**: Built and distributed as a wheel file to private GCP Artifact Registry PyPI repository
- **Reproducible Pipelines**: Version-controlled package ensures consistent training across environments

The polymodel package enables declarative ML pipeline definition while maintaining full traceability and versioning through MLflow and GCP infrastructure.

## DevOps Strategy

### Branching Model

This project implements a structured branching strategy for ML experimentation and production deployment:

#### Branch Types

| Branch Pattern | Purpose | Pipeline Behavior |
|----------------|---------|-------------------|
| `feature/*` | General development and experimentation | Triggers development training pipeline with branch-specific MLflow experiment |
| `model/*` | Model development and tuning | Triggers development pipeline on push; production deployment on merge to `main` |
| `main` | Production-ready code | Protected branch; only receives validated PRs |

#### Workflow

```
feature/* or model/* → Push
    ↓
PR to main
    ├─ Validate wheel build
    └─ Run development pipeline (branch-specific experiment)
    ↓
PR Merged (model/* only)
    ├─ Build & upload wheel to GCP Artifact Registry
    └─ Trigger production pipeline
        ├─ Experiment: Banking-main
        └─ Model: BankingModel-main
```

### CI/CD Pipelines

1. **Validate Wheel Build**: Runs on every PR to ensure the polymodel package builds successfully
2. **Build and Upload Wheel**: Publishes the `polymodel` wheel to private GCP Artifact Registry after PR merge
3. **Development Training Pipeline**: Runs on `feature/*` and `model/*` branches with branch-specific experiments
4. **Production Training Pipeline**: Runs after successful wheel build for `model/*` merges only

### Traceability

- **Git SHA**: Embedded in Docker images for exact code version tracking
- **MLflow Artifacts**: Complete pipeline (model + preprocessors) registered with full metadata
- **Package Versioning**: Semantic versioning enforced; only new versions are published to Artifact Registry
- **Reproducibility**: Complete lineage from code commit to trained model artifacts

## Development Guide

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for containerized testing)
- MLflow tracking server (local or remote)

### Setup

1. **Install dependencies with uv**
   ```bash
   uv sync --dev
   ```

   This installs all project dependencies including the `polymodel` package in editable mode.

2. **Configure environment variables**

   Create a `.env` file in the project root for local development:

   ```bash
   MLFLOW_EXPERIMENT_NAME=Experiment-local
   MLFLOW_MODEL_NAME=BankingModel-local
   GCP_PROJECT_ID=your-gcp-project-id
   GCP_LOCATION=europe-west1
   GCP_BQ_DATASET=your_bigquery_dataset
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id  # Required for local container
   ```

3. **Configure MLflow tracking** (if using remote server)
   ```bash
   export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"
   export MLFLOW_TRACKING_USERNAME="your-username"
   export MLFLOW_TRACKING_PASSWORD="your-password"
   ```

   Add these to your `.env` file or export them in your shell.

### Local Development Workflow

#### 1. Run Experiments Locally

Test your model configurations and changes locally before pushing to remote:

```bash
bash scripts/run_local_experiment.sh
```

This creates local MLflow runs with your current configuration. Check MLflow UI to inspect results.

#### 2. Test in Container (Optional)

Validate the full containerized pipeline locally:

```bash
bash scripts/run_local_docker.sh
```

This builds and runs the Docker image locally, simulating the cloud environment.

#### 3. Trigger Cloud Development Pipeline

When ready to test in the cloud environment:

```bash
# Create and push a feature or model branch
git checkout -b feature/new-preprocessing
git add .
git commit -m "Add new preprocessing step"
git push origin feature/new-preprocessing
```

This automatically triggers the development training pipeline on GCP with a branch-specific MLflow experiment (`Banking-feature-new-preprocessing`).

#### 4. Deploy to Production

For model changes ready for production:

```bash
# Use model/* branch pattern
git checkout -b model/improved-hyperparameters
# Make changes, commit, and push
git push origin model/improved-hyperparameters
# Open PR to main → after merge, production pipeline runs automatically
```

### Configuration

Model and training parameters are defined in [config.yaml](config.yaml).
Modify this file to experiment with different model architectures and preprocessing pipelines.
