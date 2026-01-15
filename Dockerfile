FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy everything needed
COPY pyproject.toml uv.lock /app/
COPY polymodel /app/polymodel
COPY src /app/src
COPY config.yaml /app/config.yaml

# Use uv workspace mode to install also workspace packages (polymodel), with frozen dependencies fom uv.lock to system Python env
RUN uv venv --system && uv sync --frozen

# The commit SHA should be passed as a build argument to connect the mlflow run to code version
ARG GIT_SHA
ENV GIT_COMMIT_SHA=${GIT_SHA}

CMD ["uv", "run", "src/main.py"]
