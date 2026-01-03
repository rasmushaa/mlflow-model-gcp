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

# Use system Python (no venv)
RUN uv pip install --system -r pyproject.toml

# Build polymodel to include the package wheel
RUN uv build polymodel/ --out-dir /app/dist

# The commit SHA should be passed as a build argument to connect the mlflow run to code version
ARG GIT_SHA
ENV GIT_COMMIT_SHA=${GIT_SHA}

CMD ["python", "src/main.py"]
