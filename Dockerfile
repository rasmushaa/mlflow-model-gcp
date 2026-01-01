FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir uv
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
# The commit SHA should be passed as a build argument to connect the mlflow run to code version
ARG GIT_SHA
ENV GIT_COMMIT_SHA=${GIT_SHA}

CMD ["python", "src/main.py"]
