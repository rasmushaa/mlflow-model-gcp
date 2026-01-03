#!/usr/bin/env bash
set -euo pipefail


# Configuration
IMAGE_NAME="mlflow-pipeline"
IMAGE_NAME_TAG="${IMAGE_NAME}:latest"
CONTAINER_NAME="${IMAGE_NAME}-local"
PORT=8080


# Check for .env file
if [[ ! -f .env ]]; then
  echo ".env file not found! Please create one according to the README instructions."
  exit 1
fi


# Ensure Docker Desktop / daemon is running; open it if not and wait until ready.
ensure_docker_running() {
  local timeout=120   # seconds
  local interval=2    # seconds
  local waited=0

  if docker info >/dev/null 2>&1; then
    return 0
  fi

  echo "Docker daemon not available. Opening Docker.app..."
  open -a Docker

  printf "Waiting for Docker to be ready"
  while ! docker info >/dev/null 2>&1; do
    sleep "$interval"
    waited=$((waited + interval))
    printf "."
    if [ "$waited" -ge "$timeout" ]; then
      echo
      echo "Timed out after ${timeout}s waiting for Docker to start."
      return 1
    fi
  done
  echo
  echo "Docker is ready (waited ${waited}s)."
  return 0
}
if ! ensure_docker_running; then
  echo "Cannot connect to Docker daemon. Please start Docker Desktop manually and retry."
  exit 1
fi


# Build the Docker image
echo -e "\n1. Building Docker image ${IMAGE_NAME_TAG}..."
docker build -t "${IMAGE_NAME_TAG}" .


# Stop and remove any existing container with the same name
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo -e "\n2. Stopping and removing existing container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}"
fi


# Run the Docker container, mount the local gcloud config for authentication. You should: gcloud auth application-default login, gcloud auth login first.
echo -e "\n3. Running container ${CONTAINER_NAME} from image ${IMAGE_NAME_TAG} (port ${PORT})..."
docker run -d \
--env-file .env \
-p "${PORT}:${PORT}" \
-v $HOME/.config/gcloud:/root/.config/gcloud \
-e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
--name "${CONTAINER_NAME}" "${IMAGE_NAME_TAG}"
echo -e "\n4. Container ${CONTAINER_NAME} is running. Access the app at http://localhost:${PORT}\n"
