#!/bin/bash
set -e

IMAGE_NAME="sqp-solver"
CONTAINER_NAME="sqp-dev"

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping and removing existing container: ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}"
fi

# Build the new image
echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo "Done! Image '${IMAGE_NAME}' is ready."
