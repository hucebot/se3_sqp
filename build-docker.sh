#!/bin/bash
set -e

# Enable BuildKit for parallel stage builds and better caching
export DOCKER_BUILDKIT=1

IMAGE_NAME="sqp-solver:latest"
CONTAINER_NAME="sqp-dev"

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping and removing existing container: ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}"
fi

# Build the optimized image
echo "Building Docker image: ${IMAGE_NAME}"
docker build -f Dockerfile -t "${IMAGE_NAME}" .

echo ""
echo "Done! Image tagged as: ${IMAGE_NAME}"
echo ""
echo "Image size:"
docker images "sqp-solver" --format "  {{.Repository}}:{{.Tag}} - {{.Size}}"
