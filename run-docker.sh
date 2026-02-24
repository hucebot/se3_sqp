#!/bin/bash
# Script to run/attach to the SQP solver development container

CONTAINER_NAME="sqp-dev"
IMAGE_NAME="sqp-solver:latest"
WORKSPACE_DIR="$(pwd)"

# Check if container exists and is running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' is running. Attaching..."
    docker exec -it "${CONTAINER_NAME}" /bin/bash
# Check if container exists but is stopped
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' exists but stopped. Starting and attaching..."
    docker start "${CONTAINER_NAME}"
    docker exec -it "${CONTAINER_NAME}" /bin/bash
else
    echo "Creating new container '${CONTAINER_NAME}'..."
    docker run -it \
        --name "${CONTAINER_NAME}" \
        -v "${WORKSPACE_DIR}:/workspace" \
        -w /workspace \
        -p 8080:8080 \
        "${IMAGE_NAME}" \
        /bin/bash
fi
