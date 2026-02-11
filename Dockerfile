# Dockerfile for SQP Solver Development Environment
# HPIPM + Pinocchio on ARM64 (Apple M1)

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libeigen3-dev \
    libboost-all-dev \
    liburdfdom-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Pinocchio - Rigid Body Dynamics Library
# ============================================
# Build first (independent, slow, rarely changes)
# Using -j2 to prevent OOM during template-heavy compilation
RUN git clone --depth 1 --recursive https://github.com/stack-of-tasks/pinocchio.git /tmp/pinocchio && \
    cd /tmp/pinocchio && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_PYTHON_INTERFACE=OFF \
        -DBUILD_TESTING=OFF && \
    make -j2 && \
    make install && \
    rm -rf /tmp/pinocchio

# ============================================
# BLASFEO - Basic Linear Algebra Subroutines
# ============================================
RUN git clone --depth 1 https://github.com/giaf/blasfeo.git /tmp/blasfeo && \
    cd /tmp/blasfeo && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/blasfeo \
        -DTARGET=GENERIC \
        -DBUILD_SHARED_LIBS=ON \
        -DBLASFEO_EXAMPLES=OFF \
        -DBLAS_API=OFF && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/blasfeo

# ============================================
# HPIPM - High-Performance Interior Point Method
# ============================================
RUN git clone --depth 1 https://github.com/giaf/hpipm.git /tmp/hpipm && \
    cd /tmp/hpipm && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/hpipm \
        -DTARGET=GENERIC \
        -DBUILD_SHARED_LIBS=ON \
        -DHPIPM_TESTING=OFF \
        -DBLASFEO_PATH=/opt/blasfeo && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/hpipm

# ============================================
# Example Robot Data - Robot Models for Testing
# ============================================
RUN git clone --depth 1 --recursive https://github.com/Gepetto/example-robot-data.git /tmp/example-robot-data && \
    cd /tmp/example-robot-data && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_PYTHON_INTERFACE=OFF && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/example-robot-data

# ============================================
# Environment Configuration
# ============================================
ENV LD_LIBRARY_PATH=/opt/blasfeo/lib:/opt/hpipm/lib:/usr/local/lib
ENV CMAKE_PREFIX_PATH=/opt/blasfeo:/opt/hpipm:/usr/local
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

# Update shared library cache
RUN ldconfig

# Additional development tools
RUN apt-get update && apt-get install -y \
    cmake-curses-gui \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
