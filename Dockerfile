# Multi-Stage Dockerfile for SQP Solver
# Optimized for performance and image size
#
# Architecture:
#   Stage 1: base-builder - Common build dependencies
#   Stage 2: pinocchio-builder - Build Pinocchio with AVX optimizations
#   Stage 3: blasfeo-hpipm-builder - Build optimized linear algebra libraries
#   Stage 4: dev - Full development environment (default target)
#
# Build: docker build -f Dockerfile.optimized --target dev -t sqp-solver:dev .
# Run:   docker run -it -v $(pwd):/workspace sqp-solver:dev

# ============================================
# STAGE 1: Base Builder - Common Dependencies
# ============================================
FROM ubuntu:22.04 AS base-builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Consolidate all apt-get calls for better layer caching
# Install build tools and common dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    pkg-config \
    libeigen3-dev \
    libboost-all-dev \
    liburdfdom-dev \
    nlohmann-json3-dev \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# STAGE 2: Pinocchio Builder - Rigid Body Dynamics Library
# ============================================
FROM base-builder AS pinocchio-builder

# Build Pinocchio with AVX optimizations
# -march=native enables AVX (256-bit SIMD) vs SSE (128-bit SIMD)
# This provides 10-30% performance improvement in Pinocchio's math kernels
RUN git clone --depth 1 --recursive https://github.com/stack-of-tasks/pinocchio.git /tmp/pinocchio && \
    cd /tmp/pinocchio && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CXX_FLAGS="-march=native" \
        -DBUILD_PYTHON_INTERFACE=OFF \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF && \
    make -j2 && \
    make install DESTDIR=/opt/pinocchio-install && \
    rm -rf /tmp/pinocchio

# ============================================
# STAGE 3: BLASFEO/HPIPM Builder - Optimized Linear Algebra
# ============================================
FROM base-builder AS blasfeo-hpipm-builder

# Detect architecture and set appropriate TARGET
# x86-64 -> X64_AUTOMATIC (auto-detects AVX2/FMA)
# arm64 -> ARMV8A_APPLE_M1
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo "ARMV8A_APPLE_M1" > /tmp/blasfeo_target; \
    else \
        echo "X64_AUTOMATIC" > /tmp/blasfeo_target; \
    fi

# Build BLASFEO with optimized target (30-50% faster than GENERIC)
RUN BLASFEO_TARGET=$(cat /tmp/blasfeo_target) && \
    git clone --depth 1 https://github.com/giaf/blasfeo.git /tmp/blasfeo && \
    cd /tmp/blasfeo && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/blasfeo \
        -DTARGET=${BLASFEO_TARGET} \
        -DBUILD_SHARED_LIBS=ON \
        -DBLASFEO_EXAMPLES=OFF \
        -DBLAS_API=OFF && \
    make -j$(nproc) && \
    make install DESTDIR=/opt/blasfeo-install && \
    rm -rf /tmp/blasfeo

# Build HPIPM with optimized target (depends on BLASFEO)
RUN BLASFEO_TARGET=$(cat /tmp/blasfeo_target) && \
    git clone --depth 1 https://github.com/giaf/hpipm.git /tmp/hpipm && \
    cd /tmp/hpipm && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/hpipm \
        -DTARGET=${BLASFEO_TARGET} \
        -DBUILD_SHARED_LIBS=ON \
        -DHPIPM_TESTING=OFF \
        -DBLASFEO_PATH=/opt/blasfeo-install/opt/blasfeo && \
    make -j$(nproc) && \
    make install DESTDIR=/opt/hpipm-install && \
    rm -rf /tmp/hpipm

# ============================================
# STAGE 4: Development Environment
# ============================================
FROM ubuntu:22.04 AS dev

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime + development packages in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libraries
    libgomp1 \
    libboost-filesystem1.74.0 \
    libboost-serialization1.74.0 \
    libboost-system1.74.0 \
    # Development headers and libraries
    libeigen3-dev \
    libboost-all-dev \
    liburdfdom-dev \
    nlohmann-json3-dev \
    pybind11-dev \
    # Build tools
    build-essential \
    cmake \
    cmake-curses-gui \
    git \
    pkg-config \
    # Debugging tools
    gdb \
    valgrind \
    # Python environment
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled libraries from builder stages
COPY --from=pinocchio-builder /opt/pinocchio-install/usr/local /usr/local
COPY --from=blasfeo-hpipm-builder /opt/blasfeo-install/opt/blasfeo /opt/blasfeo
COPY --from=blasfeo-hpipm-builder /opt/hpipm-install/opt/hpipm /opt/hpipm

# Install Python packages for visualization and robot modeling
RUN pip3 install --no-cache-dir \
    pin \
    viser \
    yourdfpy \
    numpy

# Set environment variables for library discovery
ENV LD_LIBRARY_PATH=/opt/blasfeo/lib:/opt/hpipm/lib:/usr/local/lib
ENV CMAKE_PREFIX_PATH=/opt/blasfeo:/opt/hpipm:/usr/local
ENV PKG_CONFIG_PATH=/opt/blasfeo/lib/pkgconfig:/opt/hpipm/lib/pkgconfig:/usr/local/lib/pkgconfig

# Update shared library cache
RUN ldconfig

# Create workspace
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
