# Multi-Stage Dockerfile for SQP Solver
# Optimized for performance and image size
#
# Architecture:
#   Stage 1: base-builder - Common build dependencies
#   Stage 2: eigenpy-builder - Build eigenpy (needed by pinocchio Python bindings)
#   Stage 3: pinocchio-builder - Build Pinocchio with AVX + Python interface
#   Stage 4: blasfeo-hpipm-builder - Build optimized linear algebra libraries
#   Stage 5: dev - Full development environment (default target)
#
# Build: docker build --target dev -t sqp-solver:dev .
# Run:   docker run -it -v $(pwd):/workspace sqp-solver:dev

# ============================================
# STAGE 1: Base Builder - Common Dependencies
# ============================================
FROM ubuntu:24.04 AS base-builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

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
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# STAGE 2: eigenpy Builder - Eigen/Python Bridge
# (Must be built BEFORE pinocchio for Python bindings)
# ============================================
FROM base-builder AS eigenpy-builder

RUN git clone --depth 1 --branch v3.10.0 https://github.com/stack-of-tasks/eigenpy.git /tmp/eigenpy && \
    cd /tmp/eigenpy && \
    sed -i 's/cmake_minimum_required.*/cmake_minimum_required(VERSION 3.22)/' CMakeLists.txt && \
    sed -i '/add_subdirectory(unittest)/d' CMakeLists.txt && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF \
        -DPYTHON_EXECUTABLE=$(which python3) && \
    make -j8 && \
    make install DESTDIR=/opt/eigenpy-install && \
    rm -rf /tmp/eigenpy

# ============================================
# STAGE 3: Pinocchio Builder - Rigid Body Dynamics Library
# Built WITH Python interface using eigenpy
# ============================================
FROM base-builder AS pinocchio-builder

# Copy eigenpy from previous stage
COPY --from=eigenpy-builder /opt/eigenpy-install/usr/local /usr/local

# Build Pinocchio with AVX optimizations AND Python bindings
RUN git clone --depth 1 --recursive https://github.com/stack-of-tasks/pinocchio.git /tmp/pinocchio && \
    cd /tmp/pinocchio && \
    sed -i 's/cmake_minimum_required.*/cmake_minimum_required(VERSION 3.22)/' CMakeLists.txt && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DPYTHON_EXECUTABLE=$(which python3) && \
    make -j8 && \
    make install DESTDIR=/opt/pinocchio-install && \
    rm -rf /tmp/pinocchio

# ============================================
# STAGE 4: BLASFEO/HPIPM Builder - Optimized Linear Algebra
# ============================================
FROM base-builder AS blasfeo-hpipm-builder

# Detect architecture and set appropriate TARGET
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo "ARMV8A_APPLE_M1" > /tmp/blasfeo_target; \
    else \
        echo "X64_AUTOMATIC" > /tmp/blasfeo_target; \
    fi

# Build BLASFEO with optimized target
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
    make -j8 && \
    make install DESTDIR=/opt/blasfeo-install && \
    rm -rf /tmp/blasfeo

# Build HPIPM with optimized target
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
    make -j8 && \
    make install DESTDIR=/opt/hpipm-install && \
    rm -rf /tmp/hpipm

# ============================================
# STAGE 5: Development Environment
# ============================================
FROM ubuntu:24.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime + development packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libraries
    libgomp1 \
    libboost-filesystem1.83.0 \
    libboost-serialization1.83.0 \
    libboost-system1.83.0 \
    libboost-python1.83.0 \
    # Development headers and libraries
    libeigen3-dev \
    libboost-all-dev \
    liburdfdom-dev \
    nlohmann-json3-dev \
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
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Copy eigenpy (includes Python module)
COPY --from=eigenpy-builder /opt/eigenpy-install/usr/local /usr/local

# Copy pinocchio (includes C++ lib AND Python module)
COPY --from=pinocchio-builder /opt/pinocchio-install/usr/local /usr/local

# Copy BLASFEO/HPIPM
COPY --from=blasfeo-hpipm-builder /opt/blasfeo-install/opt/blasfeo /opt/blasfeo
COPY --from=blasfeo-hpipm-builder /opt/hpipm-install/opt/hpipm /opt/hpipm

# Install Python packages for visualization (NO pin - we use source-built pinocchio)
RUN pip3 install --no-cache-dir --break-system-packages \
    viser \
    yourdfpy \ 
    mujoco

# Set environment variables
ENV LD_LIBRARY_PATH=/opt/blasfeo/lib:/opt/hpipm/lib:/usr/local/lib
ENV CMAKE_PREFIX_PATH=/opt/blasfeo:/opt/hpipm:/usr/local
ENV PKG_CONFIG_PATH=/opt/blasfeo/lib/pkgconfig:/opt/hpipm/lib/pkgconfig:/usr/local/lib/pkgconfig
ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages

# Update shared library cache
RUN ldconfig

WORKDIR /workspace

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        x11-apps terminator && \
    rm -rf /var/lib/apt/lists/*
    
RUN apt update && apt install -y \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-shape0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libgl1 libglx0 libegl1 libsecret-1-0
    
CMD ["/bin/bash"]
