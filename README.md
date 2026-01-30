# SE3 SQP Solver

A Sequential Quadratic Programming (SQP) solver for trajectory optimization on SE(3), built on top of HPIPM and Pinocchio.

## Overview

This project implements an SQP-based trajectory optimization framework for robotic systems. It combines:

- **HPIPM**: High-Performance Interior Point Method solver for quadratic programs
- **Pinocchio**: Rigid body dynamics library for computing forward/inverse dynamics and their derivatives

## Project Structure

```
se3_sqp/
├── code/
│   ├── src/           # Library source files
│   ├── include/       # Public headers
│   ├── examples/      # Example applications
│   └── CMakeLists.txt
├── Dockerfile         # Development environment
└── docker-run.sh      # Container access script
```

## Requirements

- Docker (for containerized development)
- ARM64 architecture (optimized for Apple M1/M2)

## Getting Started

### Build the Docker Image

```bash
docker build -t sqp-solver .
```

### Run the Development Container

```bash
./docker-run.sh
```

This script will:
- Start a new container if none exists
- Attach to a running container if one exists
- Start and attach to a stopped container

### Build the Project

Inside the container:

```bash
cd /workspace/code
mkdir build && cd build
cmake ..
make -j2
```

### Run the Verification Example

```bash
./verify_imports
```

## VS Code Development

Open the project in VS Code and use "Reopen in Container" for full IDE integration with autocomplete support.

## Dependencies

Built into the Docker image:

| Library | Version | Location |
|---------|---------|----------|
| BLASFEO | latest | /opt/blasfeo |
| HPIPM | latest | /opt/hpipm |
| Pinocchio | latest | /usr/local |
| Eigen3 | 3.x | system |
| Boost | 1.74 | system |

## Using HPIPM from C++

HPIPM is a C library. Use `extern "C"` blocks:

```cpp
extern "C" {
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_d_ocp_qp_ipm.h>
}
```

## License

[TBD]
