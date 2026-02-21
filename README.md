# SE(3) SQP Solver

SQP-based trajectory optimization for robots, built on HPIPM and Pinocchio.

## Build

```bash
./build-docker.sh
./run-docker.sh
```

Inside the container:

```bash
cd /workspace/code
mkdir build && cd build
cmake ..
make -j2
```

## Examples

```bash
./example_trajopt     # pendubot swing-up
./franka_reaching     # Franka Panda end-effector tracking
./go2_trot            # Unitree Go2 trotting gait
```
