# Compiler & Dockerfile Notes

## The Eigen Alignment Problem

When building with `-march=native` on an x86 machine that supports AVX, the compiler
defines `__AVX__`. Eigen detects this and switches from **16-byte** (SSE) to **32-byte**
(AVX) aligned allocations. This alignment setting is baked into the binary through
`EIGEN_IDEAL_MAX_ALIGN_BYTES` and `EIGEN_DEFAULT_ALIGN_BYTES`.

Pinocchio uses Eigen's `aligned_allocator` for its internal data structures
(`pinocchio::Data` contains `std::vector<..., Eigen::aligned_allocator<...>>`).
When pinocchio's `.so` and your code disagree on the alignment, one side allocates at
16-byte alignment while the other tries to free assuming 32-byte alignment. This
corrupts the heap and causes a double-free / segfault at destruction time.

### What is AVX?

| ISA   | Register width | Doubles per op | Alignment |
|-------|---------------|----------------|-----------|
| SSE2  | 128-bit       | 2              | 16 bytes  |
| AVX   | 256-bit       | 4              | 32 bytes  |
| AVX-512| 512-bit      | 8              | 64 bytes  |

AVX is fully compatible with pinocchio and Eigen. The only requirement is that
**all translation units that share Eigen-aligned types must use the same alignment**.

### Why `EIGEN_MAX_ALIGN_BYTES=16` doesn't work

Eigen computes alignment in `ConfigureVectorization.h`:

```
EIGEN_IDEAL_MAX_ALIGN_BYTES = 32    (from __AVX__)
EIGEN_MAX_ALIGN_BYTES       = 16    (from our -D flag)
EIGEN_DEFAULT_ALIGN_BYTES   = max(IDEAL, MAX) = 32   <-- ignores our override!
```

`EIGEN_DEFAULT_ALIGN_BYTES` is unconditionally redefined and cannot be overridden by a
preprocessor define. This is an Eigen design limitation.

## Current Workaround

In `CMakeLists.txt`:
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mno-avx -DNDEBUG")
```

`-mno-avx` disables AVX instruction generation while keeping all other `-march=native`
tuning (cache sizes, branch hints, SSE4.2, etc.). This ensures Eigen sees SSE-only and
uses 16-byte alignment, matching the precompiled pinocchio.

## Proper Fix: Rebuild pinocchio with `-march=native`

The Dockerfile currently builds pinocchio without `-march=native`:

```dockerfile
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    ...
```

To get full AVX performance in both pinocchio and your code, change the pinocchio
build to:

```dockerfile
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=native" \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -DBUILD_TESTING=OFF
```

Then remove `-mno-avx` from `CMakeLists.txt`:

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
```

### BLASFEO note

BLASFEO also has architecture-specific optimizations. The Dockerfile currently uses
`-DTARGET=GENERIC`. For best performance, set the target to match your CPU:

```dockerfile
# For x86-64 with AVX2 (most modern Intel/AMD):
cmake .. -DTARGET=X64_AUTOMATIC ...

# For Apple M1/M2 (ARM):
cmake .. -DTARGET=ARMV8A_APPLE_M1 ...
```

See https://github.com/giaf/blasfeo for the full target list.
