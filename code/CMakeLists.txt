cmake_minimum_required(VERSION 3.16)
project(sqp_solver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ========================================
# MAXIMUM PERFORMANCE OPTIMIZATION FLAGS
# ========================================

# Aggressive optimization for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Link-Time Optimization (LTO) - CRITICAL for header-only performance
# Allows compiler to inline across translation units
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# Additional performance flags
add_compile_options(
    $<$<CONFIG:Release>:-ffast-math>         # Aggressive floating-point optimizations
    $<$<CONFIG:Release>:-funroll-loops>      # Unroll loops for speed
    $<$<CONFIG:Release>:-finline-functions>  # Inline aggressively
)

message(STATUS "Performance optimizations enabled for Release builds:")
message(STATUS "  - O3 optimization level")
message(STATUS "  - march=native (CPU-specific vectorization)")
message(STATUS "  - Link-Time Optimization (LTO)")
message(STATUS "  - Fast-math, loop unrolling, aggressive inlining")

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(blasfeo REQUIRED)
find_package(hpipm REQUIRED)
find_package(pinocchio REQUIRED)

# sqp_solver library
file(GLOB_RECURSE LIB_SOURCES "src/**.cpp")
add_library(${PROJECT_NAME} ${LIB_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    Eigen3::Eigen
    blasfeo
    hpipm
    pinocchio::pinocchio
)

# Examples
add_executable(verify_imports examples/verify_imports.cpp)
target_link_libraries(verify_imports ${PROJECT_NAME})

add_executable(test_hpipm_integration examples/test_hpipm_integration.cpp)
target_link_libraries(test_hpipm_integration ${PROJECT_NAME})

add_executable(double_integrator_example examples/double_integrator_example.cpp)
target_link_libraries(double_integrator_example ${PROJECT_NAME})


add_executable(example_trajopt examples/example_trajopt.cpp)
target_link_libraries(example_trajopt ${PROJECT_NAME})
