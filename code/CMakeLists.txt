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

# ========================================
# EXAMPLES CONFIGURATION
# ========================================
option(BUILD_EXAMPLES "Build example executables" OFF)

if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES "examples/*.cpp")
    foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
        target_link_libraries(${EXAMPLE_NAME} ${PROJECT_NAME})
    endforeach()
endif()

# ========================================
# TESTING CONFIGURATION
# ========================================
option(BUILD_TESTING "Build unit tests" OFF)

if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()
