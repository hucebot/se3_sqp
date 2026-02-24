cmake_minimum_required(VERSION 3.16)
project(sqp_solver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ========================================
# RELEASE BUILD OPTIMIZATIONS
# ========================================
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# Pinocchio built with -march=native in Docker, enabling full AVX support
# This provides 10-30% performance improvement over SSE-only builds

add_compile_options(
    $<$<CONFIG:Release>:-funroll-loops>
    $<$<CONFIG:Release>:-finline-functions>
)

# ========================================
# DEPENDENCIES
# ========================================
find_package(Eigen3 REQUIRED)
find_package(blasfeo REQUIRED)
find_package(hpipm REQUIRED)
find_package(pinocchio REQUIRED)
find_package(nlohmann_json REQUIRED)

# ========================================
# LIBRARY
# ========================================
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
    nlohmann_json::nlohmann_json
)

# ========================================
# PROFILING CONFIGURATION
# ========================================
option(ENABLE_PROFILING "Enable built-in performance profiling (timing output)" OFF)

if(ENABLE_PROFILING)
    add_compile_definitions(SQP_PROFILING)
    message(STATUS "Performance profiling enabled (SQP_PROFILING)")
endif()

# ========================================
# EXAMPLES CONFIGURATION
# ========================================
option(BUILD_EXAMPLES "Build example executables" ON)

if(BUILD_EXAMPLES)
    file(GLOB_RECURSE EXAMPLE_SOURCES "examples/*.cpp" "examples/**/*.cpp")
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

# ========================================
# PYTHON BINDINGS
# ========================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(sqp_solver_py bindings/bindings.cpp)
    target_link_libraries(sqp_solver_py PRIVATE ${PROJECT_NAME})
    set_target_properties(sqp_solver_py PROPERTIES OUTPUT_NAME "sqp_solver")
endif()
