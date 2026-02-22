cmake_minimum_required(VERSION 3.16)
project(sqp_solver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ========================================
# RELEASE BUILD OPTIMIZATIONS
# ========================================
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mno-avx -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -mno-avx -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# pinocchio was built without AVX (SSE only, 16-byte Eigen alignment).
# -march=native enables AVX → Eigen uses 32-byte alignment, mismatching pinocchio's
# allocator across the shared library boundary (heap corruption on free).
# -mno-avx keeps all other -march=native tuning while matching pinocchio's alignment.

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
find_package(example-robot-data REQUIRED)
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
    file(GLOB EXAMPLE_SOURCES "examples/*.cpp")
    foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
        target_link_libraries(${EXAMPLE_NAME} ${PROJECT_NAME} example-robot-data::example-robot-data)
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
