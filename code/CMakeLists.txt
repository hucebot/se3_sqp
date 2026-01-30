cmake_minimum_required(VERSION 3.16)
project(sqp_solver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(blasfeo REQUIRED)
find_package(hpipm REQUIRED)
find_package(pinocchio REQUIRED)

# sqp_solver library
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
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
