cmake_minimum_required(VERSION 3.15)
project(SimpleCNN_OOP)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags - better to use target_compile_options for modern CMake
if(MSVC)
    # For MSVC, set optimization and color diagnostics (optional)
    target_compile_options(SimpleCNN PRIVATE /O2)
else()
    # For GCC/Clang
    add_compile_options(-O2 -fdiagnostics-color=always)
endif()

# Include directories (prefer target_include_directories)
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    "C:/Libraries/Eigen-3.4.0"
)

# Gather all source files recursively
file(GLOB_RECURSE SRC_FILES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# Add executable target
add_executable(SimpleCNN ${SRC_FILES})

# Prefer modern CMake approach to include directories per target
target_include_directories(SimpleCNN PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    "C:/Libraries/Eigen-3.4.0"
)
