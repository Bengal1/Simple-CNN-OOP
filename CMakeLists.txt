cmake_minimum_required(VERSION 3.15)
project(SimpleCNN_OOP LANGUAGES CXX)

# Use C++17 standard without compiler-specific extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Gather all source files recursively
file(GLOB_RECURSE SRC_FILES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# Add executable target
add_executable(SimpleCNN ${SRC_FILES})

# Include directories scoped to this target only
target_include_directories(SimpleCNN PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    "C:/Libraries/Eigen-3.4.0"
)

# Compiler-specific flags and warnings
if(MSVC)
    target_compile_options(SimpleCNN PRIVATE /O2 /W4)
else()
    target_compile_options(SimpleCNN PRIVATE -O2 -Wall -Wextra -Wpedantic -fdiagnostics-color=always)
endif()
