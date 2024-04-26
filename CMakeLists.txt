cmake_minimum_required(VERSION 3.0.0)
project(TinyMPC VERSION 0.2.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ftree-vectorize")

# set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")

include_directories(include/Eigen)

add_subdirectory(src/tinympc)
add_subdirectory(examples)
add_subdirectory(test)