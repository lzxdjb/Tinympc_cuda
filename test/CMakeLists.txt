find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
add_executable(test1 test1.cu)
target_link_libraries(test1 LINK_PUBLIC tinympc)