add_executable(quadrotor_tracking quadrotor_tracking.cu)
target_link_libraries(quadrotor_tracking LINK_PUBLIC tinympc)

# add_executable(quadrotor_hovering quadrotor_hovering.cpp)
# target_link_libraries(quadrotor_hovering LINK_PUBLIC tinympc)

# add_executable(codegen_random codegen_random.cpp)
# target_link_libraries(codegen_random LINK_PUBLIC tinympc)

# add_executable(codegen_cartpole codegen_cartpole.cpp)
# target_link_libraries(codegen_cartpole LINK_PUBLIC tinympc)

add_executable(cpu_test cpu_test.cpp)
target_link_libraries(cpu_test LINK_PUBLIC tinympc)

add_executable(cuda_test cuda_test.cpp)
target_link_libraries(cuda_test LINK_PUBLIC tinympc)

add_executable(same_function_test same_function_test.cu)
target_link_libraries(same_function_test LINK_PUBLIC tinympc)