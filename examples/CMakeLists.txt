add_executable(quadrotor_tracking quadrotor_tracking.cu)
target_link_libraries(quadrotor_tracking LINK_PUBLIC tinympc)

# add_executable(quadrotor_hovering quadrotor_hovering.cpp)
# target_link_libraries(quadrotor_hovering LINK_PUBLIC tinympc)

# add_executable(codegen_random codegen_random.cpp)
# target_link_libraries(codegen_random LINK_PUBLIC tinympc)

# add_executable(codegen_cartpole codegen_cartpole.cpp)
# target_link_libraries(codegen_cartpole LINK_PUBLIC tinympc)

add_executable(cpu cpu.cu)
target_link_libraries(cpu LINK_PUBLIC tinympc)

add_executable(gpu gpu.cu)
target_link_libraries(gpu LINK_PUBLIC tinympc)

add_executable(energy_test energy_test.cu)
target_link_libraries(energy_test LINK_PUBLIC tinympc)


