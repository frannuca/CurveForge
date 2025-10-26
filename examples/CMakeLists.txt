cmake_minimum_required(VERSION 3.21)


# Optimizer Examples executable
add_executable(optimizer_example
        advanced_multi_curve.cpp
)


set_target_properties(optimizer_example PROPERTIES
        OUTPUT_NAME "optimizer_example"
)
