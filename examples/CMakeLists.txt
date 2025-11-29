cmake_minimum_required(VERSION 3.21)


# Optimizer Examples executable
add_executable(optimizer_example
        advanced_multi_curve.cpp
)


set_target_properties(optimizer_example PROPERTIES
        OUTPUT_NAME "optimizer_example"
)

# Volatility Surface Calibration Example
add_executable(vol_surface_example
        vol_surface_calibration.cpp
)

target_link_libraries(vol_surface_example PRIVATE
        CurveForge::volatility
        CurveForge::interpolation
        CurveForge::datacontracts
)


set_target_properties(vol_surface_example PROPERTIES
        OUTPUT_NAME "vol_surface_example"
)
