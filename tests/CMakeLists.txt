cmake_minimum_required(VERSION 3.21)

# Spline test executable
add_executable(run_spline_tests
        test_bspline.cpp
)

target_link_libraries(run_spline_tests
        PRIVATE
        CurveForge::mathx
)

add_test(NAME run_spline_tests COMMAND run_spline_tests)
set_tests_properties(run_spline_tests PROPERTIES PASS_REGULAR_EXPRESSION "KNOTS_OK")

# Spline smoothing test
add_executable(run_spline_smoothing_tests
        test_bspline_smoothing.cpp
)

target_link_libraries(run_spline_smoothing_tests
        PRIVATE
        CurveForge::mathx
)

add_test(NAME run_spline_smoothing_tests COMMAND run_spline_smoothing_tests)
set_tests_properties(run_spline_smoothing_tests PROPERTIES PASS_REGULAR_EXPRESSION "SMOOTH_OK")
