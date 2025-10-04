cmake_minimum_required(VERSION 3.21)

# Spline test executable
add_executable(run_spline_tests
        mathx/test_bspline.cpp
)

target_link_libraries(run_spline_tests
        PRIVATE
        CurveForge::mathx
)

add_test(NAME run_spline_tests COMMAND run_spline_tests)
set_tests_properties(run_spline_tests PROPERTIES PASS_REGULAR_EXPRESSION "KNOTS_OK")

# Spline smoothing test
add_executable(run_spline_smoothing_tests
        mathx/test_bspline_smoothing.cpp
)

target_link_libraries(run_spline_smoothing_tests
        PRIVATE
        CurveForge::mathx
)

add_test(NAME run_spline_smoothing_tests COMMAND run_spline_smoothing_tests)
set_tests_properties(run_spline_smoothing_tests PROPERTIES PASS_REGULAR_EXPRESSION "SMOOTH_OK")


# time  test
add_executable(run_calendar_tests
        time/test_calendars.cpp
)

target_link_libraries(run_calendar_tests
        PRIVATE
        CurveForge::time
)

add_test(NAME run_calendar_tests COMMAND run_calendar_tests)
set_tests_properties(run_calendar_tests PROPERTIES PASS_REGULAR_EXPRESSION "CALENDAR_OK")


# async  test
add_executable(run_async_tests
        asynchelpers/test_async.cpp
)

target_include_directories(run_async_tests
        PRIVATE
        ${CMAKE_SOURCE_DIR}/libs/asynchelpers/include
)

target_link_libraries(run_async_tests
        PRIVATE
        CurveForge::asynchelpers
)

add_test(NAME run_async_tests COMMAND run_async_tests)
set_tests_properties(run_async_tests PROPERTIES PASS_REGULAR_EXPRESSION "ASYNC_OK")


# Bond pricing test
add_executable(run_bond_pricing_tests
        pricing/test_bond_pricing.cpp
)

target_link_libraries(run_bond_pricing_tests
        PRIVATE
        CurveForge::pricing
)

add_test(NAME run_bond_pricing_tests COMMAND run_bond_pricing_tests)
set_tests_properties(run_bond_pricing_tests PROPERTIES PASS_REGULAR_EXPRESSION "BOND_PRICING_OK")

