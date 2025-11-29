cmake_minimum_required(VERSION 3.21)

# Spline test executable
add_executable(run_spline_tests
        interpolation/test_bspline.cpp
)

target_link_libraries(run_spline_tests
        PRIVATE
        CurveForge::interpolation
)

add_test(NAME run_spline_tests COMMAND run_spline_tests)
set_tests_properties(run_spline_tests PROPERTIES PASS_REGULAR_EXPRESSION "KNOTS_OK")

# Spline smoothing test
add_executable(run_spline_smoothing_tests
        interpolation/test_bspline_smoothing.cpp
)

target_link_libraries(run_spline_smoothing_tests
        PRIVATE
        CurveForge::interpolation
)

add_test(NAME run_spline_smoothing_tests COMMAND run_spline_smoothing_tests)
set_tests_properties(run_spline_smoothing_tests PROPERTIES PASS_REGULAR_EXPRESSION "SMOOTH_OK")


# swap test
add_executable(run_swap_tests
        instruments/test_swap.cpp
)

target_link_libraries(run_swap_tests
        PRIVATE
        CurveForge::instruments
        CurveForge::curve
        CurveForge::time
)

add_test(NAME run_swap_tests COMMAND run_swap_tests)
set_tests_properties(run_swap_tests PROPERTIES PASS_REGULAR_EXPRESSION "SWAP_OK")

#xcswap test
add_executable(run_xcswap_tests
        instruments/test_xcswap.cpp
)

target_link_libraries(run_xcswap_tests
        PRIVATE
        CurveForge::instruments
        CurveForge::curve
        CurveForge::time
)

add_test(NAME run_xcswap_tests COMMAND run_xcswap_tests)
set_tests_properties(run_xcswap_tests PROPERTIES PASS_REGULAR_EXPRESSION "SWAP_OK")


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

# pricing  test
add_executable(run_pricing_tests
        pricing/test_pricing.cpp
)

target_link_libraries(run_pricing_tests
        PRIVATE
        CurveForge::time
        CurveForge::instruments
        CurveForge::curve
        CurveForge::pricing
)

add_test(NAME run_pricing_tests COMMAND run_pricing_tests)
set_tests_properties(run_pricing_tests PROPERTIES PASS_REGULAR_EXPRESSION "PRICING_OK")


# Date modifier tests
add_executable(run_date_modifier_tests
        time/test_date_modifier.cpp
)

target_link_libraries(run_date_modifier_tests
        PRIVATE
        CurveForge::time
)

add_test(NAME run_date_modifier_tests COMMAND run_date_modifier_tests)
set_tests_properties(run_date_modifier_tests PROPERTIES PASS_REGULAR_EXPRESSION "CALENDAR_OK")


# Date modifier tests
add_executable(run_test_scheduler
        time/test_scheduler.cpp
)

target_link_libraries(run_test_scheduler
        PRIVATE
        CurveForge::time
)

add_test(NAME run_test_scheduler COMMAND run_test_scheduler)
set_tests_properties(run_test_scheduler PROPERTIES PASS_REGULAR_EXPRESSION "CALENDAR_OK")

# data contracts tests
add_executable(run_datacontracts_tests
        datacontracts/datacontracts_tests.cpp
)

target_link_libraries(run_datacontracts_tests
        PRIVATE
        CurveForge::datacontracts
)

add_test(NAME run_datacontracts_tests COMMAND run_datacontracts_tests)
set_tests_properties(run_datacontracts_tests PROPERTIES PASS_REGULAR_EXPRESSION "XInstruments_OK")
