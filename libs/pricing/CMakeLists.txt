cmake_minimum_required(VERSION 3.21)

find_package(Eigen3 REQUIRED CONFIG)
if (Eigen3_FOUND)
    message(STATUS "interpolation: Eigen3 found, enabling Eigen features")
    target_link_libraries(interpolation PUBLIC Eigen3::Eigen)
    target_compile_definitions(interpolation PUBLIC INTERPOLATION_HAS_EIGEN=1)
else ()
    message(STATUS "interpolation: Eigen3 not found, building without Eigen")
endif ()

set(PRICING_SOURCES
        src/FixFloatSwapPricer.cpp
        src/XCSwapPricer.cpp
        include/pricing/XCSwapPricer.h
        src/GreekCalculator.cpp
        include/pricing/GreekCalculator.h
)
add_library(pricing ${PRICING_SOURCES})

# Link internal project dependencies
target_link_libraries(pricing PUBLIC CurveForge::time)
target_link_libraries(pricing PUBLIC CurveForge::instruments)
target_link_libraries(pricing PUBLIC CurveForge::curve)


target_include_directories(pricing
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(CurveForge::pricing ALIAS pricing)

set_target_properties(pricing PROPERTIES
        OUTPUT_NAME "pricing"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)