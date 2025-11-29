cmake_minimum_required(VERSION 3.21)

set(CURVE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Pillar.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ICurveCalibration.cpp
        src/ICurve.cpp
        src/FlatRateCurve.cpp
        ../volatility/include/volatility/VolatilityFactory.h
)

add_library(curve ${CURVE_SOURCES})

# Link internal project dependencies
target_link_libraries(curve PUBLIC CurveForge::time)
target_link_libraries(curve PUBLIC CurveForge::interpolation)

# Public include dir for consumers
target_include_directories(curve
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(CurveForge::curve ALIAS curve)

set_target_properties(curve PROPERTIES
        OUTPUT_NAME "curve"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)