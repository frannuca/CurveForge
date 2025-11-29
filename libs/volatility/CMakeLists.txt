cmake_minimum_required(VERSION 3.21)

find_package(Eigen3 REQUIRED CONFIG)

add_library(volatility
        src/BlackScholes.cpp
        src/ImpliedVolSurface.cpp
        include/volatility/ImpliedVolSurface.h
        include/volatility/BlackScholes.h
)

if (Eigen3_FOUND)
    message(STATUS "volatility: Eigen3 found, enabling Eigen features")
    target_link_libraries(volatility PUBLIC Eigen3::Eigen)
    target_compile_definitions(volatility PUBLIC VOLATILITY_HAS_EIGEN=1)
else ()
    message(STATUS "volatility: Eigen3 not found, building without Eigen")
endif ()


target_include_directories(volatility
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(volatility PUBLIC
        CurveForge::datacontracts
        CurveForge::interpolation
)
add_library(CurveForge::volatility ALIAS volatility)

set_target_properties(volatility PROPERTIES
        OUTPUT_NAME "volatility"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)