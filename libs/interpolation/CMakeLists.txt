cmake_minimum_required(VERSION 3.21)

find_package(Eigen3 REQUIRED CONFIG)

add_library(interpolation
        src/bspline.cpp
        include/interpolation/linear.h
        src/linear.cpp
        include/interpolation/linear.h
)

if (Eigen3_FOUND)
    message(STATUS "interpolation: Eigen3 found, enabling Eigen features")
    target_link_libraries(interpolation PUBLIC Eigen3::Eigen)
    target_compile_definitions(interpolation PUBLIC INTERPOLATION_HAS_EIGEN=1)
else ()
    message(STATUS "interpolation: Eigen3 not found, building without Eigen")
endif ()


target_include_directories(interpolation
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(CurveForge::interpolation ALIAS interpolation)

set_target_properties(interpolation PROPERTIES
        OUTPUT_NAME "interpolation"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)