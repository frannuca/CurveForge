cmake_minimum_required(VERSION 3.21)

find_package(Eigen3 REQUIRED CONFIG)

add_library(mathx
        src/mathx.cpp
        src/bspline.cpp
        include/mathx/bspline.h
)

if (Eigen3_FOUND)
    message(STATUS "mathx: Eigen3 found, enabling Eigen features")
    target_link_libraries(mathx PUBLIC Eigen3::Eigen)
    target_compile_definitions(mathx PUBLIC MATHX_HAS_EIGEN=1)
else()
    message(STATUS "mathx: Eigen3 not found, building without Eigen")
endif()


target_include_directories(mathx
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(CurveForge::mathx ALIAS mathx)

set_target_properties(mathx PROPERTIES
        OUTPUT_NAME "mathx"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)