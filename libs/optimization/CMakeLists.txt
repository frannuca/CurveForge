cmake_minimum_required(VERSION 3.21)

find_package(Eigen3 REQUIRED CONFIG)

set(OPTIMIZATION_SOURCES
        src/OptimizerBase.cpp
        include/optimization/OptAlgoParams.h
        # Note: do not list headers using wrong relative paths here.
        # The header lives in include/optimization/OptSolution.h and is exported
        # via target_include_directories. Previously this file listed
        # "optimization/OptSolution.h" which caused CMake to fail because
        # that path does not exist relative to this CMakeLists.txt.
)
add_library(optimization
        ${OPTIMIZATION_SOURCES}
)

if (Eigen3_FOUND)
    message(STATUS "optimization: Eigen3 found, enabling Eigen features")
    target_link_libraries(optimization PUBLIC Eigen3::Eigen)
    target_compile_definitions(optimization PUBLIC INTERPOLATION_HAS_EIGEN=1)
else ()
    message(STATUS "interpolation: Eigen3 not found, building without Eigen")
endif ()


find_package(NLopt CONFIG REQUIRED)
target_link_libraries(optimization PRIVATE NLopt::nlopt)


target_include_directories(optimization
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(CurveForge::optimization ALIAS optimization)

set_target_properties(optimization PROPERTIES
        OUTPUT_NAME "optimization"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)