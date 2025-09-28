cmake_minimum_required(VERSION 3.21)

project(CurveForge
        VERSION 0.1.0
        DESCRIPTION "Rates calibration multiproject (libs + apps + tests)"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CURVEFORGE_BUILD_TESTS "Build unit tests" ON)
option(CURVEFORGE_BUILD_APPS "Build applications" ON)
option(CURVEFORGE_BUILD_SHARED "Build shared libs" OFF)

if (CURVEFORGE_BUILD_SHARED)
    set(BUILD_SHARED_LIBS ON)
endif ()

add_subdirectory(libs/mathx)
add_subdirectory(libs/pricing)
add_subdirectory(libs/time)
add_subdirectory(libs/asynchelpers)


if (CURVEFORGE_BUILD_APPS)
    add_subdirectory(apps/curveforge-cli)
endif ()

if (CURVEFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()