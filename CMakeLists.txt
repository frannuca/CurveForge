cmake_minimum_required(VERSION 3.21)
# Must be set before project() to take effect globally
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.6" CACHE STRING "Minimum OS X deployment version" FORCE)


# cmake
# Put this at the very top of `CMakeLists.txt` (before project())
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(_vcpkg_dir "${CMAKE_SOURCE_DIR}/vcpkg")
    set(_vcpkg_toolchain "${_vcpkg_dir}/scripts/buildsystems/vcpkg.cmake")
    if (EXISTS "${_vcpkg_toolchain}")
        message(STATUS "Enabling vcpkg toolchain: ${_vcpkg_toolchain}")
        set(CMAKE_TOOLCHAIN_FILE "${_vcpkg_toolchain}" CACHE STRING "Vcpkg toolchain file" FORCE)
    endif ()
endif ()

# ... then your usual cmake_minimum_required / project() / targets ...
# Example: find_package will now pick packages declared in `vcpkg.json`
#find_package(Boost REQUIRED COMPONENTS uuid)

# Set policy to use new Boost CMake config instead of FindBoost module
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()

project(CurveForge
        VERSION 0.1.0
        DESCRIPTION "Rates calibration multiproject (libs + apps + tests)"
        LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CURVEFORGE_BUILD_TESTS "Build unit tests" ON)
option(CURVEFORGE_BUILD_APPS "Build applications" ON)
option(CURVEFORGE_BUILD_SHARED "Build shared libs" ON)
option(BOOST_BUILD_IF_MISSING "Build boost libs" ON)


if (CURVEFORGE_BUILD_SHARED)
    set(BUILD_SHARED_LIBS ON)
endif ()

add_subdirectory(libs/datacontracts)
add_subdirectory(libs/interpolation)
add_subdirectory(libs/curve)
add_subdirectory(libs/pricing)
add_subdirectory(libs/optimization)
add_subdirectory(libs/time)
add_subdirectory(libs/instruments)
add_subdirectory(libs/volatility)
add_subdirectory(libs/analytical_pricers)


if (CURVEFORGE_BUILD_APPS)
    add_subdirectory(apps/curveforge-cli)
endif ()

if (CURVEFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

# Add examples directory
add_subdirectory(examples)