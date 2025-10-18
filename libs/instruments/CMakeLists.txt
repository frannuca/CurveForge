cmake_minimum_required(VERSION 3.21)

# Require Boost::uuid for the instruments library. Prefer managing Boost via vcpkg
# or system package manager. Users should configure CMake with:
#   -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
# and ensure `vcpkg install boost-uuid` was run for the desired triplet.
find_package(Boost REQUIRED COMPONENTS uuid)

set(INSTRUMENT_SOURCES
        src/bond.cpp
        src/european_option.cpp
        src/forward.cpp
        src/fra.cpp
        src/swap.cpp
        src/InstrumentBase.cpp
)

add_library(instruments ${INSTRUMENT_SOURCES})

# Link internal project dependencies
target_link_libraries(instruments PUBLIC CurveForge::time)

# Link Boost::uuid
if (TARGET Boost::uuid)
    target_link_libraries(instruments PRIVATE Boost::uuid)
else ()
    # Fallback (older CMake/FindBoost behavior)
    target_include_directories(instruments PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(instruments PRIVATE ${Boost_LIBRARIES})
endif ()

# Public include dir for consumers
target_include_directories(instruments
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(CurveForge::instruments ALIAS instruments)

set_target_properties(instruments PROPERTIES
        OUTPUT_NAME "instruments"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)