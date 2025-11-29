cmake_minimum_required(VERSION 3.21)

set(INSTRUMENT_SOURCES
        src/swap.cpp
        src/Leg.cpp
        src/Instrument.cpp
        src/FixFloatSwap.cpp
        src/XCSwap.cpp
)

add_library(instruments ${INSTRUMENT_SOURCES})

find_package(Boost REQUIRED COMPONENTS uuid)

if (Boost_FOUND)
    message(STATUS "Found Boost::uuid")
    target_link_libraries(instruments PUBLIC Boost::uuid)
else ()
    message(FATAL_ERROR "Boost::uuid not found")
endif ()


add_library(CurveForge::instruments ALIAS instruments)

# Link internal project dependencies
target_link_libraries(instruments PUBLIC CurveForge::time)
target_link_libraries(instruments PUBLIC CurveForge::interpolation)


# Public include dir for consumers
target_include_directories(instruments
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)


set_target_properties(instruments PROPERTIES
        OUTPUT_NAME "instruments"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)