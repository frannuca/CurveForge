cmake_minimum_required(VERSION 3.21)

# Set policy CMP0167 to OLD to preserve the FindBoost module behavior and suppress warning.
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif ()

set(INSTRUMENT_SOURCES
        src/swap.cpp
        include/instruments/Basket.h
        include/instruments/Leg.h
        src/Leg.cpp
        include/market/marketdata.h
        include/curves/ICurve.h
        include/volatility/IVolatility.h
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