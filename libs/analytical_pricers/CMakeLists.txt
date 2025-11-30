cmake_minimum_required(VERSION 3.21)

set(ANALYTICAL_PRICERS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/BlackScholes.cpp
)

add_library(analytical_pricers ${ANALYTICAL_PRICERS_SOURCES})

# Link internal project dependencies
target_link_libraries(analytical_pricers PUBLIC CurveForge::time)

# Public include dir for consumers
target_include_directories(analytical_pricers
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(CurveForge::analytical_pricers ALIAS analytical_pricers)

set_target_properties(analytical_pricers PROPERTIES
        OUTPUT_NAME "analytical_pricers"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)