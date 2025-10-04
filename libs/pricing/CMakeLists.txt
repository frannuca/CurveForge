cmake_minimum_required(VERSION 3.21)

add_library(pricing
        src/engine.cpp
        src/curve.cpp
        src/ois_deposit.cpp
        src/fra.cpp
        src/futures.cpp
        src/irswap.cpp
        src/yield_curve.cpp
        src/bootstrapper.cpp
        src/bond/bond.cpp
        src/bond/carry_roll.cpp
        src/bond/bond_future.cpp
        src/bond/inss.cpp
        src/curvebase.cpp
)

target_include_directories(pricing
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(pricing
        PUBLIC
        CurveForge::mathx
)

add_library(CurveForge::pricing ALIAS pricing)

set_target_properties(pricing PROPERTIES
        OUTPUT_NAME "pricing"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)