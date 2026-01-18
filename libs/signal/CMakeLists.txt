cmake_minimum_required(VERSION 3.21)

set(SIGNAL_SOURCES src/ExponentialMovingAverage.cpp
        src/CrossMovingAverage.cpp
        src/SignalTransforms.cpp)
add_library(signal
        ${SIGNAL_SOURCES}
)

find_package(Boost REQUIRED COMPONENTS math)

target_link_libraries(signal PRIVATE
        Boost::math
)

# Ensure header-only Boost targets or include dirs are available for signal
if (TARGET Boost::headers)
    target_link_libraries(signal PUBLIC Boost::headers)
elseif (TARGET Boost::boost)
    target_link_libraries(signal PUBLIC Boost::boost)
else ()
    target_include_directories(signal PUBLIC ${Boost_INCLUDE_DIRS})
endif ()

target_include_directories(signal
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(CurveForge::signal ALIAS signal)

set_target_properties(signal PROPERTIES
        OUTPUT_NAME "signal"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)