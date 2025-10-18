cmake_minimum_required(VERSION 3.21)

add_library(time
        src/calendar_factory.cpp
        src/calendars.cpp
        src/daycount.cpp
        src/scheduler.cpp
        src/date_modifier.cpp
        include/time/scheduler.h
)


target_include_directories(time
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)


add_library(CurveForge::time ALIAS time)

set_target_properties(time PROPERTIES
        OUTPUT_NAME "time"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)