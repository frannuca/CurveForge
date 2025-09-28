cmake_minimum_required(VERSION 3.21)


add_library(asynchelpers INTERFACE)
target_sources(asynchelpers INTERFACE
        include/asynchelpers/awaitablehelpers.h
)


target_include_directories(asynchelpers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(CurveForge::asynchelpers ALIAS asynchelpers)
