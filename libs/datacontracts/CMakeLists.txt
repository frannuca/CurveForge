cmake_minimum_required(VERSION 3.16)
project(datacontracts LANGUAGES CXX)

# 1) Find the xsd compiler (CodeSynthesis)
find_program(XSD_EXECUTABLE NAMES xsd)
if (NOT XSD_EXECUTABLE)
    message(FATAL_ERROR "xsd compiler not found. Install CodeSynthesis XSD and ensure 'xsd' is in PATH.")
endif ()

# 2) XSD runtime headers
find_path(XSD_INCLUDE_DIR
        NAMES xsd/cxx/pre.hxx
        HINTS
        /opt/homebrew/opt/xsd/include
        /usr/local/include
        /opt/homebrew/include
)
if (NOT XSD_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find XSD headers (xsd/cxx/pre.hxx).")
endif ()
message(STATUS "Found XSD headers in: ${XSD_INCLUDE_DIR}")

# 3) Xerces-C via vcpkg / system
find_package(XercesC REQUIRED)
if (NOT TARGET XercesC::XercesC)
    message(FATAL_ERROR "Could not find Xerces-C CMake target")
endif ()
message(STATUS "Found Xerces-C: ${XercesC_VERSION}")

# 4) All schemas (for dependency tracking)
file(GLOB XSD_SCHEMAS "${CMAKE_CURRENT_SOURCE_DIR}/xsd/*.xsd")

# ---- LIST OF ROOT SCHEMAS TO GENERATE CODE FOR ----------------
# Add here all XSDs that should generate their own .hxx/.cxx
set(XSD_ROOT_SCHEMAS
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/Bond.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/commons.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/CrossCurrencySwap.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/Forward.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/Instrument.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/IRSwap.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/Leg.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/marketdata.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/Option.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/portfolio.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/quotes.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/vol.xsd
        ${CMAKE_CURRENT_SOURCE_DIR}/xsd/yield.xsd
        # add more roots here if needed, e.g. Instruments.xsd, Commons.xsd, ...
)

# 5) Build-tree layout (common dirs)
set(GEN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/datacontracts)
set(GEN_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/datacontracts)

# 6) Namespace mappings for all CurveForge schemas
set(XSD_NAMESPACE_MAP_ARGS
        --namespace-map=http://curveforge.com/portfolio=portfolio
        --namespace-map=http://curveforge.com/marketdata=marketdata
        --namespace-map=http://curveforge.com/yield=yield
        --namespace-map=http://curveforge.com/vol=vol
        --namespace-map=http://curveforge.com/quotes=quotes
        --namespace-map=http://curveforge.com/instruments=instruments
        --namespace-map=http://curveforge.com/commons=commons
        # add more namespace maps if you introduce more targetNamespace values
)

# 7) Generate code for each root schema
set(GENERATED_SOURCES "")
set(GENERATED_HEADERS "")

foreach (XSD_ROOT ${XSD_ROOT_SCHEMAS})
    # e.g. portfolio.xsd -> portfolio
    get_filename_component(SCHEMA_NAME "${XSD_ROOT}" NAME_WE)

    set(GEN_SOURCE "${GEN_SRC_DIR}/${SCHEMA_NAME}.cxx")
    set(GEN_HEADER "${GEN_INCLUDE_DIR}/${SCHEMA_NAME}.hxx")

    add_custom_command(
            OUTPUT ${GEN_SOURCE} ${GEN_HEADER}
            COMMAND ${CMAKE_COMMAND} -E make_directory
            ${GEN_SRC_DIR} ${GEN_INCLUDE_DIR}
            COMMAND ${XSD_EXECUTABLE}
            cxx-tree
            --generate-serialization
            ${XSD_NAMESPACE_MAP_ARGS}
            --output-dir ${GEN_SRC_DIR}
            ${XSD_ROOT}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${GEN_SRC_DIR}/${SCHEMA_NAME}.hxx
            ${GEN_HEADER}
            DEPENDS ${XSD_ROOT} ${XSD_SCHEMAS}
            COMMENT "Generating C++ from ${XSD_ROOT} with xsd"
            VERBATIM
    )

    list(APPEND GENERATED_SOURCES ${GEN_SOURCE})
    list(APPEND GENERATED_HEADERS ${GEN_HEADER})
endforeach ()

# 8) Static library from ALL generated sources
add_library(datacontracts SHARED ${GENERATED_SOURCES} ${GENERATED_HEADERS})
add_library(CurveForge::datacontracts ALIAS datacontracts)

target_link_libraries(datacontracts
        PUBLIC
        XercesC::XercesC
)

target_include_directories(datacontracts
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${XSD_INCLUDE_DIR}        # so consumers see xsd/cxx/*
)

target_compile_features(datacontracts PUBLIC cxx_std_17)