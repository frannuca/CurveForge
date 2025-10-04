cmake_minimum_required(VERSION 3.21)

# Bond Pricing Examples executable
add_executable(bond_pricing_examples
        bond_pricing_examples.cpp
)

target_link_libraries(bond_pricing_examples
        PRIVATE
        CurveForge::pricing
)

set_target_properties(bond_pricing_examples PROPERTIES
        OUTPUT_NAME "bond_pricing_examples"
)
