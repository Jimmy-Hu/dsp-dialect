# Configure the Lit test suite for the DSP dialect.
# In LLVM/MLIR, we do not place CMakeLists.txt inside subdirectories like test/DSP/.
# Lit will automatically discover all .mlir files recursively.

# Ensure that the external llvm-lit tool is available
if(NOT LLVM_EXTERNAL_LIT)
    message(WARNING "LLVM_EXTERNAL_LIT is not set. Tests will not be configured.")
    return()
endif()

# Pass path configurations to Lit
set(DSP_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(DSP_SOURCE_DIR ${CMAKE_SOURCE_DIR})

# Generate the site-specific Lit configuration file from the template
configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Add the test suite target: check-dsp
add_lit_testsuite(check-dsp "Running the DSP dialect regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS dsp-opt
)

# Organize the target in IDEs
set_target_properties(check-dsp PROPERTIES FOLDER "Tests")