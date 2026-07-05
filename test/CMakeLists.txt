# Configure the Lit test suite for the DSP dialect.
# In LLVM/MLIR, we do not place CMakeLists.txt inside subdirectories like test/DSP/.
# Lit will automatically discover all .mlir files recursively.

# Ensure that the external llvm-lit tool is available
if(NOT LLVM_EXTERNAL_LIT)
    message(WARNING "LLVM_EXTERNAL_LIT is not set. Tests will not be configured.")
    return()
endif()

# ------------------------------------------------------------------
# 1. Compile the Test Generator Tool
# ------------------------------------------------------------------
# Find the Threading Building Blocks (TBB) library for parallel execution
find_package(TBB REQUIRED)

# Add the C++23 test generator executable
add_executable(GenerateDSPTests ${CMAKE_SOURCE_DIR}/tools/GenerateDSPTests.cpp)
target_compile_features(GenerateDSPTests PRIVATE cxx_std_23)
target_link_libraries(GenerateDSPTests PRIVATE TBB::tbb)

# ------------------------------------------------------------------
# 2. Setup Auto-Generation Custom Command
# ------------------------------------------------------------------
# Define the output directory strictly inside the build tree
set(DSP_GENERATED_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/DSP/Generated)
file(MAKE_DIRECTORY ${DSP_GENERATED_TEST_DIR})

# Create a custom command that runs the generator.
# We use a .stamp file to track whether the generation has been run to avoid
# unnecessary re-runs unless the GenerateDSPTests binary itself is recompiled.
add_custom_command(
    OUTPUT ${DSP_GENERATED_TEST_DIR}/.generated.stamp
    COMMAND GenerateDSPTests ${DSP_GENERATED_TEST_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${DSP_GENERATED_TEST_DIR}/.generated.stamp
    DEPENDS GenerateDSPTests
    COMMENT "Auto-generating DSP dialect scalable tests into ${DSP_GENERATED_TEST_DIR}"
)

# Create a custom target wrapping the stamp file
add_custom_target(GenerateDSPTestsTarget DEPENDS ${DSP_GENERATED_TEST_DIR}/.generated.stamp)

# ------------------------------------------------------------------
# 3. Lit Configuration
# ------------------------------------------------------------------
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
# Inject GenerateDSPTestsTarget as a DEPENDS so it automatically triggers before lit testing
add_lit_testsuite(check-dsp "Running the DSP dialect regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS dsp-opt GenerateDSPTestsTarget
)

# Organize the target in IDEs
set_target_properties(check-dsp PROPERTIES FOLDER "Tests")