# Configure the Lit test suite for the DSP dialect.

if(NOT LLVM_EXTERNAL_LIT)
    message(WARNING "LLVM_EXTERNAL_LIT is not set. Tests will not be configured.")
    return()
endif()

# ------------------------------------------------------------------
# 1. Compile the Test Generator Tool
# ------------------------------------------------------------------
find_package(TBB REQUIRED)

add_executable(GenerateDSPTests ${CMAKE_SOURCE_DIR}/tools/GenerateDSPTests.cpp)
target_compile_features(GenerateDSPTests PRIVATE cxx_std_23)
target_link_libraries(GenerateDSPTests PRIVATE TBB::tbb)

# ------------------------------------------------------------------
# 2. Setup Auto-Generation Custom Command
# ------------------------------------------------------------------
set(DSP_GENERATED_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/DSP/Generated)
file(MAKE_DIRECTORY ${DSP_GENERATED_TEST_DIR})

add_custom_command(
    OUTPUT ${DSP_GENERATED_TEST_DIR}/.generated.stamp
    COMMAND GenerateDSPTests ${DSP_GENERATED_TEST_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${DSP_GENERATED_TEST_DIR}/.generated.stamp
    DEPENDS GenerateDSPTests
    COMMENT "Auto-generating DSP dialect scalable tests into ${DSP_GENERATED_TEST_DIR}"
)

add_custom_target(GenerateDSPTestsTarget DEPENDS ${DSP_GENERATED_TEST_DIR}/.generated.stamp)

# ------------------------------------------------------------------
# 3. Compile the C++ Verification Library for JIT Execution
# ------------------------------------------------------------------
add_library(DSPTestUtils SHARED ${CMAKE_CURRENT_SOURCE_DIR}/DSP/DSPTestUtils.cpp)
target_compile_features(DSPTestUtils PRIVATE cxx_std_23)

# ------------------------------------------------------------------
# 4. Lit Configuration
# ------------------------------------------------------------------
set(DSP_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(DSP_SOURCE_DIR ${CMAKE_SOURCE_DIR})

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Inject both GenerateDSPTestsTarget and DSPTestUtils as DEPENDS
add_lit_testsuite(check-dsp "Running the DSP dialect regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS dsp-opt GenerateDSPTestsTarget DSPTestUtils
)

set_target_properties(check-dsp PROPERTIES FOLDER "Tests")