# Define the mlir-dsp library
# We add DSPToLinalg.cpp to the source file list to compile our new conversion pass
add_mlir_dialect_library(MLIRDSP
    DSPDialect.cpp
    DSPOps.cpp
    DSPToLinalg.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/DSP

    DEPENDS
    MLIRDSPIncGen
    DSPPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRPass
    MLIRTransforms
)

# ------------------------------------------------------------------
# Explicitly enforce C++23 standard for the DSP Dialect library.
# This is strictly required to enable C++20 Concepts (std::floating_point) 
# and C++23 size_t literals (uz) used in our generic lowering pass.
# Note: add_mlir_dialect_library generates an object library prefixed 
# with "obj.", so we must apply the compile features to both targets.
# ------------------------------------------------------------------
target_compile_features(MLIRDSP PUBLIC cxx_std_23)
target_compile_features(obj.MLIRDSP PUBLIC cxx_std_23)