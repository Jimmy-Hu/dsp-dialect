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