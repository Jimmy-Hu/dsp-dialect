add_mlir_dialect_library(MLIRDSP
        DSPDialect.cpp
        DSPOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/DSP

        DEPENDS
        MLIRDSPOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
