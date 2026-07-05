# ------------------------------------------------------------------
# 1. Generate Dialect declarations/definitions from DSP.td
# ------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS DSP.td)
mlir_tablegen(DSPDialect.h.inc -gen-dialect-decls -dialect=dsp)
mlir_tablegen(DSPDialect.cpp.inc -gen-dialect-defs -dialect=dsp)

# ------------------------------------------------------------------
# 2. Generate Operation declarations/definitions from DSPOps.td
# ------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS DSPOps.td)
mlir_tablegen(DSPOps.h.inc -gen-op-decls)
mlir_tablegen(DSPOps.cpp.inc -gen-op-defs)

# ------------------------------------------------------------------
# Bundle both Dialect and Ops generation into a single target.
# This target name (MLIRDSPIncGen) matches the dependency required 
# by lib/DSP/CMakeLists.txt.
# ------------------------------------------------------------------
add_public_tablegen_target(MLIRDSPIncGen)

# ------------------------------------------------------------------
# 3. Generate Pass declarations from DSPPasses.td
# ------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS DSPPasses.td)
mlir_tablegen(DSPPasses.h.inc -gen-pass-decls -name DSP)
add_public_tablegen_target(DSPPassIncGen)

# ------------------------------------------------------------------
# 4. Generate Dialect Documentation
# ------------------------------------------------------------------
add_mlir_doc(DSPOps DSP Dialect/ -gen-op-doc)

# Combine all targets as dependencies for MLIR headers
add_dependencies(mlir-headers MLIRDSPIncGen DSPPassIncGen)