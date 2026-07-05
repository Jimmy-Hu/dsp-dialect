add_mlir_dialect(DSP dsp)
add_mlir_doc(DSP DSP Dialect/ -gen-dialect-doc)

# ------------------------------------------------------------------
# Add TableGen command to generate Pass declarations
# ------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS DSPPasses.td)
mlir_tablegen(DSPPasses.h.inc -gen-pass-decls -name DSP)
add_public_tablegen_target(DSPPassIncGen)

add_dependencies(mlir-headers DSPIncGen DSPPassIncGen)