//===- DSPPasses.td - DSP dialect passes -------------------*- tablegen -*-===//
//
// Defines the passes for the DSP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DSP_PASSES
#define DSP_PASSES

include "mlir/Pass/PassBase.td"

// ------------------------------------------------------------------
// Define the pass that lowers the DSP dialect down to Linalg and Arith.
// ------------------------------------------------------------------
def ConvertDSPToLinalg : Pass<"convert-dsp-to-linalg", "func::FuncOp"> {
  let summary = "Lower DSP dialect operations to Linalg and Arith dialects.";
  let description = [{
    This pass converts high-level DSP operations like `dsp.dct` and 
    `dsp.idct` into lower-level linear algebra operations (`linalg.matmul`) 
    and arithmetic constants.
    
    For example, a 2D DCT on an 8x8 block is lowered to a sequence of 
    two matrix multiplications: Y = C * X * C^T.
  }];
  
  // The C++ constructor will be generated automatically
  let constructor = "mlir::dsp::createConvertDSPToLinalgPass()";
  
  // Specify the dialects this pass might generate or depend on
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // DSP_PASSES