#ifndef DSP_OPS
#define DSP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// 1. Define the DSP Dialect
def DSP_Dialect : Dialect {
  let name = "dsp";
  let summary = "A dialect for Digital Signal Processing.";
  let description = [{
    The DSP dialect defines high-level mathematical operations like DCT and FFT
    to enable algebraic simplifications before lowering to hardware or loops.
  }];
  let cppNamespace = "::mlir::dsp";
}

// 2. Define a base class for DSP operations
class DSP_Op<string mnemonic, list<Trait> traits = []> :
    Op<DSP_Dialect, mnemonic, traits>;

// 3. Define the DCT (Discrete Cosine Transform) operation
// Add SameOperandsAndResultType since DCT preserves tensor shape and element type
def DCTOp : DSP_Op<"dct", [Pure, SameOperandsAndResultType]> {
  let summary = "Discrete Cosine Transform";
  
  // Define inputs and outputs
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$res);

  // Allow this operation to have Canonicalization rules
  let hasCanonicalizer = 1;

  // Define custom assembly format for the MLIR parser
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

// 4. Define the IDCT (Inverse Discrete Cosine Transform) operation
// Add SameOperandsAndResultType since IDCT preserves tensor shape and element type
def IDCTOp : DSP_Op<"idct", [Pure, SameOperandsAndResultType]> {
  let summary = "Inverse Discrete Cosine Transform";
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$res);

  // Allow this operation to have Canonicalization rules
  let hasCanonicalizer = 1;

  // Define custom assembly format for the MLIR parser
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // DSP_OPS