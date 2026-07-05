#ifndef DSP_OPS
#define DSP_OPS

include "DSP/DSP.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// 1. Define the DCT (Discrete Cosine Transform) operation
// Add SameOperandsAndResultType since DCT preserves tensor shape and element type
def DCTOp : DSP_Op<"dct", [Pure, SameOperandsAndResultType]> 
{
    let summary = "Discrete Cosine Transform";
    
    // Define inputs and outputs
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$res);

    // Allow this operation to have Canonicalization rules
    let hasCanonicalizer = 1;

    // Define custom assembly format for the MLIR parser
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

// 2. Define the IDCT (Inverse Discrete Cosine Transform) operation
// Add SameOperandsAndResultType since IDCT preserves tensor shape and element type
def IDCTOp : DSP_Op<"idct", [Pure, SameOperandsAndResultType]> 
{
    let summary = "Inverse Discrete Cosine Transform";
    
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$res);

    // Allow this operation to have Canonicalization rules
    let hasCanonicalizer = 1;

    // Define custom assembly format for the MLIR parser
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // DSP_OPS