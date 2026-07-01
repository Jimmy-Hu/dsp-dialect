//===- DSPOps.td - DSP dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DSP_OPS
#define DSP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DSP_Dialect : Dialect {
  let name = "dsp";
  let summary = "A dialect for Digital Signal Processing.";
  let description = [{
    The DSP dialect defines high-level mathematical operations like DCT and FFT
    to enable algebraic simplifications before lowering to hardware or loops.
  }];
  let cppNamespace = "::mlir::dsp";
}

class DSP_Op<string mnemonic, list<Trait> traits = []> :
    Op<DSP_Dialect, mnemonic, traits>;

def DCTOp : DSP_Op<"dct", [Pure]> {
  let summary = "Discrete Cosine Transform";
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$res);

  let hasCanonicalizer = 1; 
}

def IDCTOp : DSP_Op<"idct", [Pure]> {
  let summary = "Inverse Discrete Cosine Transform";
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$res);

  let hasCanonicalizer = 1; 
}

#endif // DSP_OPS
