//===- DSPDialect.td - DSP dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DSP_DIALECT
#define DSP_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DSP dialect definition.
//===----------------------------------------------------------------------===//

def DSP_Dialect : Dialect {
    let name = "dsp";
    let summary = "A dsp out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::dsp";
}

//===----------------------------------------------------------------------===//
// Base dsp operation definition.
//===----------------------------------------------------------------------===//

class DSP_Op<string mnemonic, list<Trait> traits = []> :
        Op<DSP_Dialect, mnemonic, traits>;

#endif // DSP_DIALECT