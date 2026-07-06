//===- dsp-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DSP/DSPDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Auto-generated pass registration logic
namespace mlir 
{
namespace dsp 
{
#define GEN_PASS_REGISTRATION
#include "DSP/DSPPasses.h.inc"
} // namespace dsp
} // namespace mlir

int main(int argc, char **argv) 
{
    // Register all standard MLIR passes
    mlir::registerAllPasses();
    
    // Register custom DSP passes (e.g., convert-dsp-to-linalg)
    mlir::dsp::registerDSPPasses();

    mlir::DialectRegistry registry;
    
    // Register all standard MLIR dialects (required for Linalg, Tensor, etc.)
    mlir::registerAllDialects(registry);
    
    // Register custom DSP dialect
    registry.insert<mlir::dsp::DSPDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "DSP optimizer driver\n", registry));
}