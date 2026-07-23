// ------------------------------------------------------------------
// Main entry point for the DSP dialect optimizer tool.
// This version strictly registers ONLY the necessary dialects and passes
// to avoid upstream API breaking changes in InitAllDialects.h
// ------------------------------------------------------------------

#include "DSP/DSPDialect.h"
#include "DSP/DSPOps.h"

// Explicitly include only the dialects we need for lowering
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// ------------------------------------------------------------------
// Include the auto-generated pass registration logic for the DSP dialect
// ------------------------------------------------------------------
namespace mlir 
{
namespace dsp 
{
#define GEN_PASS_REGISTRATION
#include "DSP/DSPPasses.h.inc"
} // namespace dsp
} // namespace mlir

int main(const int argc, char **argv) 
{
    // ------------------------------------------------------------------
    // Step 1: Register Custom Passes
    // ------------------------------------------------------------------
    // Register ONLY our custom DSP passes (e.g., convert-dsp-to-linalg)
    mlir::dsp::registerDSPPasses();

    // ------------------------------------------------------------------
    // Step 2: Register Specific Dialects
    // ------------------------------------------------------------------
    mlir::DialectRegistry registry;
    
    // Explicitly register our custom DSP dialect
    registry.insert<mlir::dsp::DSPDialect>();

    // Explicitly register the target dialects for our lowering pass
    registry.insert<
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect,
        mlir::linalg::LinalgDialect,
        mlir::tensor::TensorDialect
    >();

    // ------------------------------------------------------------------
    // Step 3: Run the mlir-opt core logic
    // ------------------------------------------------------------------
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(
            argc, 
            argv, 
            "DSP dialect optimizer driver\n", 
            registry
        )
    );
}