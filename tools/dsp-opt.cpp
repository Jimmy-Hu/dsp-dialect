// ------------------------------------------------------------------
// Main entry point for the DSP dialect optimizer tool.
// This tool registers both custom DSP passes and all standard MLIR
// dialects/passes to enable end-to-end lowering from DSP to LLVM IR.
// ------------------------------------------------------------------

#include "DSP/DSPDialect.h"
#include "DSP/DSPOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
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
    // Step 1: Register Passes
    // ------------------------------------------------------------------
    // Register all standard MLIR passes.
    // This provides critical lowering pipelines like bufferization, 
    // scf-to-cf, and conversions down to the LLVM dialect.
    mlir::registerAllPasses();

    // Register our custom DSP passes (e.g., convert-dsp-to-linalg)
    mlir::dsp::registerDSPPasses();

    // ------------------------------------------------------------------
    // Step 2: Register Dialects
    // ------------------------------------------------------------------
    mlir::DialectRegistry registry;
    
    // Register all standard MLIR dialects.
    // This is required to parse and manipulate intermediate dialects
    // like Linalg, Arith, SCF, MemRef, Vector, and LLVM.
    mlir::registerAllDialects(registry);
    
    // Explicitly register our custom DSP dialect
    registry.insert<mlir::dsp::DSPDialect>();

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