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
// Include the auto-generated pass registration logic
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
    // Register all standard MLIR dialects and passes
    mlir::registerAllPasses();

    // Register our custom DSP passes
    mlir::dsp::registerDSPPasses();

    // Initialize the MLIR context with all registered dialects
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    
    // Explicitly register our custom DSP dialect
    registry.insert<mlir::dsp::DSPDialect>();

    // Run the mlir-opt main tool logic
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "DSP dialect optimizer driver\n", registry));
}