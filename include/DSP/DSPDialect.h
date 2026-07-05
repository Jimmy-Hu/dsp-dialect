#ifndef DSP_DSPDIALECT_H
#define DSP_DSPDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// Include the auto-generated dialect header.
// MLIR TableGen appends "Dialect.h.inc" suffix regardless of the .td file name.
#include "DSP/DSPDialect.h.inc"

namespace mlir
{
namespace dsp
{
    class DSPDialect : public ::mlir::Dialect
    {
    public:
        explicit DSPDialect(::mlir::MLIRContext *context);
        
        static ::llvm::StringRef getDialectNamespace() 
        {
            return "dsp";
        }
        
        void initialize();
    };
} // namespace dsp
} // namespace mlir

#endif // DSP_DSPDIALECT_H