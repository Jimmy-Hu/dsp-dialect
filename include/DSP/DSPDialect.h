#ifndef DSP_DSPDIALECT_H
#define DSP_DSPDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// Include the auto-generated dialect header.
// MLIR TableGen appends "Dialect.h.inc" suffix regardless of the .td file name.
#include "DSP/DSPDialect.h.inc"

#endif // DSP_DSPDIALECT_H