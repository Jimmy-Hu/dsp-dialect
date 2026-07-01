//===- DSPOps.cpp - DSP dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DSP/DSPDialect.h"
#include "DSP/DSPOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::dsp;

// ------------------------------------------------------------------
// Define the rewrite pattern: SimplifyIDCTofDCT
// ------------------------------------------------------------------
struct SimplifyIDCTofDCT : public OpRewritePattern<IDCTOp>
{
    // Inherit from OpRewritePattern and specify IDCTOp as the matching target
    SimplifyIDCTofDCT(MLIRContext *context) : OpRewritePattern<IDCTOp>(context, /*benefit=*/1) 
    {
    }

    // The core logic for pattern matching and graph rewriting
    LogicalResult matchAndRewrite(IDCTOp idctOp, PatternRewriter &rewriter) const override
    {
        // 1. Retrieve the input source value of the IDCT operation
        Value idctInput = idctOp.getInput();

        // 2. Check if this input is produced by a `dsp.dct` operation node.
        // getDefiningOp traverses upwards to find the source operation that generated this value.
        DCTOp dctOp = idctInput.getDefiningOp<DCTOp>();

        // 3. If the defining operation cannot be found, or it is not a DCT operation, 
        // the match fails. Abort the optimization.
        if (!dctOp)
        {
            return failure();
        }

        // 4. Match successful! We have identified the pattern IDCT(DCT(x)).
        // We use the "input (x)" of the DCT node to completely replace the "output (y)" 
        // of the current IDCT node.
        rewriter.replaceOp(idctOp, dctOp.getInput());

        // Return success to notify the MLIR framework that the IR graph has been modified
        return success();
    }
};

// ------------------------------------------------------------------
// Register canonicalization patterns into the MLIR system
// ------------------------------------------------------------------

// Instruct MLIR to load these rewrite rules when attempting to optimize IDCTOp
void IDCTOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context)
{
    results.add<SimplifyIDCTofDCT>(context);
}

// If you wish to implement the inverse pattern DCT(IDCT(x)) = x in the future, 
// it should be registered here:
void DCTOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context)
{
    // results.add<SimplifyDCTofIDCT>(context);
}
