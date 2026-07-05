// ------------------------------------------------------------------
// A standalone C++23 tool to auto-generate both exhaustive and 
// randomized deeply nested MLIR canonicalization tests.
// ------------------------------------------------------------------

#include "DSP/DSPDialect.h"
#include "DSP/DSPOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>
#include <vector>

// Include the auto-generated pass definitions from TableGen
namespace mlir 
{
namespace dsp 
{
#define GEN_PASS_DEF_CONVERTDSPTOLINALG
#include "DSP/DSPPasses.h.inc"
} // namespace dsp
} // namespace mlir

using namespace mlir;
using namespace mlir::dsp;

namespace 
{

// ------------------------------------------------------------------
// Helper function to generate the 8x8 DCT-II constant coefficient matrix.
// Elements are calculated with maximum precision (double) before 
// being stored as single-precision floats required by MLIR f32 types.
// ------------------------------------------------------------------
constexpr std::vector<float> generateDCTCoefficientMatrix()
{
    // The matrix dimension is fixed at 8x8 for standard block-based DCT
    constexpr std::size_t n{8uz};
    std::vector<float> matrix;
    matrix.reserve(n * n);

    // Calculate DCT-II coefficients using exact math functions
    for (std::size_t i = 0uz; i < n; ++i)
    {
        for (std::size_t j = 0uz; j < n; ++j)
        {
            // The scaling factor is 1/sqrt(N) for the DC component (i=0), 
            // and sqrt(2/N) for the AC components (i>0).
            const double alpha = (i == 0uz) ? std::sqrt(1.0 / static_cast<double>(n)) 
                                            : std::sqrt(2.0 / static_cast<double>(n));
            
            // Standard DCT-II formula for the current element
            const double value = alpha * std::cos((M_PI * static_cast<double>(i) * (2.0 * static_cast<double>(j) + 1.0)) / 
                                                  (2.0 * static_cast<double>(n)));
            
            // Cast down to single precision float for the tensor array
            matrix.emplace_back(static_cast<float>(value));
        }
    }
    
    return matrix;
}

// ------------------------------------------------------------------
// Helper function to transpose the 8x8 DCT matrix
// ------------------------------------------------------------------
constexpr std::vector<float> transposeMatrix(
    const std::vector<float>& inputMatrix, 
    const std::size_t n)
{
    std::vector<float> transposed;
    transposed.reserve(n * n);

    for (std::size_t i = 0uz; i < n; ++i)
    {
        for (std::size_t j = 0uz; j < n; ++j)
        {
            // Access elements in column-major order to build the transposed row-major matrix
            transposed.emplace_back(inputMatrix[j * n + i]);
        }
    }

    return transposed;
}

// ------------------------------------------------------------------
// The conversion pattern to lower dsp.dct to linalg.matmul
// 
// Mathematically, 2D DCT of an 8x8 block X is: Y = C * X * C^T
// We lower this into a sequence of MLIR operations:
// 1. Create a constant tensor for C (DCT matrix)
// 2. Create a constant tensor for C_T (Transposed DCT matrix)
// 3. Compute Temp = linalg.matmul(C, X)
// 4. Compute Y = linalg.matmul(Temp, C_T)
// ------------------------------------------------------------------
struct DCTOpConversion : public OpConversionPattern<DCTOp> 
{
    using OpConversionPattern<DCTOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        DCTOp op, 
        OpAdaptor adaptor, 
        ConversionPatternRewriter &rewriter) const override 
    {
        Location loc = op.getLoc();
        
        // 1. Ensure the input tensor matches our expected 8x8 f32 shape
        RankedTensorType inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
        
        if (!inputType || inputType.getRank() != 2 || 
            inputType.getShape()[0] != 8 || inputType.getShape()[1] != 8 || 
            !inputType.getElementType().isF32()) 
        {
            return rewriter.notifyMatchFailure(op, "Expected an 8x8 f32 tensor as input for DCT");
        }

        // Define the tensor type and shape expected for all intermediate computations
        RankedTensorType tensorType = RankedTensorType::get({8, 8}, rewriter.getF32Type());

        // 2. Pre-compute and create MLIR Constants for C and C^T matrices
        std::vector<float> dctMatrixElements = generateDCTCoefficientMatrix();
        std::vector<float> transposedMatrixElements = transposeMatrix(dctMatrixElements, 8uz);

        // Build DenseElementsAttr to hold the float arrays for the MLIR framework
        DenseElementsAttr cAttr = DenseElementsAttr::get(tensorType, ArrayRef<float>(dctMatrixElements));
        DenseElementsAttr cTAttr = DenseElementsAttr::get(tensorType, ArrayRef<float>(transposedMatrixElements));

        // Materialize the arithmetic constant operations in the IR
        Value cTensor = rewriter.create<arith::ConstantOp>(loc, tensorType, cAttr);
        Value cTTensor = rewriter.create<arith::ConstantOp>(loc, tensorType, cTAttr);

        // 3. Initialize zero tensors to hold the results of the linalg.matmul operations
        // MLIR requires explicit allocation of output buffer representations (empty tensors)
        Value emptyTensor1 = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{8, 8}, rewriter.getF32Type());
        Value emptyTensor2 = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{8, 8}, rewriter.getF32Type());
        
        Value zeroVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
        
        // Fill the empty tensors with zero to initialize the accumulators for matmul
        Value zeroTensor1 = rewriter.create<linalg::FillOp>(loc, zeroVal, emptyTensor1).getResult(0);
        Value zeroTensor2 = rewriter.create<linalg::FillOp>(loc, zeroVal, emptyTensor2).getResult(0);

        // 4. Create the first matrix multiplication: Temp = C * X
        // The input 'X' is provided by the matched operation's adaptor
        auto matmul1 = rewriter.create<linalg::MatmulOp>(
            loc, 
            TypeRange{tensorType},      // Output type
            ValueRange{cTensor, adaptor.getInput()}, // Inputs (C, X)
            ValueRange{zeroTensor1}     // Initialized output buffer (Accumulator)
        );
        Value tempTensor = matmul1.getResult(0);

        // 5. Create the second matrix multiplication: Y = Temp * C^T
        auto matmul2 = rewriter.create<linalg::MatmulOp>(
            loc, 
            TypeRange{tensorType},      // Output type
            ValueRange{tempTensor, cTTensor}, // Inputs (Temp, C^T)
            ValueRange{zeroTensor2}     // Initialized output buffer (Accumulator)
        );

        // 6. Replace the original dsp.dct operation with the final resulting tensor Y
        rewriter.replaceOp(op, matmul2.getResult(0));

        return success();
    }
};

// ------------------------------------------------------------------
// The Pass Implementation Class
// Orchestrates the conversion process over an entire MLIR Function.
// ------------------------------------------------------------------
struct ConvertDSPToLinalgPass : public impl::ConvertDSPToLinalgBase<ConvertDSPToLinalgPass> 
{
    void runOnOperation() override 
    {
        func::FuncOp func = getOperation();
        MLIRContext *context = &getContext();

        // Step 1: Define the Conversion Target (Legality Rules)
        ConversionTarget target(*context);
        
        // We want to lower INTO these standard dialects, so they are explicitly legal
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<linalg::LinalgDialect>();
        target.addLegalDialect<tensor::TensorDialect>();
        
        // We explicitly declare the operation we want to eliminate as ILLEGAL.
        // If the pass finishes and any dsp.dct remains, the compilation will fail.
        target.addIllegalOp<DCTOp>();

        // Step 2: Populate the Rewrite Patterns
        RewritePatternSet patterns(context);
        patterns.add<DCTOpConversion>(context);

        // Step 3: Apply the partial conversion to the function
        if (failed(applyPartialConversion(func, target, std::move(patterns)))) 
        {
            signalPassFailure();
        }
    }
};

} // anonymous namespace

// ------------------------------------------------------------------
// The factory function declared by TableGen
// ------------------------------------------------------------------
std::unique_ptr<Pass> mlir::dsp::createConvertDSPToLinalgPass() 
{
    return std::make_unique<ConvertDSPToLinalgPass>();
}