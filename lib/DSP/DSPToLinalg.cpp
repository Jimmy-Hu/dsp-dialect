// ------------------------------------------------------------------
// Lowering Pass: Converts DSP dialect operations (like dsp.dct) 
// into Linalg and Arith dialect operations.
// Supports generic floating-point types and variable block sizes 
// via dynamic type and size dispatch.
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

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>

// Include the auto-generated pass definitions from TableGen
namespace mlir 
{
namespace dsp 
{
#define GEN_PASS_DECL
#define GEN_PASS_DEF_CONVERTDSPTOLINALG
#include "DSP/DSPPasses.h.inc"
} // namespace dsp
} // namespace mlir

using namespace mlir;
using namespace mlir::dsp;

namespace 
{

// ------------------------------------------------------------------
// Helper template to generate the NxN DCT-II constant coefficient matrix.
// Constrained to floating-point types using C++20 concepts and fully 
// parameterized by size N to avoid any dynamic memory allocation.
// ------------------------------------------------------------------
template <std::floating_point T, std::size_t N>
constexpr std::array<T, N * N> generateDCTCoefficientMatrix()
{
    std::array<T, N * N> matrix{};

    constexpr double kPi{3.14159265358979323846};

    for (std::size_t i{0}; i < N; ++i)
    {
        for (std::size_t j{0}; j < N; ++j)
        {
            const double alpha{
                (i == 0) ? std::sqrt(1.0 / static_cast<double>(N)) 
                         : std::sqrt(2.0 / static_cast<double>(N))
            };
            
            const double value{
                alpha * std::cos(
                    (kPi * static_cast<double>(i) * (2.0 * static_cast<double>(j) + 1.0)) / 
                    (2.0 * static_cast<double>(N))
                )
            };
            
            matrix[i * N + j] = static_cast<T>(value);
        }
    }
    
    return matrix;
}

// ------------------------------------------------------------------
// Helper template to transpose the NxN DCT matrix
// ------------------------------------------------------------------
template <std::floating_point T, std::size_t N>
constexpr std::array<T, N * N> transposeMatrix(
    const std::array<T, N * N>& inputMatrix)
{
    std::array<T, N * N> transposed{};

    for (std::size_t i{0}; i < N; ++i)
    {
        for (std::size_t j{0}; j < N; ++j)
        {
            transposed[i * N + j] = inputMatrix[j * N + i];
        }
    }

    return transposed;
}

// ------------------------------------------------------------------
// The generic conversion pattern to lower dsp.dct to linalg.matmul
// ------------------------------------------------------------------
struct DCTOpConversion : public OpConversionPattern<DCTOp> 
{
    using OpConversionPattern<DCTOp>::OpConversionPattern;

private:
    // ------------------------------------------------------------------
    // A highly generic inner worker function.
    // Encapsulates lowering logic parameterized by BOTH type T and size N.
    // ------------------------------------------------------------------
    template <std::floating_point T, std::size_t N>
    LogicalResult lowerDCTWithSpecificTypeAndSize(
        DCTOp op, 
        OpAdaptor adaptor, 
        ConversionPatternRewriter &rewriter,
        FloatType mlirFloatType) const
    {
        const Location loc = op.getLoc();
        const RankedTensorType tensorType = 
            RankedTensorType::get({static_cast<int64_t>(N), static_cast<int64_t>(N)}, mlirFloatType);

        // 1. Generate generic coefficients and transpose purely on the stack
        const std::array<T, N * N> dctMatrixElements{generateDCTCoefficientMatrix<T, N>()};
        const std::array<T, N * N> transposedMatrixElements{transposeMatrix<T, N>(dctMatrixElements)};

        // 2. Build dense attributes from stack arrays
        const DenseElementsAttr cAttr = DenseElementsAttr::get(tensorType, ArrayRef<T>(dctMatrixElements));
        const DenseElementsAttr cTAttr = DenseElementsAttr::get(tensorType, ArrayRef<T>(transposedMatrixElements));

        const Value cTensor = rewriter.create<arith::ConstantOp>(loc, cAttr);
        const Value cTTensor = rewriter.create<arith::ConstantOp>(loc, cTAttr);

        // 3. Initialize dynamic empty tensors
        const Value emptyTensor1 = rewriter.create<tensor::EmptyOp>(
            loc, ArrayRef<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(N)}, mlirFloatType);
        const Value emptyTensor2 = rewriter.create<tensor::EmptyOp>(
            loc, ArrayRef<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(N)}, mlirFloatType);
        
        const Value zeroVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(mlirFloatType, 0.0));
        
        const Value zeroTensor1 = rewriter.create<linalg::FillOp>(loc, zeroVal, emptyTensor1).getResult(0);
        const Value zeroTensor2 = rewriter.create<linalg::FillOp>(loc, zeroVal, emptyTensor2).getResult(0);

        // 4. Sequence matrix multiplications
        // Cast input to a standard Value to satisfy C++23 ValueRange initializer list deduction
        const Value inputVal = adaptor.getInput();

        auto matmul1 = rewriter.create<linalg::MatmulOp>(
            loc, 
            ValueRange{cTensor, inputVal}, 
            ValueRange{zeroTensor1}
        );
        const Value tempTensor = matmul1.getResult(0);

        auto matmul2 = rewriter.create<linalg::MatmulOp>(
            loc, 
            ValueRange{tempTensor, cTTensor}, 
            ValueRange{zeroTensor2}
        );

        // 5. Replace original operation
        rewriter.replaceOp(op, matmul2.getResult(0));

        return success();
    }

    // ------------------------------------------------------------------
    // Runtime size dispatcher to map dynamic IR shape to static N
    // ------------------------------------------------------------------
    template <std::floating_point T>
    LogicalResult dispatchSize(
        DCTOp op, 
        OpAdaptor adaptor, 
        ConversionPatternRewriter &rewriter,
        FloatType mlirFloatType,
        const int64_t matrixSize) const
    {
        if (matrixSize == 4)
        {
            return lowerDCTWithSpecificTypeAndSize<T, 4uz>(op, adaptor, rewriter, mlirFloatType);
        }
        else if (matrixSize == 8)
        {
            return lowerDCTWithSpecificTypeAndSize<T, 8uz>(op, adaptor, rewriter, mlirFloatType);
        }
        else if (matrixSize == 16)
        {
            return lowerDCTWithSpecificTypeAndSize<T, 16uz>(op, adaptor, rewriter, mlirFloatType);
        }
        else if (matrixSize == 32)
        {
            return lowerDCTWithSpecificTypeAndSize<T, 32uz>(op, adaptor, rewriter, mlirFloatType);
        }

        return rewriter.notifyMatchFailure(
            op, "Unsupported DCT matrix size! (Currently supporting 4x4, 8x8, 16x16, 32x32)");
    }

public:
    // ------------------------------------------------------------------
    // The main dispatcher. Interrogates the IR type and shape at runtime.
    // ------------------------------------------------------------------
    LogicalResult matchAndRewrite(
        DCTOp op, 
        OpAdaptor adaptor, 
        ConversionPatternRewriter &rewriter) const override 
    {
        const RankedTensorType inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
        
        if (!inputType || inputType.getRank() != 2) 
        {
            return rewriter.notifyMatchFailure(op, "Expected a 2D ranked tensor as input for DCT");
        }

        const int64_t dim0{inputType.getShape()[0]};
        const int64_t dim1{inputType.getShape()[1]};

        if (dim0 != dim1) 
        {
            return rewriter.notifyMatchFailure(op, "Expected a square tensor as input for DCT");
        }

        const Type elemType{inputType.getElementType()};

        if (elemType.isF32())
        {
            return dispatchSize<float>(
                op, adaptor, rewriter, cast<FloatType>(elemType), dim0);
        }
        else if (elemType.isF64())
        {
            return dispatchSize<double>(
                op, adaptor, rewriter, cast<FloatType>(elemType), dim0);
        }

        return rewriter.notifyMatchFailure(
            op, "Unsupported floating-point type for DCT! Only f32 and f64 are currently supported."
        );
    }
};

// ------------------------------------------------------------------
// The Pass Implementation Class
// ------------------------------------------------------------------
struct ConvertDSPToLinalgPass : public mlir::dsp::impl::ConvertDSPToLinalgBase<ConvertDSPToLinalgPass> 
{
    void runOnOperation() override 
    {
        func::FuncOp func = getOperation();
        MLIRContext *context = &getContext();

        ConversionTarget target(*context);
        
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<linalg::LinalgDialect>();
        target.addLegalDialect<tensor::TensorDialect>();
        
        target.addIllegalOp<DCTOp>();

        RewritePatternSet patterns(context);
        patterns.add<DCTOpConversion>(context);

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
namespace mlir 
{
namespace dsp 
{

std::unique_ptr<Pass> createConvertDSPToLinalgPass() 
{
    return std::make_unique<ConvertDSPToLinalgPass>();
}

} // namespace dsp
} // namespace mlir