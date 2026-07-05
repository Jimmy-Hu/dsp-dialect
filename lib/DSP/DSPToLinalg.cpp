// ------------------------------------------------------------------
// Lowering Pass: Converts DSP dialect operations (like dsp.dct) 
// into Linalg and Arith dialect operations.
// Supports generic floating-point types via dynamic type dispatch.
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
// Helper template to generate the 8x8 DCT-II constant coefficient matrix.
// Constrained to floating-point types using C++20 concepts.
// Calculations are internally performed using double to guarantee 
// maximum precision before being cast to the target type T.
// ------------------------------------------------------------------
template <std::floating_point T>
constexpr std::array<T, 64> generateDCTCoefficientMatrix()
{
    constexpr std::size_t n{8};
    std::array<T, 64> matrix{};

    constexpr double kPi{3.14159265358979323846};

    for (std::size_t i{0}; i < n; ++i)
    {
        for (std::size_t j{0}; j < n; ++j)
        {
            const double alpha{
                (i == 0) ? std::sqrt(1.0 / static_cast<double>(n)) 
                         : std::sqrt(2.0 / static_cast<double>(n))
            };
            
            const double value{
                alpha * std::cos(
                    (kPi * static_cast<double>(i) * (2.0 * static_cast<double>(j) + 1.0)) / 
                    (2.0 * static_cast<double>(n))
                )
            };
            
            matrix[i * n + j] = static_cast<T>(value);
        }
    }
    
    return matrix;
}

// ------------------------------------------------------------------
// Helper template to transpose the 8x8 DCT matrix
// ------------------------------------------------------------------
template <std::floating_point T>
constexpr std::array<T, 64> transposeMatrix(
    const std::array<T, 64>& inputMatrix)
{
    constexpr std::size_t n{8};
    std::array<T, 64> transposed{};

    for (std::size_t i{0}; i < n; ++i)
    {
        for (std::size_t j{0}; j < n; ++j)
        {
            transposed[i * n + j] = inputMatrix[j * n + i];
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
    // It encapsulates all the lowering logic and accepts the target 
    // C++ type T and its corresponding MLIR FloatType.
    // ------------------------------------------------------------------
    template <std::floating_point T>
    LogicalResult lowerDCTWithSpecificType(
        DCTOp op, 
        OpAdaptor adaptor, 
        ConversionPatternRewriter &rewriter,
        FloatType mlirFloatType) const
    {
        const Location loc{op.getLoc()};
        const RankedTensorType tensorType{RankedTensorType::get({8, 8}, mlirFloatType)};

        // 1. Generate generic coefficients and transpose
        const std::array<T, 64> dctMatrixElements{generateDCTCoefficientMatrix<T>()};
        const std::array<T, 64> transposedMatrixElements{transposeMatrix<T>(dctMatrixElements)};

        // 2. Build dense attributes. DenseElementsAttr gracefully handles ArrayRef<T>
        const DenseElementsAttr cAttr{DenseElementsAttr::get(tensorType, ArrayRef<T>(dctMatrixElements))};
        const DenseElementsAttr cTAttr{DenseElementsAttr::get(tensorType, ArrayRef<T>(transposedMatrixElements))};

        const Value cTensor{rewriter.create<arith::ConstantOp>(loc, cAttr)};
        const Value cTTensor{rewriter.create<arith::ConstantOp>(loc, cTAttr)};

        // 3. Initialize dynamic empty tensors based on the resolved MLIR FloatType
        const Value emptyTensor1{rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{8, 8}, mlirFloatType)};
        const Value emptyTensor2{rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{8, 8}, mlirFloatType)};
        
        // Use rewriter.getFloatAttr(Type, double) to dynamically create the zero value
        const Value zeroVal{rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(mlirFloatType, 0.0))};
        
        const Value zeroTensor1{rewriter.create<linalg::FillOp>(loc, zeroVal, emptyTensor1).getResult(0)};
        const Value zeroTensor2{rewriter.create<linalg::FillOp>(loc, zeroVal, emptyTensor2).getResult(0)};

        // 4. Sequence matrix multiplications
        auto matmul1 = rewriter.create<linalg::MatmulOp>(
            loc, 
            ValueRange{cTensor, adaptor.getInput()}, 
            ValueRange{zeroTensor1}
        );
        const Value tempTensor{matmul1.getResult(0)};

        auto matmul2 = rewriter.create<linalg::MatmulOp>(
            loc, 
            ValueRange{tempTensor, cTTensor}, 
            ValueRange{zeroTensor2}
        );

        // 5. Replace original operation
        rewriter.replaceOp(op, matmul2.getResult(0));

        return success();
    }

public:
    // ------------------------------------------------------------------
    // The main dispatcher. It interrogates the IR type at runtime 
    // and statically dispatches to the correct generic implementation.
    // ------------------------------------------------------------------
    LogicalResult matchAndRewrite(
        DCTOp op, 
        OpAdaptor adaptor, 
        ConversionPatternRewriter &rewriter) const override 
    {
        const RankedTensorType inputType{dyn_cast<RankedTensorType>(op.getInput().getType())};
        
        if (!inputType || inputType.getRank() != 2 || 
            inputType.getShape()[0] != 8 || inputType.getShape()[1] != 8) 
        {
            return rewriter.notifyMatchFailure(op, "Expected an 8x8 tensor as input for DCT");
        }

        // Interrogate the element type from the parsed IR
        const Type elemType{inputType.getElementType()};

        // Dynamically route to the strongly-typed generic lowering worker
        if (elemType.isF32())
        {
            return lowerDCTWithSpecificType<float>(
                op, adaptor, rewriter, cast<FloatType>(elemType));
        }
        else if (elemType.isF64())
        {
            return lowerDCTWithSpecificType<double>(
                op, adaptor, rewriter, cast<FloatType>(elemType));
        }

        // We can easily expand this to f16, bf16, etc. in the future!
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