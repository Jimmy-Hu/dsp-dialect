// RUN: dsp-opt %s --convert-dsp-to-linalg \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --lower-affine \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-cf-to-llvm \
// RUN:   --expand-strided-metadata \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-math-to-llvm \
// RUN:   --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=i32 \
// RUN:   -shared-libs=%dsp_obj_root/test/libDSPTestUtils%shlibext | FileCheck %s

// 1. Declare the external C verification function.
// Since we run one-shot-bufferize, we can just declare it taking a tensor
// and the bufferization pass will automatically convert it to a memref!
func.func private @verify_dct_result(tensor<8x8xf32>) -> i32

// 2. Define the main function that returns an integer (exit code).
func.func @main() -> i32 {
    // Create a constant dense tensor (8x8) filled with the value 1.0.
    %cst_input = arith.constant dense<1.0> : tensor<8x8xf32>

    // Call our custom DSP dialect operation.
    %dct_result = dsp.dct %cst_input : tensor<8x8xf32>

    // Call the external C function to verify the mathematical result.
    %exit_code = func.call @verify_dct_result(%dct_result) : (tensor<8x8xf32>) -> i32

    // CHECK: VERIFICATION_SUCCESS
    return %exit_code : i32
}