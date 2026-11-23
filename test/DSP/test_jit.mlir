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

// ------------------------------------------------------------------
// End-to-end JIT verification test for BOTH f32 and f64 precision.
// ------------------------------------------------------------------

// 1. Declare external C verification hooks for both f32 and f64.
// Bufferization will automatically lower these tensor arguments into unpacked MemRefs.
func.func private @verify_dct_result_f32(tensor<8x8xf32>) -> i32
func.func private @verify_dct_result_f64(tensor<8x8xf64>) -> i32

    // Call our custom DSP dialect operation.
    %dct_result = dsp.dct %cst_input : tensor<8x8xf32>

    // Call the external C function to verify the mathematical result.
    %exit_code = func.call @verify_dct_result(%dct_result) : (tensor<8x8xf32>) -> i32

    // CHECK: VERIFICATION_SUCCESS
    return %exit_code : i32
}