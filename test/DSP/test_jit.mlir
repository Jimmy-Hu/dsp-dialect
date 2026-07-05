// RUN: dsp-opt %s --convert-dsp-to-linalg \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --lower-affine \
// RUN:   --convert-scf-to-cf \
// RUN:   --expand-strided-metadata \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-math-to-llvm \
// RUN:   --convert-arith-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=i32 \
// RUN:   -shared-libs=%dsp_obj_root/test/libDSPTestUtils%shlibext | FileCheck %s

// ------------------------------------------------------------------
// A simple MLIR test case to verify the dsp.dct lowering on CPU.
// ------------------------------------------------------------------

// 1. Declare our custom C++ verification function.
// MLIR's bufferization pass will automatically convert this tensor signature 
// to match the MemRef2DF32 pointer signature in our C++ code.
func.func private @verify_dct_result(tensor<8x8xf32>) -> i32

// 2. Define the main function that returns an integer (exit code).
func.func @main() -> i32 {
    // 3. Create a constant dense tensor (8x8) filled with the value 1.0.
    %cst_input = arith.constant dense<1.0> : tensor<8x8xf32>

    // 4. Call our custom DSP dialect operation.
    %dct_result = dsp.dct %cst_input : tensor<8x8xf32>

    // 5. Pass the computed tensor directly into our C++ verification hook.
    %status = func.call @verify_dct_result(%dct_result) : (tensor<8x8xf32>) -> i32

    // 6. Return the status from the C++ code (0 for success, 1 for failure).
    return %status : i32
}

// 7. Verify the JIT output. 
// We now rely purely on the C++ epsilon calculation. 
// If the absolute error for ANY of the 64 elements exceeds 1e-5, it prints Mismatch 
// and VERIFICATION_FAILED, causing FileCheck to fail!
// CHECK: VERIFICATION_SUCCESS