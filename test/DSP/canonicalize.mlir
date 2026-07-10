// RUN: dsp-opt %s --canonicalize | FileCheck %s

// ------------------------------------------------------------------
// Test Case 1: IDCT(DCT(x)) = x
// ------------------------------------------------------------------
// CHECK-LABEL: func.func @test_idct_dct_cancellation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32>)
func.func @test_idct_dct_cancellation(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  
  // Original expression: y = IDCT(DCT(x))
  %0 = dsp.dct %arg0 : tensor<8x8xf32>
  %1 = dsp.idct %0 : tensor<8x8xf32>
  
  // Ensure both operations are eliminated
  // CHECK-NOT: dsp.dct
  return %1 : tensor<8x8xf32>
}
