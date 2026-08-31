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
  // CHECK-NOT: dsp.idct

  // The output should be directly wired to the input argument
  // CHECK: return %[[ARG0]] : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// ------------------------------------------------------------------
// Test Case 2: DCT(IDCT(x)) = x
// ------------------------------------------------------------------
// CHECK-LABEL: func.func @test_dct_idct_cancellation
// CHECK-SAME: (%[[ARG1:.*]]: tensor<8x8xf32>)
func.func @test_dct_idct_cancellation(%arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  
  // Original expression: y = DCT(IDCT(x))
  %0 = dsp.idct %arg1 : tensor<8x8xf32>
  %1 = dsp.dct %0 : tensor<8x8xf32>
  
  // Ensure both operations are eliminated
  // CHECK-NOT: dsp.idct
  // CHECK-NOT: dsp.dct

  // The output should be directly wired to the input argument
  // CHECK: return %[[ARG1]] : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// ------------------------------------------------------------------
// Test Case 3: Nested IDCT(IDCT(DCT(DCT(x)))) = x
// ------------------------------------------------------------------
// CHECK-LABEL: func.func @test_nested_cancellation
// CHECK-SAME: (%[[ARG2:.*]]: tensor<8x8xf32>)
func.func @test_nested_cancellation(%arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  
  // Original expression: y = IDCT(IDCT(DCT(DCT(x))))
  %0 = dsp.dct %arg2 : tensor<8x8xf32>
  %1 = dsp.dct %0 : tensor<8x8xf32>
  %2 = dsp.idct %1 : tensor<8x8xf32>
  %3 = dsp.idct %2 : tensor<8x8xf32>
  
  // Ensure all operations are eliminated
  // CHECK-NOT: dsp.dct
  // CHECK-NOT: dsp.idct

  // The output should be directly wired to the input argument
  // CHECK: return %[[ARG2]] : tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}

// ------------------------------------------------------------------
// Test Case 4: Alternating IDCT(DCT(IDCT(DCT(x)))) = x
// ------------------------------------------------------------------
// CHECK-LABEL: func.func @test_alternating_cancellation
// CHECK-SAME: (%[[ARG3:.*]]: tensor<8x8xf32>)
func.func @test_alternating_cancellation(%arg3: tensor<8x8xf32>) -> tensor<8x8xf32> {
  
  // Original expression: y = IDCT(DCT(IDCT(DCT(x))))
  %0 = dsp.dct %arg3 : tensor<8x8xf32>
  %1 = dsp.idct %0 : tensor<8x8xf32>
  %2 = dsp.dct %1 : tensor<8x8xf32>
  %3 = dsp.idct %2 : tensor<8x8xf32>
  
  // Ensure all operations are eliminated
  // CHECK-NOT: dsp.dct
  // CHECK-NOT: dsp.idct

  // The output should be directly wired to the input argument
  // CHECK: return %[[ARG3]] : tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}

