// ------------------------------------------------------------------
// C++ Utility library for verifying MLIR JIT execution results.
// This will be compiled into a shared library (.so) and loaded by mlir-runner.
// ------------------------------------------------------------------

#include <cmath>
#include <cstdint>
#include <iostream>

extern "C"
{

// ------------------------------------------------------------------
// Verify the 8x8 DCT result against expected values.
// In MLIR, by default, MemRefs are passed to external functions as 
// unpacked arguments (allocated, aligned, offset, sizes..., strides...).
// ------------------------------------------------------------------
int32_t verify_dct_result(
    float* allocated, 
    float* aligned, 
    intptr_t offset, 
    intptr_t size0, 
    intptr_t size1, 
    intptr_t stride0, 
    intptr_t stride1)
{
    // Set a strict epsilon for floating-point comparison
    const float epsilon{1e-4f};
    
    // Calculate the actual starting pointer using the offset
    const float* data{aligned + offset};

    bool passed{true};

    for (intptr_t i{0}; i < size0; ++i)
    {
        for (intptr_t j{0}; j < size1; ++j)
        {
            const float actualValue{data[i * stride0 + j * stride1]};
            
            // Expected mathematical result: DC component is 8.0, all AC components are 0.0
            const float expectedValue{(i == 0 && j == 0) ? 8.0f : 0.0f};

            if (std::abs(actualValue - expectedValue) > epsilon)
            {
                std::cerr << "Mismatch at [" << i << ", " << j << "]: "
                          << "Expected " << expectedValue << ", but got " << actualValue << "\n";
                passed = false;
            }
        }
    }

    if (passed)
    {
        std::cout << "VERIFICATION_SUCCESS\n";
        return 0; // Equivalent to EXIT_SUCCESS
    }

    std::cout << "VERIFICATION_FAILED\n";
    return 1; // Equivalent to EXIT_FAILURE
}

} // extern "C"