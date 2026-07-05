// ------------------------------------------------------------------
// C++ Utility library for verifying MLIR JIT execution results.
// This will be compiled into a shared library (.so) and loaded by mlir-cpu-runner.
// ------------------------------------------------------------------

#include <cmath>
#include <cstdint>
#include <iostream>

extern "C"
{

// ------------------------------------------------------------------
// The standard MLIR ABI struct for a 2D ranked memref of floats.
// This must perfectly match the internal memory layout of memref<8x8xf32>.
// ------------------------------------------------------------------
struct MemRef2DF32
{
    float* allocated;
    float* aligned;
    intptr_t offset;
    intptr_t sizes[2];
    intptr_t strides[2];
};

// ------------------------------------------------------------------
// Verify the 8x8 DCT result against expected values with a strict epsilon.
// Returns 0 on success, 1 on failure.
// ------------------------------------------------------------------
int32_t verify_dct_result(MemRef2DF32* memref)
{
    // Set a strict epsilon for floating-point comparison
    const float epsilon{1e-5f};
    
    // Calculate the actual starting pointer using the offset
    const float* data{memref->aligned + memref->offset};
    const intptr_t strideRow{memref->strides[0]};
    const intptr_t strideCol{memref->strides[1]};

    bool passed{true};

    for (intptr_t i{0}; i < 8; ++i)
    {
        for (intptr_t j{0}; j < 8; ++j)
        {
            const float actualValue{data[i * strideRow + j * strideCol]};
            
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