# MLIR DSP Dialect

A modern, out-of-tree [MLIR](https://mlir.llvm.org/) dialect designed for Digital Signal Processing (DSP). This project demonstrates how to build a high-level domain-specific language in MLIR, optimize it through canonicalization, and lower it down to executable CPU code via Linear Algebra (`linalg`) and MLIR's JIT infrastructure.

## ✨ Key Features

* **Custom DSP Operations:** Introduces high-level signal processing operations like `dsp.dct` (Discrete Cosine Transform) and `dsp.idct` (Inverse DCT) for 2D tensors.

* **Algebraic Simplification:** Implements powerful canonicalization patterns (e.g., `IDCT(DCT(x)) -> x` and `DCT(IDCT(x)) -> x`) to eliminate redundant transformations at the AST/IR level.

* **Generic Linalg Lowering:** Converts `dsp.dct` into sequences of `linalg.matmul` operations ($Y = C \times X \times C^T$). The conversion pass leverages **C++23** and **C++20 Concepts** (`std::floating_point`) to generate static, highly generic constant coefficient matrices at compile time without dynamic memory allocation.

* **JIT Execution & Verification:** Provides an end-to-end testing pipeline using `mlir-cpu-runner`. It compiles MLIR down to LLVM IR, executes it on the host CPU, and verifies the floating-point results using a custom C++ shared library (`libDSPTestUtils.so`).

* **Scalable Stress Testing:** Includes a custom parallel test generator (`GenerateDSPTests`) powered by **Intel TBB** to dynamically construct massive MLIR ASTs for canonicalizer stress testing.

## 🛠 Prerequisites

To build and compile this dialect, you will need:

* **C++23 Compiler:** GCC 13+ or Clang 17+ (Required for `uz` literals and `std::floating_point` concepts).

* **CMake & Ninja:** For the build system.

* **Intel TBB:** Required for the scalable test generator (`libtbb-dev`).

* **LLVM / MLIR:** A pre-built LLVM/MLIR environment (e.g., from CIRCT or upstream LLVM).

## 🚀 How to Build and Test

This setup assumes you have built LLVM/MLIR (for example, inside a CIRCT repository). You can build the dialect and run all integration tests using the following commands:

```bash
# 1. Create and enter the build directory
mkdir -p build && cd build

# 2. Configure the project with CMake
# Make sure to point MLIR_DIR and LLVM_EXTERNAL_LIT to your actual LLVM build paths
cmake -G Ninja .. \
  -DMLIR_DIR=$HOME/GitHub/circt/llvm/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$HOME/GitHub/circt/llvm/build/bin/llvm-lit

# 3. Build the tool and run the Lit test suite
ninja check-dsp
```

### What does `ninja check-dsp` do?

1. Auto-generates exhaustive scalable tests via `GenerateDSPTests`.
2. Compiles the `dsp-opt` tool and the `DSPTestUtils` shared library.
3. Runs MLIR `FileCheck` tests for IR canonicalization.
4. Executes the `test_jit.mlir` file through `mlir-cpu-runner` to mathematically verify the DCT lowering pipeline on your actual CPU.

## 🧠 Architecture Overview

### 1. The Dialect (`DSP.td` & `DSPOps.td`)
Defines the `dsp` namespace and the operations. They are registered into the MLIR context using TableGen auto-generated C++ classes.

### 2. The Optimizer (`dsp-opt`)
A custom MLIR driver similar to `mlir-opt`. It registers our custom passes alongside necessary upstream dialects (`arith`, `func`, `linalg`, `tensor`).

### 3. Lowering Pass (`DSPToLinalg.cpp`)
Targets the `dsp.dct` operation. It inspects the input tensor's shape and data type (supporting `f32` and `f64` across 4x4, 8x8, 16x16, and 32x32 block sizes) at runtime, generates the exact DCT-II coefficient matrices using C++ templates, and replaces the high-level operation with pure `linalg.matmul` and `arith.constant` ops.

## 📄 License

This dialect is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.