rm -rf build
mkdir -p build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$HOME/GitHub/circt/llvm/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$HOME/GitHub/circt/llvm/build/bin/llvm-lit

ninja check-dsp