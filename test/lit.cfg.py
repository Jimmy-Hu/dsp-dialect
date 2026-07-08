# -*- Python -*-

import os
import lit.formats
import lit.llvm
from lit.llvm import llvm_config

# Initialize LLVM-specific lit configurations
lit.llvm.initialize(lit_config, config)

# Basic Configuration
config.name = 'DSP Dialect'

# LLVM 23+ deprecates external shell execution for tests. 
# We must explicitly use the internal shell by setting execute_external=False.
config.test_format = lit.formats.ShTest(execute_external=False)

config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.dsp_obj_root, 'test')

# Add tool paths to the environment so tools can be found
llvm_config.with_environment('PATH', config.dsp_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# Register custom substitutions for shared libraries
config.substitutions.append(('%dsp_obj_root', config.dsp_obj_root))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

# Safely resolve mlir-cpu-runner using the dynamic LLVM tools directory from CMake
mlir_cpu_runner_path = os.path.join(config.llvm_tools_dir, 'mlir-cpu-runner')
config.substitutions.append(('%mlir_cpu_runner', mlir_cpu_runner_path))