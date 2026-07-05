# -*- Python -*-

import os
import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'DSP'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Add system environment variables that might be needed by the compiler
llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])
llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite.
config.excludes = ['CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.dsp_obj_root, 'test')
config.dsp_tools_dir = os.path.join(config.dsp_obj_root, 'bin')

# Add the DSP tools directory to the path so that 'dsp-opt' can be found during tests.
llvm_config.with_environment('PATH', config.dsp_tools_dir, append_path=True)

config.substitutions.append(('%dsp_obj_root', config.dsp_obj_root))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))