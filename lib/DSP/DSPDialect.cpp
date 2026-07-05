//===- DSPDialect.cpp - DSP dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DSP/DSPDialect.h"
#include "DSP/DSPOps.h"

using namespace mlir;
using namespace mlir::dsp;

//===----------------------------------------------------------------------===//
// DSP dialect.
//===----------------------------------------------------------------------===//

void DSPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DSP/DSP.cpp.inc"
      >();
}
