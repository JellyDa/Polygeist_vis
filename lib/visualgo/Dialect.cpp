//===- PolygeistDialect.cpp - Polygeist dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "visualgo/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "visualgo/Ops.h"

using namespace mlir;
using namespace mlir::visualgo;

//===----------------------------------------------------------------------===//
// Visualgo dialect.
//===----------------------------------------------------------------------===//

void VisualgoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "visualgo/VisualgoOps.cpp.inc"
      >();
}

#include "visualgo/VisualgoOpsDialect.cpp.inc"

void DumpIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value tarcevarname, Value var) {
  odsState.addTypes(IntegerType::get(odsBuilder.getContext(), 32));
  odsState.addOperands({tarcevarname, var});
}
