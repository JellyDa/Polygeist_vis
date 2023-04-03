//===- Visualgo.td - Visualgo dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VISUALGO_OPS
#define VISUALGO_OPS

include "Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td"

// visualgo OP
def DumpIntOp : Visualgo_Op<"dumpInt"> {
    let summary = "Dump integer of tracing";
    let arguments = (ins AnyType : $tarcevarname, I32 : $var);
    let results = (outs I32 : $traceRet);
    let builders = [
        OpBuilder<(ins "Value" : $tarcevarname, "Value" : $var)>
    ];
}
#endif // VISUALGO_OPS
