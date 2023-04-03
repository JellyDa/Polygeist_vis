//===- VISDialect.td - VIS dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VISUALGO_DIALECT
#define VISUALGO_DIALECT

include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// VIS dialect definition.
//===----------------------------------------------------------------------===//
def Visualgo_Dialect : Dialect {
    let name = "visualgo";
    let description = [{}];
    let cppNamespace = "::mlir::visualgo";
}

//===----------------------------------------------------------------------===//
// Base BFV operation definition.
//===----------------------------------------------------------------------===//

class Visualgo_Op<string mnemonic, list<Trait> traits = []>
: Op<Visualgo_Dialect, mnemonic, traits>;

#endif // VISUALGO_DIALECT
