#ifndef VISUALGO_PASSES
#define VISUALGO_PASSES

include "mlir/Pass/PassBase.td"

def convertVisualgoToLLVM : Pass<"convert-visualgo-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert visualgo to the LLVM dialect";
  let constructor = "mlir::visualgo::createConvertVisualgoToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}


#endif // VISUALGO_PASSES
