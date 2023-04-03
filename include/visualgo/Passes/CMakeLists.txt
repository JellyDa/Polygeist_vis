set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name visualgo)
add_public_tablegen_target(MLIRVisualgoPassIncGen)

# add_mlir_doc(Passes VisualgoPasses ./ -gen-pass-doc)
