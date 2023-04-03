
add_mlir_dialect_library(MLIRVisualgo
Dialect.cpp
Ops.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/visualgo

DEPENDS
MLIRVisualgoOpsIncGen

LINK_LIBS PUBLIC
MLIRIR
MLIRMemRefDialect
MLIRLLVMDialect
MLIROpenMPDialect
MLIRAffineDialect
MLIRSupport
MLIRSCFTransforms
)
add_subdirectory(Passes)
