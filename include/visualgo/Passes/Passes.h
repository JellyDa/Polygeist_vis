#ifndef VISUALGO_DIALECT_VISUALGO_PASSES_H
#define VISUALGO_DIALECT_VISUALGO_PASSES_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir {
class PatternRewriter;
class RewritePatternSet;
class DominanceInfo;
namespace visualgo {

std::unique_ptr<Pass> createConvertVisualgoToLLVMPass();

} // namespace polygeist
} // namespace mlir

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // end namespace arith

namespace scf {
class SCFDialect;
} // end namespace scf

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace func {
class FuncDialect;
}

class AffineDialect;
namespace LLVM {
class LLVMDialect;
}

#define GEN_PASS_REGISTRATION
#include "visualgo/Passes/Passes.h.inc"

} // end namespace mlir

#endif // VISUALGO_DIALECT_VISUALGO_PASSES_H
