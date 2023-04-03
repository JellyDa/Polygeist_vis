//===- TrivialUse.cpp - Remove trivial use instruction ---------------- -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

// #include "mlir/../../lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "visualgo/Ops.h"
#include <string>
#include <vector>
#define DEBUG_TYPE "convert-visualgo-to-llvm"

using namespace mlir;
using namespace visualgo;
using namespace std;

namespace {

class DumpIntOpLowering : public ConversionPattern {
   public:
    explicit DumpIntOpLowering(MLIRContext* context)
        : ConversionPattern(visualgo::DumpIntOp::getOperationName(),
                            1,
                            context) {}

    LogicalResult matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override {
        // auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
        // auto memRefShape = memRefType.getShape();
        auto loc = op->getLoc();

        ModuleOp parentModule = op->getParentOfType<ModuleOp>();

        // Get a symbol reference to the printf function, inserting it if
        // necessary.
        auto dumpIntRef = getOrInsertDumpInt(rewriter, parentModule);
        // Value formatSpecifierCst = getOrCreateGlobalString(
        //     loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
        // Value newLineCst = getOrCreateGlobalString(
        //     loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

        // Notify the rewriter that this operation has been removed.
        rewriter.replaceOpWithNewOp<func::CallOp>(
            op, dumpIntRef, rewriter.getIntegerType(32), operands);
        return success();
    }

   private:
    /// Return a symbol reference to the printf function, inserting it into the
    /// module if necessary.
    static FlatSymbolRefAttr getOrInsertDumpInt(PatternRewriter& rewriter,
                                                ModuleOp module) {
        // 在Clang中，_Z8前缀和Pkci后缀是由C++编译器生成的，用于标识函数的名称和参数类型。其中_Z8是指函数名的长度为8个字符，Pkci则是指函数参数类型为pointer
        // to constant integer（指向常量整数的指针）
        string func_name_tmp = 
        getMangledNameFromString("dump_int(const char *, int)->int");
        StringRef func_name(func_name_tmp.c_str());
            
        // StringRef funcName("dump_int");
        auto* context = module.getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(func_name))
            return SymbolRefAttr::get(context, func_name);

        // Create a function declaration for dump_int, the signature is:
        //   * `i32 (i8*, i32)`
        auto llvmI32Ty = IntegerType::get(context, 32);
        auto llvmI8PtrTy =
            LLVM::LLVMPointerType::get(IntegerType::get(context, 8));

        auto llvmFnType = LLVM::LLVMFunctionType::get(
            llvmI32Ty, {llvmI8PtrTy, llvmI32Ty}, /*isVarArg=*/false);

        // Insert the dump_int function into the body of the parent module.
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), func_name,
                                          llvmFnType);
        return SymbolRefAttr::get(context, func_name);
    }

    /// Return a value representing an access into a global string with the
    /// given name, creating the string if necessary.
    static Value getOrCreateGlobalString(Location loc,
                                         OpBuilder& builder,
                                         StringRef name,
                                         StringRef value,
                                         ModuleOp module) {
        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
            OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto type = LLVM::LLVMArrayType::get(
                IntegerType::get(builder.getContext(), 8), value.size());
            global = builder.create<LLVM::GlobalOp>(
                loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
                builder.getStringAttr(value),
                /*alignment=*/0);
        }

        // Get the pointer to the first character in the global string.
        Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = builder.create<LLVM::ConstantOp>(
            loc, IntegerType::get(builder.getContext(), 64),
            builder.getIntegerAttr(builder.getIndexType(), 0));
        return builder.create<LLVM::GEPOp>(
            loc,
            LLVM::LLVMPointerType::get(
                IntegerType::get(builder.getContext(), 8)),
            globalPtr, ArrayRef<Value>({cst0, cst0}));
    }

    // ref:
    // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling-structure
    
    static string getMangledNameFromString(string func_def) {
        // get builtin type
        auto getBuiltinType = [](string type) {
            string ch;
            if (type == "int") {
                ch = "i";
            } else if (type == "float") {
                ch = "f";
            } else if (type == "double") {
                ch = "d";
            } else if (type == "char") {
                ch = "c";
            } else if (type == "int *") {
                ch = "Pi";
            } else if (type == "char *") {
                ch = "Pc";
            } else if (type == "float *") {
                ch = "Pf";
            } else if (type == "const char *") {
                ch = "PKc";
            } else {
                ch = "unexpectedName";
            }

            return ch;
        };

        // split func_name by "->"
        size_t pos = func_def.find("->");
        string left = func_def.substr(0, pos);
        string right = func_def.substr(pos + 2);
        // split left by "("
        pos = left.find("(");
        string func = left.substr(0, pos);
        string params = left.substr(pos + 1);
        // remove the trailing ")"
        params.pop_back();
        // split params by ","
        stringstream ss(params);
        string param;
        std::vector<string> param_list;
        while (getline(ss, param, ',')) {
            // trim the whitespace
            param.erase(0, param.find_first_not_of(" "));
            param.erase(param.find_last_not_of(" ") + 1);
            // push the param to the vector
            param_list.push_back(param);
        }
        // initialize an empty string for MangleName
        string mangled_name = "";
        // add the prefix "_Z" and the length of func
        mangled_name += "_Z" + to_string(func.length());
        // add the func
        mangled_name += func;
        // for each param in param_list
        for (string param : param_list) {
            // convert the param to the corresponding character
            string ch = getBuiltinType(param);
            // add the char to MangleName
            mangled_name += ch;
        }

        // generate return type
        // // add the suffix "->"
        // MangleName += "->";
        // // convert the right to the corresponding character
        // string ch = getBuiltinType(right);
        // // add the char to MangleName
        // MangleName += ch;
        // // return MangleName
        return mangled_name;
    }
};
}  // namespace

//
struct ConvertVisualgoToLLVMPass
    : public convertVisualgoToLLVMBase<ConvertVisualgoToLLVMPass> {
    ConvertVisualgoToLLVMPass() = default;
    void runOnOperation() override {
        LLVMConversionTarget target(getContext());
        target.addLegalOp<ModuleOp>();
        target.addIllegalOp<visualgo::DumpIntOp>();

        LLVMTypeConverter typeConverter(&getContext());
        RewritePatternSet patterns(&getContext());

        patterns.add<DumpIntOpLowering>(&getContext());
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);

        auto module = getOperation();
        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

std::unique_ptr<Pass> mlir::visualgo::createConvertVisualgoToLLVMPass() {
    // TODO: meaningful arguments to this pass should be specified as
    // Option<...>'s to the pass in Passes.td. For now, we'll provide some dummy
    // default values to allow for pass creation.

    return std::make_unique<ConvertVisualgoToLLVMPass>();
}
