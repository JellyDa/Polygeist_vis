; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@str2 = internal constant [7 x i8] c"x = %d\00"
@str1 = internal constant [13 x i8] c"testtesttest\00"
@str0 = internal constant [5 x i8] c"test\00"

; Function Attrs: inaccessiblememonly mustprogress nofree nounwind willreturn
declare noalias noundef i8* @malloc(i64) #0

declare void @free(i8*) #1

declare i32 @_Z8dump_intPKci(i8*, i32) #1

declare i32 @printf(i8*, ...) #1

define i32 @main() #1 !dbg !3 {
  %1 = call i32 @_Z8dump_intPKci(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @str0, i32 0, i32 0), i32 20), !dbg !7
  %2 = call i32 @_Z8dump_intPKci(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @str1, i32 0, i32 0), i32 %1), !dbg !7
  %3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @str2, i32 0, i32 0), i32 4), !dbg !9
  ret i32 0, !dbg !10
}

define float @_Z4funcf(float %0) #1 !dbg !11 {
  %2 = fmul float %0, %0, !dbg !12
  ret float %2, !dbg !14
}

attributes #0 = { inaccessiblememonly mustprogress nofree nounwind willreturn "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 9, type: !5, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "testDump.cpp", directory: "/home/xiaopeng/llvm/visualgo/Polygeist_vis/test/visualgo")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 15, column: 5, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 16, column: 5, scope: !8)
!10 = !DILocation(line: 19, column: 1, scope: !8)
!11 = distinct !DISubprogram(name: "_Z4funcf", linkageName: "_Z4funcf", scope: null, file: !4, line: 5, type: !5, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!12 = !DILocation(line: 6, column: 14, scope: !13)
!13 = !DILexicalBlockFile(scope: !11, file: !4, discriminator: 0)
!14 = !DILocation(line: 7, column: 1, scope: !13)

