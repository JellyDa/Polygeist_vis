; ModuleID = 'testDump.cpp'
source_filename = "testDump.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

$_ZSt4sqrtf = comdat any

@.str = private unnamed_addr constant [5 x i8] c"test\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"x = %d\00", align 1

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define dso_local float @_Z4funcf(float %0) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = load float, float* %2, align 4
  %5 = fmul float %3, %4
  ret float %5
}

; Function Attrs: noinline norecurse optnone uwtable mustprogress
define dso_local i32 @main() #1 {
  %1 = alloca i32, align 4
  %2 = alloca float, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %5 = call float @_Z4funcf(float 4.000000e+00)
  store float %5, float* %2, align 4
  %6 = load float, float* %2, align 4
  %7 = call float @_ZSt4sqrtf(float %6)
  %8 = fptosi float %7 to i32
  store i32 %8, i32* %3, align 4
  %9 = load float, float* %2, align 4
  %10 = call float @_ZSt4sqrtf(float %9)
  %11 = load i32, i32* %3, align 4
  %12 = sitofp i32 %11 to float
  %13 = load float, float* %2, align 4
  %14 = fadd float %12, %13
  %15 = fptosi float %14 to i32
  store i32 %15, i32* %4, align 4
  %16 = load i32, i32* %4, align 4
  %17 = call i32 @_Z8dump_intPKci(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), i32 %16)
  store i32 %17, i32* %3, align 4
  %18 = load i32, i32* %3, align 4
  %19 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i64 0, i64 0), i32 %18)
  %20 = load i32, i32* %3, align 4
  ret i32 %20
}

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define linkonce_odr dso_local float @_ZSt4sqrtf(float %0) #0 comdat {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @sqrtf(float %3) #4
  ret float %4
}

declare dso_local i32 @_Z8dump_intPKci(i8*, i32) #2

declare dso_local i32 @printf(i8*, ...) #2

; Function Attrs: nounwind
declare dso_local float @sqrtf(float) #3

attributes #0 = { noinline nounwind optnone uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline norecurse optnone uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 12.0.1-++20211029101322+fed41342a82f-1~exp1~20211029221816.4"}
