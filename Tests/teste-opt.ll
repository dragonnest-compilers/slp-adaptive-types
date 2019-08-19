; ModuleID = 'teste-opt.bc'
source_filename = "teste.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4 0x55a53bb6d698
  %2 = alloca i32, align 4 0x55a53bb6d6f8
  %3 = alloca i32, align 4 0x55a53bb6d758
  %4 = alloca i32, align 4 0x55a53bb6d7b8
  %5 = alloca i64, align 8 0x55a53bb6d818
  %6 = alloca i64, align 8 0x55a53bb6d878
  %7 = alloca i64, align 8 0x55a53bb6dbe8
  store i32 0, i32* %1, align 4 0x55a53bb6dc60
  store i32 2, i32* %2, align 4 0x55a53bb6dce0
  store i32 3, i32* %3, align 4 0x55a53bb6dd60
  %8 = load i32, i32* %2, align 4 0x55a53bb6ddc8
  %9 = load i32, i32* %3, align 4 0x55a53bb6de28
  store i64 2, i64* %5, align 8 0x55a53bb6dea0
  store i64 3, i64* %6, align 8 0x55a53bb6df20
  %10 = load i64, i64* %5, align 8 0x55a53bb6df88
  %11 = load i64, i64* %6, align 8 0x55a53bb6dfe8
  %12 = sext i32 %8 to i64 0x55a53bb6e048
  %13 = insertelement <2 x i64> undef, i64 %12, i64 0 0x55a53bb6e0d8
  %14 = insertelement <2 x i64> %13, i64 %10, i64 1 0x55a53bb6e168
  %15 = sext i32 %9 to i64 0x55a53bb6e1c8
  %16 = insertelement <2 x i64> undef, i64 %15, i64 0 0x55a53bb6e258
  %17 = insertelement <2 x i64> %16, i64 %11, i64 1 0x55a53bb6e2e8
  %18 = add nsw <2 x i64> %14, %17 0x55a53bb67110
  %19 = extractelement <2 x i64> %18, i64 0 0x55a53bb671a0
  %20 = trunc i64 %19 to i32 0x55a53bb6e348
  %21 = extractelement <2 x i64> %18, i64 1 0x55a53bb67240
  store i64 %21, i64* %7, align 8 0x55a53bb6e3c0
  store i32 %20, i32* %4, align 4 0x55a53bb6e440
  ret i32 0 0x55a53bb6e4a8
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0 "}
