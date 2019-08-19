; ModuleID = 'teste.c'
source_filename = "teste.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4 0x55fff3dc36a8
  %2 = alloca i32, align 4 0x55fff3dc2e28
  %3 = alloca i32, align 4 0x55fff3dc2ad8
  %4 = alloca i32, align 4 0x55fff3dc2bb8
  %5 = alloca i64, align 8 0x55fff3dc48c8
  %6 = alloca i64, align 8 0x55fff3dc49d8
  %7 = alloca i64, align 8 0x55fff3dc4ae8
  store i32 0, i32* %1, align 4 0x55fff3dc2dc0
  store i32 2, i32* %2, align 4 0x55fff3dc2a70
  store i32 3, i32* %3, align 4 0x55fff3dc2b50
  %8 = load i32, i32* %2, align 4 0x55fff3dc2c18
  %9 = load i32, i32* %3, align 4 0x55fff3dc47e8
  %10 = add nsw i32 %8, %9 0x55fff3dc2a00
  store i32 %10, i32* %4, align 4 0x55fff3dc4860
  store i64 2, i64* %5, align 8 0x55fff3dc4970
  store i64 3, i64* %6, align 8 0x55fff3dc4a80
  %11 = load i64, i64* %5, align 8 0x55fff3dc4b48
  %12 = load i64, i64* %6, align 8 0x55fff3dc4ba8
  %13 = add nsw i64 %11, %12 0x55fff3dc2990
  store i64 %13, i64* %7, align 8 0x55fff3dc4c20
  ret i32 0 0x55fff3dc4d08
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0 "}
