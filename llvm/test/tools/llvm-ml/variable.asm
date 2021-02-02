# RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data
t1_value equ 1 or 2

t1 BYTE t1_value DUP (0)
; CHECK: t1:
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .byte 0
; CHECK-NOT: .byte 0

END
