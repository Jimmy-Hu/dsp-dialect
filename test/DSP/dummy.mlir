// RUN: dsp-opt %s | dsp-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = dsp.foo %{{.*}} : i32
        %res = dsp.foo %0 : i32
        return
    }
}
