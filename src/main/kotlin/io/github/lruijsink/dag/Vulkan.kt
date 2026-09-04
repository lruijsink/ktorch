package io.github.lruijsink.dag

val vulkan = Device("vulkan").apply {
    impl(expOp) { (x) -> TODO() }
    impl(plusOp) { (x, y) -> TODO() }
    impl(concatOp) { (axis, tensors) -> TODO() }
    impl(plusAssignOp) { (x, y) -> TODO() }
}
