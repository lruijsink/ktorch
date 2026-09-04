package io.github.lruijsink.dag

import kotlin.math.exp

val cpu = Device("cpu").apply {
    impl(scalarFactory) { x ->
        val storage = RAM(Float.SIZE_BYTES.toLong())
        storage.floats(0).put(x)
        Tensor(shape = intArrayOf(), strides = intArrayOf(), storage = storage, device = this)
    }

    impl(vectorFactory) { vs ->
        val storage = RAM(vs.size * Float.SIZE_BYTES.toLong())
        storage.floats().put(vs)
        Tensor(shape = intArrayOf(vs.size), strides = intArrayOf(1), storage = storage, device = this)
    }

    impl(unaryPlusOp) { (x) -> query(x) { n -> n } }
    impl(unaryMinusOp) { (x) -> query(x) { n -> -n } }

    impl(plusOp) { (x, y) -> query(x, y) { a, b -> a + b } }
    impl(minusOp) { (x, y) -> query(x, y) { a, b -> a - b } }
    impl(timesOp) { (x, y) -> query(x, y) { a, b -> a * b } }
    impl(divOp) { (x, y) -> query(x, y) { a, b -> a / b } }

    impl(plusAssignOp) { (x, y) -> queryInPlace(x, y) { a, b -> a + b } }
    impl(minusAssignOp) { (x, y) -> queryInPlace(x, y) { a, b -> a - b } }
    impl(timesAssignOp) { (x, y) -> queryInPlace(x, y) { a, b -> a * b } }
    impl(divAssignOp) { (x, y) -> queryInPlace(x, y) { a, b -> a / b } }

    impl(expOp) { (x) -> query(x, ::exp) }
}
