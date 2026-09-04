package io.github.lruijsink.dag

class Tensor(
    val shape: IntArray,
    val strides: IntArray,
    val storage: Storage,
    val device: Device,
) {
    val numElements: Long = shape.fold(1L) { a, b -> a * b }
    val dimensions: Int get() = shape.size
    val isContiguous: Boolean
    val isScalar get(): Boolean = device == cpu && shape.isEmpty()

    init {
        require(shape.size == strides.size) { "Shape rank ${shape.size} must equal number of strides ${strides.size}" }

        isContiguous = let {
            var expectedStride = 1
            for (i in shape.size - 1 downTo 0) {
                if (shape[i] == 1) continue
                if (strides[i] != expectedStride) return@let false
                expectedStride *= shape[i]
            }
            return@let true
        }
    }

    operator fun unaryPlus(): Tensor =
        unaryPlus(this)

    operator fun unaryMinus(): Tensor =
        unaryMinus(this)

    operator fun plus(t: Tensor): Tensor =
        plus(this, t)

    operator fun plus(n: Number): Tensor =
        plus(this, scalar(n))

    operator fun minus(t: Tensor): Tensor =
        minus(this, t)

    operator fun minus(n: Number): Tensor =
        minus(this, scalar(n))

    operator fun times(t: Tensor): Tensor =
        times(this, t)

    operator fun times(n: Number): Tensor =
        times(this, scalar(n))

    operator fun div(t: Tensor): Tensor =
        div(this, t)

    operator fun div(n: Number): Tensor =
        div(this, scalar(n))

    operator fun plusAssign(t: Tensor) {
        plusAssign(this, t)
    }

    operator fun plusAssign(n: Number) {
        plusAssign(this, scalar(n))
    }

    operator fun minusAssign(t: Tensor) {
        minusAssign(this, t)
    }

    operator fun minusAssign(n: Number) {
        minusAssign(this, scalar(n))
    }

    operator fun timesAssign(t: Tensor) {
        timesAssign(this, t)
    }

    operator fun timesAssign(n: Number) {
        timesAssign(this, scalar(n))
    }

    operator fun divAssign(t: Tensor) {
        divAssign(this, t)
    }

    operator fun divAssign(n: Number) {
        divAssign(this, scalar(n))
    }

    override fun toString(): String =
        "Tensor(" +
            "shape=${shape.contentToString()}, " +
            "strides=${strides.contentToString()}, " +
            "storage=$storage, " +
            "device=$device" +
            ")"
}

operator fun Number.plus(t: Tensor): Tensor =
    scalar(this) + t

operator fun Number.minus(t: Tensor): Tensor =
    scalar(this) - t

operator fun Number.times(t: Tensor): Tensor =
    scalar(this) * t

operator fun Number.div(t: Tensor): Tensor =
    scalar(this) / t

fun isContiguous(shape: IntArray, strides: IntArray): Boolean {
    var expectedStride = 1
    for (i in shape.size - 1 downTo 0) {
        if (shape[i] == 1) continue
        if (strides[i] != expectedStride) return false
        expectedStride *= shape[i]
    }
    return true
}
