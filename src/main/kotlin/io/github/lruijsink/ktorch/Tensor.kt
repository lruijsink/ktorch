package io.github.lruijsink.ktorch

import java.nio.ByteBuffer
import kotlin.math.max

/**
 * N-dimensional numeric object such as a vector or matrix.
 */
class Tensor(
    val shape: IntArray,
    val stride: IntArray,
    val backend: Backend,
    val storage: Storage,
    val storageOffset: Long,
) {
    val dimensions: Int = shape.size
    val numElements: Long = shape.fold(1L) { a, b -> a * b }
    val isContiguous: Boolean

    init {
        require(shape.size == stride.size) { "shape and stride must have same number of dimensions" }
        require(shape.all { it >= 0 }) { "shape slices cannot be negative" }
        require(stride.all { it >= 0 }) { "strides cannot be negative" }

        isContiguous = let {
            var expectedStride = 1
            for (i in dimensions - 1 downTo 0) {
                if (shape[i] == 1) continue
                if (stride[i] != expectedStride) return@let false
                expectedStride *= shape[i]
            }
            return@let true
        }
    }

    fun contiguous(): Tensor =
        if (isContiguous) this else backend.clone(this)

    fun clone(): Tensor =
        backend.clone(this)

    operator fun unaryPlus(): Tensor =
        clone()

    operator fun unaryMinus(): Tensor =
        backend.neg(this)

    operator fun plus(other: Tensor): Tensor =
        binaryOperator(other, backend::add)

    operator fun minus(other: Tensor): Tensor =
        binaryOperator(other, backend::sub)

    operator fun times(other: Tensor): Tensor =
        binaryOperator(other, backend::mul)

    operator fun div(other: Tensor): Tensor =
        binaryOperator(other, backend::div)

    private fun binaryOperator(other: Tensor, op: (Tensor, Tensor) -> Tensor): Tensor {
        require(backend == other.backend) { "tensors must use the same backend" }
        val (a, b) = this.broadcastAgainst(other)
        return op(a, b)
    }

    private fun broadcastAgainst(other: Tensor): Pair<Tensor, Tensor> {
        if (this.shape.contentEquals(other.shape)) return Pair(this, other)
        val targetShape = broadcastShape(this.shape, other.shape)
        return Pair(this.expand(targetShape), other.expand(targetShape))
    }

    operator fun times(number: Number): Tensor =
        times(matchToScalar(number))

    operator fun div(number: Number): Tensor =
        div(matchToScalar(number))

    private fun matchToScalar(value: Number): Tensor =
        when (backend.primitiveType) {
            PrimitiveType.F32 -> tensor(value.toFloat(), backend)
            PrimitiveType.F64 -> TODO("F64 scalar matching not supported yet")
        }

    fun expand(targetShape: IntArray): Tensor {
        require(this.shape.size <= targetShape.size) { "expand cannot reduce the number of dimensions" }
        if (this.shape.contentEquals(targetShape)) return this

        val newDims = targetShape.size - shape.size
        val newStride = IntArray(targetShape.size)
        for (i in targetShape.indices) {
            val iSrc = i - newDims
            if (iSrc < 0 || (shape[iSrc] == 1 && targetShape[i] != 1)) {
                newStride[i] = 0
            } else {
                require(shape[iSrc] == targetShape[i]) { "shape mismatch, not broadcastable" }
                newStride[i] = stride[iSrc]
            }
        }
        return Tensor(targetShape, newStride, backend, storage, storageOffset)
    }

    override fun toString(): String {
        val buff = ByteBuffer.allocate(storage.capacity.toInt() * backend.primitiveType.sizeBytes)
        storage.copyInto(buff)
        buff.rewind()

        val nums: List<Number> = when (backend.primitiveType) {
            PrimitiveType.F32 -> {
                val elements = FloatArray(numElements.toInt())
                buff.asFloatBuffer().get(elements)
                elements.toList()
            }

            PrimitiveType.F64 -> {
                val elements = DoubleArray(numElements.toInt())
                buff.asDoubleBuffer().get(elements)
                elements.toList()
            }
        }

        return when {
            // Empty tensor:
            numElements == 0L ->
                "∅"

            // Scalar:
            dimensions == 0 ->
                "${nums[0]}"

            // Vector:
            dimensions == 1 ->
                "[ ${nums.joinToString("  ")} ]"

            // Matrix:
            dimensions == 2 -> {
                val numStrings = nums.map { it.toString() }
                val maxLen = numStrings.maxOf { it.length }
                val (rows, cols) = shape

                numStrings
                    .chunked(cols)
                    .mapIndexed { i, row ->
                        val (r, l) = when (i) {
                            0 -> '⎡' to '⎤'
                            rows - 1 -> '⎣' to '⎦'
                            else -> '⎢' to '⎥'
                        }
                        "$r ${row.joinToString("  ") { it.padStart(maxLen, ' ') }} $l"
                    }.joinToString("\n")
            }

            // Otherwise:
            else ->
                "{shape: ${shape.toList()}, stride: ${stride.toList()}, elements: $nums}"
        }
    }
}

/**
 * Multiply by scalar, equivalent to `tensor * scalar`
 */
operator fun Number.times(tensor: Tensor): Tensor =
    tensor.times(this)

/**
 * Elementwise `exp`
 */
fun exp(tensor: Tensor): Tensor =
    tensor.backend.exp(tensor)

/**
 * Elementwise `ln`
 */
fun ln(tensor: Tensor): Tensor =
    tensor.backend.ln(tensor)

/**
 * Elementwise `sqrt`
 */
fun sqrt(tensor: Tensor): Tensor =
    tensor.backend.sqrt(tensor)

/**
 * Compute what the resulting shape would be after broadcasting [a] and [b] against each other
 */
fun broadcastShape(a: IntArray, b: IntArray): IntArray {
    val ndim = maxOf(a.size, b.size)
    return IntArray(ndim) { i ->
        val ai = if (i < ndim - a.size) 1 else a[i - (ndim - a.size)]
        val bi = if (i < ndim - b.size) 1 else b[i - (ndim - b.size)]
        if (ai == 1 || bi == 1 || ai == bi) {
            max(ai, bi)
        } else {
            throw IllegalArgumentException("shapes ${a.toList()} and ${b.toList()} not broadcastable on dim $i")
        }
    }
}

/**
 * Compute the stride a contiguous tensor with shape [shape] would have
 */
fun contiguousStride(shape: IntArray): IntArray {
    if (shape.isEmpty()) {
        return intArrayOf()
    }

    val stride = IntArray(shape.size)

    stride[shape.size - 1] = 1
    for (i in shape.size - 2 downTo 0) {
        stride[i] = shape[i + 1] * stride[i + 1]
    }

    return stride
}
