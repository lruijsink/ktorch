package io.github.lruijsink.dag

import kotlin.math.max

class Query1(
    val t1: Tensor,
) {
    val device: Device = t1.device
    val shape: IntArray = t1.shape
    val strides: Array<IntArray> = arrayOf(t1.strides)
    val numElements: Long = numElementsOfShape(shape)

    init {
        require(device == cpu) { "QueryN is only supported for CPU device" }
        require(isContiguous(shape, strides[0])) { "Non-contiguous broadcast tensors are not supported yet" }
    }

    fun map(op: (Float) -> Float): Tensor {
        val storage = RAM(sizeBytes = numElements * Float.SIZE_BYTES)
        val f1 = (t1.storage as RAM).floats(numElements = numElements)
        val fOut = storage.floats(numElements = numElements)
        while (f1.hasRemaining()) fOut.put(op(f1.get()))
        return Tensor(shape = shape, strides = contiguousStride(shape), storage = storage, device = device)
    }

    fun mapInPlace(op: (Float) -> Float): Tensor {
        val f1 = (t1.storage as RAM).floats(numElements = numElements)
        val fOut = t1.storage.floats(numElements = numElements)
        while (f1.hasRemaining()) fOut.put(op(f1.get()))
        return t1
    }

    override fun toString(): String =
        "Query1(" +
            "t1=$t1, " +
            "shape=${shape.contentToString()}, " +
            "strides=${strides.map { it.contentToString() }}, " +
            "numElements=$numElements" +
            ")"
}

class Query2(
    val t1: Tensor,
    val t2: Tensor,
) {
    val device: Device = t1.device
    val shape: IntArray = broadcastShape(t1.shape, t2.shape)
    val strides: Array<IntArray> = arrayOf(t1.expandedStrides(shape), t2.expandedStrides(shape))
    val numElements: Long = numElementsOfShape(shape)

    init {
        require(device == cpu) { "QueryN is only supported for CPU device" }
        require(t1.device == t2.device) { "Tensors must be on the same device, but got [${t1.device}, ${t2.device}]" }
        require(isContiguous(shape, strides[0])) { "Non-contiguous broadcast tensors are not supported yet" }
        require(isContiguous(shape, strides[1])) { "Non-contiguous broadcast tensors are not supported yet" }
    }

    fun map(op: (Float, Float) -> Float): Tensor {
        val storage = RAM(sizeBytes = numElements * Float.SIZE_BYTES)
        val f1 = (t1.storage as RAM).floats(numElements = numElements)
        val f2 = (t2.storage as RAM).floats(numElements = numElements)
        val fOut = storage.floats(numElements = numElements)
        while (f1.hasRemaining()) fOut.put(op(f1.get(), f2.get()))
        return Tensor(shape = shape, strides = contiguousStride(shape), storage = storage, device = device)
    }

    fun mapInPlace(op: (Float, Float) -> Float): Tensor {
        val f1 = (t1.storage as RAM).floats(numElements = numElements)
        val f2 = (t2.storage as RAM).floats(numElements = numElements)
        val fOut = t1.storage.floats(numElements = numElements)
        while (f1.hasRemaining()) fOut.put(op(f1.get(), f2.get()))
        return t1
    }

    override fun toString(): String =
        "Query2(" +
            "t1=$t1, " +
            "t2=$t2, " +
            "shape=${shape.contentToString()}, " +
            "strides=${strides.map { it.contentToString() }}, " +
            "numElements=$numElements" +
            ")"
}

fun query(t1: Tensor, op: (Float) -> Float): Tensor =
    Query1(t1).map(op)

fun query(t1: Tensor, t2: Tensor, op: (Float, Float) -> Float): Tensor =
    when {
        t1.isScalar -> {
            val c = t1.scalarValue()
            query(t2) { x -> op(c, x) }
        }

        t2.isScalar -> {
            val c = t2.scalarValue()
            query(t1) { x -> op(x, c) }
        }

        else -> Query2(t1, t2).map(op)
    }

fun queryInPlace(t1: Tensor, op: (Float) -> Float): Tensor =
    Query1(t1).mapInPlace(op)

fun queryInPlace(t1: Tensor, t2: Tensor, op: (Float, Float) -> Float): Tensor =
    when {
        t2.isScalar -> {
            val c = t2.scalarValue()
            queryInPlace(t1) { x -> op(x, c) }
        }

        else -> Query2(t1, t2).mapInPlace(op)
    }

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

fun numElementsOfShape(shape: IntArray): Long =
    shape.fold(1L) { a, b -> a * b }

private fun Tensor.expandedStrides(targetShape: IntArray): IntArray {
    require(this.shape.size <= targetShape.size) { "expand cannot reduce the number of dimensions" }
    if (this.shape.contentEquals(targetShape)) return strides

    val newDims = targetShape.size - shape.size
    val newStrides = IntArray(targetShape.size)
    for (i in targetShape.indices) {
        val iSrc = i - newDims
        if (iSrc < 0 || (shape[iSrc] == 1 && targetShape[i] != 1)) {
            newStrides[i] = 0
        } else {
            require(shape[iSrc] == targetShape[i]) { "shape mismatch, not broadcastable" }
            newStrides[i] = strides[iSrc]
        }
    }
    return newStrides
}

private fun Tensor.scalarValue(): Float {
    require(this.isScalar) { "Cannot get the value of a non-scalar" }
    return (storage as RAM).floats().get()
}
