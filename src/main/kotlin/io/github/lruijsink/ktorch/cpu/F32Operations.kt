package io.github.lruijsink.ktorch.cpu

import io.github.lruijsink.ktorch.PrimitiveType
import io.github.lruijsink.ktorch.Tensor
import io.github.lruijsink.ktorch.contiguousStride
import java.nio.FloatBuffer

/**
 * CPU operations for [PrimitiveType.F32]
 */
object F32Operations : CPUOperations {
    override fun cloneContiguous(t: Tensor): Tensor =
        unary(t) { it }

    override fun neg(t: Tensor): Tensor =
        unary(t) { -it }

    override fun add(a: Tensor, b: Tensor): Tensor =
        binary(a, b) { x, y -> x + y }

    override fun sub(a: Tensor, b: Tensor): Tensor =
        binary(a, b) { x, y -> x - y }

    override fun mul(a: Tensor, b: Tensor): Tensor =
        binary(a, b) { x, y -> x * y }

    override fun div(a: Tensor, b: Tensor): Tensor =
        binary(a, b) { x, y -> x / y }

    private fun unary(t: Tensor, op: (Float) -> Float): Tensor {
        val iter = t.iterator()
        val s = CPU.allocate(t.numElements * Float.SIZE_BYTES.toLong())
        s.write {
            val buff = it.asFloatBuffer()
            while (iter.hasRemaining()) buff.put(op(iter.next()))
        }
        return Tensor(
            shape = t.shape,
            stride = contiguousStride(t.shape),
            primitiveType = PrimitiveType.F32,
            device = CPU,
            storage = s,
            storageOffset = 0,
        )
    }

    private fun binary(a: Tensor, b: Tensor, op: (Float, Float) -> Float): Tensor {
        val aIter = a.iterator()
        val bIter = b.iterator()
        val s = CPU.allocate(a.numElements * Float.SIZE_BYTES.toLong())
        s.write {
            val sBuff = it.asFloatBuffer()
            while (aIter.hasRemaining()) sBuff.put(op(aIter.next(), bIter.next()))
        }
        return Tensor(
            shape = a.shape,
            stride = contiguousStride(a.shape),
            primitiveType = PrimitiveType.F32,
            device = CPU,
            storage = s,
            storageOffset = 0,
        )
    }

    private fun Tensor.iterator(): F32Iterator =
        if (isContiguous) F32ContiguousIterator(this) else F32NonContiguousIterator(this)
}

private interface F32Iterator {
    fun next(): Float

    fun hasRemaining(): Boolean

    fun reset()
}

private class F32ContiguousIterator(
    tensor: Tensor,
) : F32Iterator {
    private val buffer: FloatBuffer =
        (tensor.storage as RAM)
            .byteBuffer
            .rewind()
            .position(tensor.storageOffset.toInt() * Float.SIZE_BYTES)
            .asFloatBuffer()
            .limit(tensor.numElements.toInt())

    override fun next(): Float =
        buffer.get()

    override fun hasRemaining(): Boolean =
        buffer.hasRemaining()

    override fun reset() {
        buffer.rewind()
    }
}

private class F32NonContiguousIterator(
    tensor: Tensor,
) : F32Iterator {
    private val buffer: FloatBuffer =
        (tensor.storage as RAM)
            .byteBuffer
            .rewind()
            .position(tensor.storageOffset.toInt() * Float.SIZE_BYTES)
            .asFloatBuffer()

    private val shape: IntArray = tensor.shape
    private val stride: IntArray = tensor.stride
    private val numElements: Int = tensor.numElements.toInt()

    private val counter = IntArray(shape.size)
    private var position = 0
    private var consumed = 0

    init {
        reset()
    }

    override fun next(): Float {
        val value = buffer.get(position)
        consumed++

        // odometer increment, from the last dim inward,
        // adjusting currentOffset incrementally via strides
        var d = counter.size - 1
        while (d >= 0) {
            counter[d]++
            position += stride[d]
            if (counter[d] < shape[d]) break
            // carry: undo this dim's accumulated offset, reset, move to next dim left
            position -= stride[d] * shape[d]
            counter[d] = 0
            d--
        }

        return value
    }

    override fun hasRemaining(): Boolean =
        consumed < numElements

    override fun reset() {
        counter.fill(0)
        position = 0
    }
}
