package io.github.lruijsink.ktorch.cpu.f32

import io.github.lruijsink.ktorch.Tensor
import java.nio.FloatBuffer

class F32NonContiguousIterator(
    tensor: Tensor,
) : F32Iterator {
    private val buffer: FloatBuffer =
        (tensor.storage as F32RAM)
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
