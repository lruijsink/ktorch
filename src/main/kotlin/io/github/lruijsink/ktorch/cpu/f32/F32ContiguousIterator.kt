package io.github.lruijsink.ktorch.cpu.f32

import io.github.lruijsink.ktorch.Tensor
import java.nio.FloatBuffer

class F32ContiguousIterator(
    tensor: Tensor,
) : F32Iterator {
    private val buffer: FloatBuffer =
        (tensor.storage as F32RAM)
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
