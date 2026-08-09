package io.github.lruijsink.ktorch.cpu.f32

import io.github.lruijsink.ktorch.Storage
import java.nio.ByteBuffer

/**
 * CPU buffer type, e.g. RAM/host memory
 */
class F32RAM(
    override val capacity: Long,
) : Storage {
    init {
        require(capacity > 0) { "size must be positive" }
        require(capacity <= Int.MAX_VALUE) { "cannot allocate more than Int.MAX_VALUE bytes" }
    }

    val byteBuffer: ByteBuffer = ByteBuffer.allocate(capacity.toInt() * Float.SIZE_BYTES)

    override var version: Int = 0
        private set

    override fun copyInto(receiver: ByteBuffer, sourceOffset: Long, length: Long) {
        require(sourceOffset >= 0)
        require(length >= 0)
        require(sourceOffset + length <= capacity)
        require(length * Float.SIZE_BYTES <= receiver.remaining().toLong())

        val byteOffset = sourceOffset * Float.SIZE_BYTES
        val byteLength = length * Float.SIZE_BYTES

        val source = byteBuffer.duplicate()
        source.position(byteOffset.toInt())
        source.limit((byteOffset + byteLength).toInt())

        receiver.put(source)
    }

    override fun write(operation: (ByteBuffer) -> Unit) {
        byteBuffer.rewind()
        operation(byteBuffer)
        version++
    }
}
