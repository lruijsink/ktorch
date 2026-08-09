package io.github.lruijsink.ktorch

import java.nio.ByteBuffer

/**
 * No-op storage class for use by empty tensors only
 */
object NoopStorage : Storage {
    override val capacity: Long = 0
    override val version: Int = 0

    override fun copyInto(receiver: ByteBuffer, sourceOffset: Long, length: Long) {
        throw NotImplementedError()
    }

    override fun write(operation: (ByteBuffer) -> Unit) {
        throw NotImplementedError()
    }
}
