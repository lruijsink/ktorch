package io.github.lruijsink.ktorch

import java.nio.ByteBuffer

/**
 * Represents physical storage on any of the available devices.
 */
interface Storage {
    val capacity: Long
    val version: Int

    fun copyInto(receiver: ByteBuffer, sourceOffset: Long = 0, length: Long = capacity)

    fun write(operation: (ByteBuffer) -> Unit)
}
