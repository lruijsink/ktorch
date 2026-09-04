package io.github.lruijsink.dag

import java.nio.ByteBuffer
import java.nio.FloatBuffer

interface Storage {
    val sizeBytes: Long
}

class RAM(
    override val sizeBytes: Long,
) : Storage {
    private val buffer: ByteBuffer = ByteBuffer.allocate(sizeBytes.toInt())

    init {
        require(sizeBytes > 0) { "Cannot allocate a non-positive number of bytes $sizeBytes" }
        require(sizeBytes < Int.MAX_VALUE) { "Cannot allocate $sizeBytes bytes as RAM, max. is ${Int.MAX_VALUE}" }
    }

    fun floats(offsetBytes: Long = 0): FloatBuffer {
        require(offsetBytes >= 0) { "Offset may not be negative, got $offsetBytes" }
        require(offsetBytes < sizeBytes) { "Offset $offsetBytes is outside buffer capacity of $sizeBytes" }
        return buffer.position(offsetBytes.toInt()).asFloatBuffer()
    }

    fun floats(numElements: Long, offsetBytes: Long = 0): FloatBuffer {
        require(numElements >= 0) { "Number of elements must be non-negative, got $numElements" }
        require(numElements * Float.SIZE_BYTES + offsetBytes <= sizeBytes) {
            "Cannot view $numElements elements from offset byte $offsetBytes: buffer size too small ($sizeBytes bytes)"
        }
        return floats(offsetBytes).limit(numElements.toInt())
    }

    override fun toString(): String =
        "RAM(sizeBytes=$sizeBytes)"
}
