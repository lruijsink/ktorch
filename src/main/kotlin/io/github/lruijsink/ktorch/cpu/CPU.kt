package io.github.lruijsink.ktorch.cpu

import io.github.lruijsink.ktorch.Device
import io.github.lruijsink.ktorch.PrimitiveType
import io.github.lruijsink.ktorch.Storage
import io.github.lruijsink.ktorch.Tensor
import java.nio.ByteBuffer

/**
 * CPU device type
 */
object CPU : Device {
    override fun allocate(size: Long): Storage =
        RAM(size)

    override fun clone(t: Tensor): Tensor =
        ops(t).cloneContiguous(t)

    override fun neg(t: Tensor): Tensor =
        ops(t).neg(t)

    override fun add(a: Tensor, b: Tensor): Tensor =
        ops(a).add(a, b)

    override fun sub(a: Tensor, b: Tensor): Tensor =
        ops(a).sub(a, b)

    override fun mul(a: Tensor, b: Tensor): Tensor =
        ops(a).mul(a, b)

    override fun div(a: Tensor, b: Tensor): Tensor =
        ops(a).div(a, b)

    private fun ops(tensor: Tensor): CPUOperations =
        when (tensor.primitiveType) {
            PrimitiveType.F32 -> F32Operations
            PrimitiveType.F64 -> TODO("F64 not yet supported")
        }
}

/**
 * CCPU buffer type, e.g. host memory
 */
class RAM(
    override val capacity: Long,
) : Storage {
    init {
        require(capacity > 0) { "size must be positive" }
        require(capacity <= Int.MAX_VALUE) { "cannot allocate more than Int.MAX_VALUE bytes" }
    }

    val byteBuffer: ByteBuffer = ByteBuffer.allocate(capacity.toInt())

    override var version: Int = 0
        private set

    override fun copyInto(receiver: ByteBuffer, sourceOffset: Long, length: Long) {
        require(sourceOffset >= 0)
        require(length >= 0)
        require(sourceOffset + length <= byteBuffer.limit().toLong())
        require(length <= receiver.remaining().toLong())

        val source = byteBuffer.duplicate()
        source.position(sourceOffset.toInt())
        source.limit((sourceOffset + length).toInt())

        receiver.put(source)
    }

    override fun write(operation: (ByteBuffer) -> Unit) {
        byteBuffer.rewind()
        operation(byteBuffer)
        version++
    }
}

/**
 * Operator definitions for a specific primitive type, used by the CPU
 */
interface CPUOperations {
    fun cloneContiguous(t: Tensor): Tensor

    fun neg(t: Tensor): Tensor

    fun add(a: Tensor, b: Tensor): Tensor

    fun sub(a: Tensor, b: Tensor): Tensor

    fun mul(a: Tensor, b: Tensor): Tensor

    fun div(a: Tensor, b: Tensor): Tensor
}
