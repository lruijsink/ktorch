package io.github.lruijsink.ktorch.cpu.f32

import io.github.lruijsink.ktorch.Backend
import io.github.lruijsink.ktorch.PrimitiveType
import io.github.lruijsink.ktorch.Storage
import io.github.lruijsink.ktorch.Tensor
import io.github.lruijsink.ktorch.contiguousStride

object F32CPU : Backend {
    override val primitiveType: PrimitiveType = PrimitiveType.F32

    override fun allocate(capacity: Long): Storage =
        F32RAM(capacity)

    override fun clone(t: Tensor): Tensor =
        unary(t) { it }

    override fun neg(t: Tensor): Tensor =
        unary(t) { -it }

    override fun exp(t: Tensor): Tensor =
        unary(t) { kotlin.math.exp(it) }

    override fun ln(t: Tensor): Tensor =
        unary(t) { kotlin.math.ln(it) }

    override fun sqrt(t: Tensor): Tensor =
        unary(t) { kotlin.math.sqrt(it) }

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
        val s = allocate(t.numElements)
        s.write {
            val buff = it.asFloatBuffer()
            while (iter.hasRemaining()) buff.put(op(iter.next()))
        }
        return Tensor(
            shape = t.shape,
            stride = contiguousStride(t.shape),
            backend = this,
            storage = s,
            storageOffset = 0,
        )
    }

    private fun binary(a: Tensor, b: Tensor, op: (Float, Float) -> Float): Tensor {
        val aIter = a.iterator()
        val bIter = b.iterator()
        val s = allocate(a.numElements)
        s.write {
            val sBuff = it.asFloatBuffer()
            while (aIter.hasRemaining()) sBuff.put(op(aIter.next(), bIter.next()))
        }
        return Tensor(
            shape = a.shape,
            stride = contiguousStride(a.shape),
            backend = this,
            storage = s,
            storageOffset = 0,
        )
    }

    private fun Tensor.iterator(): F32Iterator =
        if (isContiguous) F32ContiguousIterator(this) else F32NonContiguousIterator(this)
}
