package io.github.lruijsink.dag

open class Op<A : OpArgs, R>(
    val name: String,
    val shape: ResultShape<R>,
) {
    operator fun invoke(args: A): R =
        args.device.dispatch(this, args)
}

abstract class OpArgs(
    val reads: List<Tensor>,
    val writes: List<Tensor> = listOf(),
    val attributes: Map<String, Any?> = mapOf(),
) {
    val device: Device

    init {
        val operands = reads + writes
        require(operands.isNotEmpty()) { "Require at least 1 operand. Use Factory for no-arg constructions." }

        val devices = (operands.map { it.device }).distinct()
        require(devices.size == 1) { "All operands must use the same device, got $devices" }

        device = devices[0]
    }
}

// ================================================================================
// Builders
// ================================================================================

fun <T : OpArgs> op(name: String): Op<T, Tensor> =
    Op(name, SingleResult)

fun unary(name: String) =
    op<UnaryArgs>(name)

fun binary(name: String) =
    op<BinaryArgs>(name)

fun <T : OpArgs> inPlaceOp(name: String): Op<T, Unit> =
    Op(name, NoResult)

fun unaryInPlace(name: String) =
    inPlaceOp<MutatingUnaryArgs>(name)

fun binaryInPlace(name: String) =
    inPlaceOp<MutatingBinaryArgs>(name)

// ================================================================================
// Standard arg types
// ================================================================================

data class UnaryArgs(
    val x: Tensor,
) : OpArgs(reads = listOf(x))

data class MutatingUnaryArgs(
    val x: Tensor,
) : OpArgs(reads = listOf(x), writes = listOf(x))

data class BinaryArgs(
    val x: Tensor,
    val y: Tensor,
) : OpArgs(reads = listOf(x, y))

data class MutatingBinaryArgs(
    val x: Tensor,
    val y: Tensor,
) : OpArgs(reads = listOf(x, y), writes = listOf(x))
