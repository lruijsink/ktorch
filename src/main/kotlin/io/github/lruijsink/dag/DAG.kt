package io.github.lruijsink.dag

import kotlin.math.exp

data class Tensor(
    val value: Float,
    val device: Device,
)

class OpDef<A>(
    val name: String,
)

data class UnaryArgs(
    val x: Tensor,
)

data class BinaryArgs(
    val x: Tensor,
    val y: Tensor,
)

fun interface Impl<A> {
    fun run(args: A): Tensor
}

class Device(
    val name: String,
) {
    val impls = mutableMapOf<OpDef<*>, Impl<*>>()

    fun <A> impl(op: OpDef<A>, block: Impl<A>) {
        impls[op] = block
    }

    @Suppress("UNCHECKED_CAST")
    fun <A> dispatch(op: OpDef<A>, args: A): Tensor =
        (impls[op] as? Impl<A> ?: throw NoImpl(op)).run(args)

    override fun toString(): String =
        name
}

class NoImpl(
    op: OpDef<*>,
) : RuntimeException("Operation '${op.name}' is not implemented for this device")

// ================================================================================
// Example 1: exp(x)

val expOp = OpDef<UnaryArgs>("exp")

fun exp(t: Tensor): Tensor =
    t.device.dispatch(expOp, UnaryArgs(t))

// ================================================================================
// Example 2: x + y

val addOp = OpDef<BinaryArgs>("add")

fun add(x: Tensor, y: Tensor): Tensor =
    x.device.dispatch(addOp, BinaryArgs(x, y))

operator fun Tensor.plus(other: Tensor): Tensor =
    add(this, other)

// ================================================================================
// Example 3: concat(axis, x, y, z, ...)

data class AxisConcatArgs(
    val axis: Int,
    val tensors: List<Tensor>,
)

val concatOp = OpDef<AxisConcatArgs>("concat")

fun concat(axis: Int, vararg tensors: Tensor): Tensor =
    tensors[0].device.dispatch(concatOp, AxisConcatArgs(axis, tensors.toList()))

// ================================================================================
// CPU device + implementations

val cpu = Device("cpu").apply {
    impl(expOp) { (x) -> Tensor(exp(x.value), x.device) }
    impl(addOp) { (x, y) -> Tensor(x.value + y.value, x.device) }
    impl(concatOp) { (axis, tensors) -> TODO("actually implement this") }
}

// ================================================================================
// Vulkan device + implementatins

val vulkan = Device("vulkan").apply {
    impl(expOp) { (x) -> Tensor(exp(x.value), x.device) }
    impl(addOp) { (x, y) -> Tensor(x.value + y.value, x.device) }
    impl(concatOp) { (axis, tensors) -> TODO("actually implement this") }
}

// ================================================================================
// main

fun main() {
    val x = Tensor(1f, cpu)
    val y = Tensor(1f, cpu)
    exp(x + y)
}
