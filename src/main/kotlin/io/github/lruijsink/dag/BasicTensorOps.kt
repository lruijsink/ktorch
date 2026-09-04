package io.github.lruijsink.dag

data class AxisConcatArgs(
    val axis: Int,
    val tensors: List<Tensor>,
) : OpArgs(reads = tensors, attributes = mapOf("axis" to axis))

val concatOp = op<AxisConcatArgs>("concat")

fun concat(axis: Int, vararg tensors: Tensor): Tensor =
    concatOp(AxisConcatArgs(axis, tensors.toList()))
