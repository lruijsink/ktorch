package io.github.lruijsink.dag

val expOp = unary("exp")

fun exp(x: Tensor): Tensor =
    expOp(UnaryArgs(x))
