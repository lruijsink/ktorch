package io.github.lruijsink.dag

val unaryPlusOp = unary("unaryPlus")
val unaryMinusOp = unary("unaryMinus")

fun unaryPlus(x: Tensor): Tensor =
    unaryPlusOp(UnaryArgs(x))

fun unaryMinus(x: Tensor): Tensor =
    unaryMinusOp(UnaryArgs(x))

val plusOp = binary("plus")
val minusOp = binary("minus")
val timesOp = binary("timesOp")
val divOp = binary("divOp")

fun plus(x: Tensor, y: Tensor): Tensor =
    plusOp(BinaryArgs(x, y))

fun minus(x: Tensor, y: Tensor): Tensor =
    minusOp(BinaryArgs(x, y))

fun times(x: Tensor, y: Tensor): Tensor =
    timesOp(BinaryArgs(x, y))

fun div(x: Tensor, y: Tensor): Tensor =
    divOp(BinaryArgs(x, y))

val plusAssignOp = binaryInPlace("plusAssign")
val minusAssignOp = binaryInPlace("minusAssign")
val timesAssignOp = binaryInPlace("timesAssign")
val divAssignOp = binaryInPlace("divAssign")

fun plusAssign(x: Tensor, y: Tensor) {
    plusAssignOp(MutatingBinaryArgs(x, y))
}

fun minusAssign(x: Tensor, y: Tensor) {
    minusAssignOp(MutatingBinaryArgs(x, y))
}

fun timesAssign(x: Tensor, y: Tensor) {
    timesAssignOp(MutatingBinaryArgs(x, y))
}

fun divAssign(x: Tensor, y: Tensor) {
    divAssignOp(MutatingBinaryArgs(x, y))
}
