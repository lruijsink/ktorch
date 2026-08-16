package io.github.lruijsink.kannoo

// ================================================================================
// Expressions

sealed class Expr {
    abstract fun mapChildren(map: (Expr) -> Expr): Expr

    operator fun unaryMinus(): Expr =
        UnaryMinus(this)

    operator fun plus(tensor: Expr): Expr =
        Plus(this, tensor)

    operator fun minus(tensor: Expr): Expr =
        Minus(this, tensor)

    operator fun times(tensor: Expr): Expr =
        Times(this, tensor)

    operator fun div(tensor: Expr): Expr =
        Div(this, tensor)
}

class Constant(
    val value: Float,
) : Expr() {
    override fun mapChildren(map: (Expr) -> Expr): Expr =
        this

    override fun toString(): String =
        value.toInt().toString()
}

class Variable(
    val name: String,
) : Expr() {
    override fun mapChildren(map: (Expr) -> Expr): Expr =
        this

    override fun toString(): String =
        name
}

val ZERO = Constant(0f)
val ONE = Constant(1f)

val Expr.isZero get(): Boolean = this is Constant && value == ZERO.value
val Expr.isOne get(): Boolean = this is Constant && value == ONE.value

// ================================================================================
// Unary operators

sealed class Unary(
    val expr: Expr,
) : Expr() {
    abstract fun mapConstant(constant: Constant): Constant

    override fun toString(): String =
        (this::class.simpleName).toString() + "($expr)"
}

class UnaryMinus(
    expr: Expr,
) : Unary(expr) {
    override fun mapChildren(map: (Expr) -> Expr): Expr =
        UnaryMinus(map(expr))

    override fun mapConstant(constant: Constant): Constant =
        Constant(-constant.value)

    override fun toString(): String =
        "(-$expr)"
}

// ================================================================================
// Binary operators

sealed class Binary(
    val lhs: Expr,
    val rhs: Expr,
) : Expr() {
    abstract fun mapConstants(lhs: Constant, rhs: Constant): Constant

    override fun toString(): String =
        (this::class.simpleName).toString() + "($lhs, $rhs)"
}

class Plus(
    lhs: Expr,
    rhs: Expr,
) : Binary(lhs, rhs) {
    override fun mapChildren(map: (Expr) -> Expr): Expr =
        Plus(map(lhs), map(rhs))

    override fun mapConstants(lhs: Constant, rhs: Constant): Constant =
        Constant(lhs.value + rhs.value)

    override fun toString(): String =
        "($lhs + $rhs)"
}

class Minus(
    lhs: Expr,
    rhs: Expr,
) : Binary(lhs, rhs) {
    override fun mapChildren(map: (Expr) -> Expr): Expr =
        Minus(map(lhs), map(rhs))

    override fun mapConstants(lhs: Constant, rhs: Constant): Constant =
        Constant(lhs.value - rhs.value)

    override fun toString(): String =
        "($lhs - $rhs)"
}

class Times(
    lhs: Expr,
    rhs: Expr,
) : Binary(lhs, rhs) {
    override fun mapChildren(map: (Expr) -> Expr): Expr =
        Times(map(lhs), map(rhs))

    override fun mapConstants(lhs: Constant, rhs: Constant): Constant =
        Constant(lhs.value * rhs.value)

    override fun toString(): String =
        "($lhs * $rhs)"
}

class Div(
    lhs: Expr,
    rhs: Expr,
) : Binary(lhs, rhs) {
    override fun mapChildren(map: (Expr) -> Expr): Expr =
        Div(map(lhs), map(rhs))

    override fun mapConstants(lhs: Constant, rhs: Constant): Constant =
        Constant(lhs.value / rhs.value)

    override fun toString(): String =
        "($lhs / $rhs)"
}

// ================================================================================
// Algebraic rewrites

fun mapConstants(expr: Expr): Expr =
    when (expr) {
        is Unary if expr.expr is Constant -> expr.mapConstant(expr.expr)
        is Binary if expr.lhs is Constant && expr.rhs is Constant -> expr.mapConstants(expr.lhs, expr.rhs)
        else -> expr
    }

fun zeroes(expr: Expr): Expr =
    when (expr) {
        is Plus if expr.lhs.isZero -> expr.rhs
        is Plus if expr.rhs.isZero -> expr.lhs
        is Minus if expr.lhs.isZero -> UnaryMinus(expr.rhs)
        is Minus if expr.rhs.isZero -> expr.lhs
        is Times if (expr.lhs.isZero || expr.rhs.isZero) -> ZERO
        is Div if expr.lhs == ZERO -> ZERO
        else -> expr
    }

fun ones(expr: Expr): Expr =
    when (expr) {
        is Times if expr.lhs.isOne -> expr.rhs
        is Times if expr.rhs.isOne -> expr.lhs
        is Div if expr.rhs.isOne -> expr.lhs
        else -> expr
    }

fun unaryMinus(expr: Expr): Expr =
    when (expr) {
        // (-x) + y => y - x
        is Plus if expr.lhs is UnaryMinus -> Minus(expr.rhs, expr.lhs.expr)

        // x + (-y) => x - y
        is Plus if expr.rhs is UnaryMinus -> Minus(expr.lhs, expr.rhs.expr)

        // x - (-y) => x + y
        is Minus if expr.rhs is UnaryMinus -> Plus(expr.lhs, expr.rhs.expr)

        // -(x - y) => y - x
        is UnaryMinus if expr.expr is Minus -> Minus(expr.expr.rhs, expr.expr.lhs)

        // -(-x) => x
        is UnaryMinus if expr.expr is UnaryMinus -> expr.expr.expr

        else -> expr
    }

typealias AlgebraicRewrite = (Expr) -> Expr

val algebraicRewrites: List<AlgebraicRewrite> = listOf(
    ::mapConstants,
    ::zeroes,
    ::ones,
    ::unaryMinus,
)

fun Expr.applyAlgebraicRewrites(): Expr =
    algebraicRewrites.fold(this) { expr, fn -> fn(expr) }

fun Expr.fixedPoint(): Expr {
    var current = this
    while (true) {
        val next = current.applyAlgebraicRewrites()
        if (next == current) return current
        println("${current.pretty().padStart(20)}  =>  ${next.pretty()}")
        current = next
    }
}

fun Expr.pretty() =
    if (this is Unary || this is Binary) {
        this.toString().drop(1).dropLast(1)
    } else {
        this.toString()
    }

// ================================================================================
// Simplify expression pipeline

fun Expr.simplify(): Expr =
    this.mapChildren { it.simplify() }.fixedPoint()

// ================================================================================
// Example

fun main() {
    println("----------------------------------------")
    println()

    // execution context
    class Tensor(val v: Float)

    val args: Array<Any?> = arrayOf(1, Tensor(0f), Tensor(1f))
    val axis = args[0] as Int
    val tensors = args.drop(0).map { it as Tensor }

    println()
    println("----------------------------------------")
    println()

    val x = Variable("x")
    val y = Variable("y")

    var e = -(-x + (y * ONE)) - ZERO

    e += ONE

    println(e.pretty())
    println()

    val eSimple = e.simplify()
    println()

    println(eSimple.pretty())

    println()
    println("----------------------------------------")
}
