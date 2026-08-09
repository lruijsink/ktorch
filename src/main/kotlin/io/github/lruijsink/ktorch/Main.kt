package io.github.lruijsink.ktorch

fun main() {
    val m = matrix(
        floatArrayOf(0.001f, 2f),
        floatArrayOf(3f, 4f),
        floatArrayOf(5f, 10f),
    )
    println(m * 2)

    val x = vector(1f, 2f, 3f)
    val y = vector(4f, 5f, 6f)
    println(x * y)
}
