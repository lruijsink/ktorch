package io.github.lruijsink.ktorch

fun main() {
    val m = matrix(
        floatArrayOf(1f, 2f),
        floatArrayOf(3f, 4f),
        floatArrayOf(5f, 10f),
    )
    println(m.T)
}
