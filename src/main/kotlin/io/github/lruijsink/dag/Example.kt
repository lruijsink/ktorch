package io.github.lruijsink.dag

fun main() {
    val x = vector(1, 2, 3)
    val y = vector(4f, 5f, 6f)
    val z = exp(x - 10.0 / y)
    println(z.elements())
}

private fun Tensor.elements(): List<Float> {
    val res = FloatArray(numElementsOfShape(shape).toInt())
    (storage as RAM).floats().get(res)
    return res.toList()
}
