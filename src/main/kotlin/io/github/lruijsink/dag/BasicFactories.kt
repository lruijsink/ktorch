package io.github.lruijsink.dag

val scalarFactory = factory<Float>("scalar")
val vectorFactory = factory<FloatArray>("tensor")

fun scalar(value: Float, device: Device = cpu): Tensor =
    scalarFactory(value, device)

fun vector(vararg elements: Float): Tensor =
    vectorFactory(elements, device = cpu)

fun vector(elements: FloatArray, device: Device = cpu): Tensor =
    vectorFactory(elements, device)

fun scalar(value: Number, device: Device = cpu): Tensor =
    scalar(value.toFloat(), device) // TODO: support other scalar types

fun vector(vararg elements: Number, device: Device = cpu): Tensor =
    vector(elements.toFloatArray(), device) // TODO: support other scalar types

private fun Array<out Number>.toFloatArray(): FloatArray =
    FloatArray(this.size) { i -> this[i].toFloat() }
