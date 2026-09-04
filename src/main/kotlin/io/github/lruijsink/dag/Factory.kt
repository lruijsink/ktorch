package io.github.lruijsink.dag

open class Factory<A, R>(
    val name: String,
    val shape: ResultShape<R>,
) {
    operator fun invoke(args: A, device: Device): R =
        device.dispatch(this, args)
}

fun <A> factory(name: String): Factory<A, Tensor> =
    Factory(name, SingleResult)
