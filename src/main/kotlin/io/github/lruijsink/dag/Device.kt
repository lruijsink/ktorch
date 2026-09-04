package io.github.lruijsink.dag

fun interface Impl<A, R> : (A) -> R

class Device(
    val name: String,
) {
    val impls = mutableMapOf<Op<*, *>, Impl<*, *>>()
    val factoryImpls = mutableMapOf<Factory<*, *>, Impl<*, *>>()

    fun <A : OpArgs, R> impl(op: Op<A, R>, block: Impl<A, R>) {
        impls[op] = block
    }

    fun <A, R> impl(factory: Factory<A, R>, block: Impl<A, R>) {
        factoryImpls[factory] = block
    }

    @Suppress("UNCHECKED_CAST")
    fun <A : OpArgs, R> dispatch(op: Op<A, R>, args: A): R {
        val impl = impls[op] as? Impl<A, R>
            ?: throw IllegalArgumentException("Operation '${op.name}' is not implemented for device $name")

        return impl(args)
    }

    @Suppress("UNCHECKED_CAST")
    fun <A, R> dispatch(op: Factory<A, R>, args: A): R {
        val impl = factoryImpls[op] as? Impl<A, R>
            ?: throw IllegalArgumentException("Factory '${op.name}' is not implemented for device $name")

        return impl(args)
    }

    override fun toString(): String =
        name
}
