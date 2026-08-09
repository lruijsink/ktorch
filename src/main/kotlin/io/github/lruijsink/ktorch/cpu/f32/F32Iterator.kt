package io.github.lruijsink.ktorch.cpu.f32

interface F32Iterator {
    fun next(): Float

    fun hasRemaining(): Boolean

    fun reset()
}
