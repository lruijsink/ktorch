package io.github.lruijsink.dag

enum class PrimitiveType(
    val sizeBytes: Int,
) {
    F32(sizeBytes = Float.SIZE_BYTES),
    F64(sizeBytes = Double.SIZE_BYTES),
}
