package io.github.lruijsink.ktorch

/**
 * Primitive to use for storage and computation
 */
enum class PrimitiveType(
    val sizeBytes: Int,
) {
    /**
     * 32-bit floating point numbers, e.g. [Float]
     */
    F32(Float.SIZE_BYTES),

    /**
     * 64-bit floating point numbers, e.g. [Double]
     */
    F64(Double.SIZE_BYTES),
}
