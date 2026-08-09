package io.github.lruijsink.ktorch

import java.nio.FloatBuffer

/**
 * Create a rank N tensor from rank N-1 slices on the device
 */
fun tensor(vararg slices: Tensor): Tensor {
    if (slices.isEmpty()) return emptyTensor()

    val backend = slices[0].backend
    val sliceShape = slices[0].shape
    require(slices.all { it.shape.contentEquals(sliceShape) }) { "slices must have equal shapes" }
    require(slices.all { it.backend == backend }) { "slices must use the same backend" }

    val numElements = slices.size * slices[0].numElements
    val shape = intArrayOf(slices.size) + sliceShape
    val stride = contiguousStride(shape)

    val storage = backend.allocate(numElements)
    storage.write { buffer ->
        for (slice in slices) {
            if (slice.isContiguous) {
                slice.storage.copyInto(buffer)
            } else {
                TODO("Non-contiguous slices in initializers are not supported yet")
            }
        }
    }

    return Tensor(
        shape = shape,
        stride = stride,
        backend = backend,
        storage = storage,
        storageOffset = 0,
    )
}

/**
 * Create an empty tensor
 */
fun tensor(): Tensor =
    Tensor(
        shape = intArrayOf(),
        stride = intArrayOf(),
        backend = DEFAULT_BACKEND,
        storage = NoopStorage,
        storageOffset = 0,
    )

/**
 * Create an empty tensor, equivalent to `device.tensor()`
 */
fun emptyTensor(): Tensor =
    tensor()

// ================================================================================
// F32 specializations
// ================================================================================

private fun tensorF32(shape: IntArray, backend: Backend = DEFAULT_BACKEND, write: (FloatBuffer) -> Unit): Tensor {
    val numElements = shape.fold(1) { a, b -> a * b }
    val storage = backend.allocate(numElements.toLong())
    storage.write { write(it.asFloatBuffer()) }

    return Tensor(
        shape = shape,
        stride = contiguousStride(shape),
        backend = backend,
        storage = storage,
        storageOffset = 0,
    )
}

/**
 * Create a 0-dimensional scalar tensor on the device
 */
fun tensor(value: Float, backend: Backend = DEFAULT_BACKEND): Tensor =
    tensorF32(intArrayOf(), backend) { it.put(value) }

/**
 * Create a 1-dimensional vector tensor on the device
 */
fun tensor(elements: FloatArray, backend: Backend = DEFAULT_BACKEND): Tensor =
    tensorF32(intArrayOf(elements.size), backend) { it.put(elements) }

/**
 * Create a scalar tensor on the device, set to [value], equivalent to `device.tensor(value)`
 */
fun scalar(value: Float, backend: Backend = DEFAULT_BACKEND): Tensor =
    tensor(value, backend)

/**
 * Create a 1-dimensional vector tensor on the device
 */
fun tensor(vararg elements: Float): Tensor =
    tensorF32(intArrayOf(elements.size)) { it.put(elements) }

/**
 * Create a tensor on the device, set to [values], equivalent to tensor(values)
 */
fun vector(vararg values: Float): Tensor =
    tensor(*values)

/**
 * Create a matrix tensor on the device, from [FloatArray] vectors
 */
fun matrix(vararg values: FloatArray): Tensor =
    tensor(*values.map { vector(*it) }.toTypedArray())
