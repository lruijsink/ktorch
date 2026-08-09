package io.github.lruijsink.ktorch

import java.nio.FloatBuffer

/**
 * Create a rank N tensor from rank N-1 slices on the device
 */
fun tensor(slices: Array<Tensor>, device: Device = DEFAULT_DEVICE): Tensor {
    if (slices.isEmpty()) return emptyTensor()

    val sliceShape = slices[0].shape
    val primitiveType = slices[0].primitiveType
    require(slices.all { it.shape.contentEquals(sliceShape) }) { "slices must have equal shapes" }
    require(slices.all { it.primitiveType == primitiveType }) { "slices must have the same primitive type" }

    val numElements = slices.size * slices[0].numElements
    val shape = intArrayOf(slices.size) + sliceShape
    val stride = contiguousStride(shape)

    val storage = device.allocate(primitiveType.sizeBytes * numElements)
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
        primitiveType = primitiveType,
        device = device,
        storage = storage,
        storageOffset = 0,
    )
}

/**
 * Create a rank N tensor from rank N-1 slices on the device
 */
fun tensor(vararg slices: Tensor): Tensor =
    @Suppress("UNCHECKED_CAST", "KotlinConstantConditions")
    tensor(slices as Array<Tensor>, DEFAULT_DEVICE)

/**
 * Create an empty tensor
 */
fun tensor(): Tensor =
    Tensor(
        shape = intArrayOf(),
        stride = intArrayOf(),
        primitiveType = PrimitiveType.F32, // irrelevant
        device = DEFAULT_DEVICE, // irrelevant
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

private fun tensorF32(shape: IntArray, device: Device = DEFAULT_DEVICE, write: (FloatBuffer) -> Unit): Tensor {
    val numElements = shape.fold(1) { a, b -> a * b }
    val storage = device.allocate(numElements * Float.SIZE_BYTES.toLong())
    storage.write { write(it.asFloatBuffer()) }

    return Tensor(
        shape = shape,
        stride = contiguousStride(shape),
        device = device,
        storage = storage,
        storageOffset = 0,
        primitiveType = PrimitiveType.F32,
    )
}

/**
 * Create a 0-dimensional scalar tensor on the device
 */
fun tensor(value: Float, device: Device = DEFAULT_DEVICE): Tensor =
    tensorF32(intArrayOf(), device) { it.put(value) }

/**
 * Create a 1-dimensional vector tensor on the device
 */
fun tensor(elements: FloatArray, device: Device = DEFAULT_DEVICE): Tensor =
    tensorF32(intArrayOf(elements.size), device) { it.put(elements) }

/**
 * Create a scalar tensor on the device, set to [value], equivalent to `device.tensor(value)`
 */
fun scalar(value: Float, device: Device = DEFAULT_DEVICE): Tensor =
    tensor(value, device)

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
