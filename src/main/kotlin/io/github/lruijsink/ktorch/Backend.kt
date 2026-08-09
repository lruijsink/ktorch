package io.github.lruijsink.ktorch

import io.github.lruijsink.ktorch.cpu.f32.F32CPU

/**
 * Compute device
 */
interface Backend {
    /**
     * Primitive type used by this backend
     */
    val primitiveType: PrimitiveType

    /**
     * Allocate [capacity] elements, size in bytes depends on [primitiveType]
     */
    fun allocate(capacity: Long): Storage

    /**
     * Copy a tensor, preserving its shape and stride
     */
    fun clone(t: Tensor): Tensor

    /**
     * Elementwise unary minus over a tensor
     */
    fun neg(t: Tensor): Tensor

    /**
     * Elementwise addition over two tensors
     */
    fun add(a: Tensor, b: Tensor): Tensor

    /**
     * Elementwise subtraction over two tensors
     */
    fun sub(a: Tensor, b: Tensor): Tensor

    /**
     * Elementwise multiplication (Hadamard product), over two tensors
     */
    fun mul(a: Tensor, b: Tensor): Tensor

    /**
     * Elementwise division over two tensors
     */
    fun div(a: Tensor, b: Tensor): Tensor

    /**
     * Elementwise `exp`
     */
    fun exp(t: Tensor): Tensor

    /**
     * Elementwise `ln`
     */
    fun ln(t: Tensor): Tensor

    /**
     * Elementwise `sqrt`
     */
    fun sqrt(t: Tensor): Tensor
}

/**
 * Default compute backend
 */
val DEFAULT_BACKEND: Backend = F32CPU
