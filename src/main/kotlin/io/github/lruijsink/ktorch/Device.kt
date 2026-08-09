package io.github.lruijsink.ktorch

import io.github.lruijsink.ktorch.cpu.CPU

/**
 * Compute device
 */
interface Device {
    /**
     * Allocate [size] bytes
     */
    fun allocate(size: Long): Storage

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
}

/**
 * Default compute device (CPU)
 */
val DEFAULT_DEVICE: Device = CPU
