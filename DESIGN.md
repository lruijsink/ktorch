# Design

## Operation implementation

Let's take the following example calculation:

```kotlin
exp(y) - matmuladd(M, x, b)
```

This contains the following three tensor operations:

- `exp`: Unary (elementwise) function
- `-`: Binary minus operator
- `matmuladd`: Fused/optimized operation

### exp

The exponent function, `exp(x)` = e<sup>x</sup>

```kotlin
class UnaryArg(t: Tensor)

val expOp = OpDef<UnaryArg>("exp")
fun exp(t: Tensor): Tensor =
    t.device.dispatch(expOp, UnaryArg(t))

cpu.impl(expOp) { (t) ->
    // compute exp(t) on the CPU
}

vulkan.impl(expOp) { (t) ->
    // compute exp(t) with Vulkan
}
```

This dispatches as following:

```kotlin
val y = exp(x) // calls Unary::invoke

class Unary {
    fun interface Impl : (Tensor) -> Tensor
  
    operator fun invoke(t: Tensor): Tensor =
        t.device.dispatch(this, t)
}

class Device {
    private val unaryImpls = mutableMapOf<Unary, Unary.Impl>()
  
    fun impl(op: Unary, impl: Unary.Impl) {
        unaryImpls[op] = impl
    }
    
    fun dispatch(op: Unary, t1: Tensor): Tensor =
        (unaryImpls[op] ?: throw NoImplException(op)).invoke(t1)
}

val cpu = Device()
val vulkan = Device()
```

### minus `-`

```kotlin
val minus = Fn2()

cpu.impl(minus) { x, y -> 
    // compute `x - y` on the CPU
}
```

Dispatch is very similar, the main difference is that there are built-in overloads for basic operators:

```kotlin
class Tensor(...) {
    operator fun minus(rhs: Tensor): Tensor =
        device.dispatch(minus, this, rhs)
}
```

Exactly the same as for unary operations, each device tracks binary operation implementations:

```kotlin
class Device {
    //...
    
    private val binaryImpls = mutableMapOf<Binary, Binary.Impl>()
  
    //...
  
    fun impl(op: Binary, impl: Binary.Impl) {
        binaryImpls[op] = impl
    }
    
    fun dispatch(op: Binary, t1: Tensor, t2: Tensor): Tensor =
        (binaryImpls[op] ?: throw NoImplException(op)).invoke(t1, t2)
}
```

### matmuladd

```kotlin
val matmuladd = Fn3 { M, x, b ->
    M matmul x + b // fallback implementation
}

vulkan.impl(matmuladd) { M, x, b ->
    // run optimized kernel
}
```

## Variadic operations

```kotlin
val c = concat(axis = 0, x, y, z, ...)

val concat = FnVariadic { 
    
}
```
