package io.github.lruijsink.dag

interface ResultShape<R> {
    fun wrap(results: List<Tensor>): R

    fun unwrap(value: R): List<Tensor>
}

object NoResult : ResultShape<Unit> {
    override fun wrap(results: List<Tensor>) {
        require(results.isEmpty()) { "expected no results (Unit), got $results" }
    }

    override fun unwrap(value: Unit): List<Tensor> =
        listOf()
}

object SingleResult : ResultShape<Tensor> {
    override fun wrap(results: List<Tensor>): Tensor {
        require(results.size == 1) { "expected 1 result, got ${results.size}" }
        return results[0]
    }

    override fun unwrap(value: Tensor) =
        listOf(value)
}
