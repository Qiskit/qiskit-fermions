# group_wise

### group_wise(mapper_fn)

Wraps a mapper function so that it maps an operator one group at a time.

Rather than mapping the operator in one go, the wrapped function maps each of its
[`groups`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.groups) separately and sums the results (see
[Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation)). The sum is the same operator either way, but its Pauli terms come
out grouped: the terms of one group are adjacent instead of interleaved with everybody else’s.

That matters because a product formula synthesizes the terms in the order it receives them. Where
the terms of a group act on disjoint qubits, having them adjacent lets them be scheduled in
parallel, which can reduce the two-qubit depth substantially at an unchanged gate count.

Use it to wrap any `MapperFunction`:

```default
MapperFnEvolutionSynthesis(group_wise(jordan_wigner))
```

An operator without groups is mapped in one go, exactly as `mapper_fn` would on its own.

#### NOTE
Composing this with [`simplify()`](qiskit_fermions.transpiler.passes.synthesis.simplify.html.md#qiskit_fermions.transpiler.passes.synthesis.simplify) defeats the purpose, since simplifying re-sorts the terms
into a canonical order and so discards the grouping this produces.

* **Parameters:**
  **mapper_fn** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[*[*OperatorTrait*](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]* *,* [*SparseObservable*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable) *]*) – the mapper function to wrap.
* **Returns:**
  A mapper function that maps `mapper_fn` over the operator’s groups and sums the results.
* **Return type:**
  [*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[*OperatorTrait*](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait), [int](https://docs.python.org/3/builtins/functions.html#int)], [*SparseObservable*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable)]
