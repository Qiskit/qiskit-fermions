# abstract_cost

### abstract_cost(circuit, context=None, , basis=('cz', 'rz', 'sx', 'x'))

The two-qubit depth and gate count of the circuit in `basis`, without routing.

The circuit is translated by Qiskit’s high-level synthesis and basis translation, as in a
transpile without layout, routing or optimization, so every gate is counted the way the device
would execute it. The count covers every two-qubit gate of the result.

* **Parameters:**
  * **circuit** (*QuantumCircuit*) – the circuit to score.
  * **context** ([*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context) *|* *None*) – ignored; accepted so that this is a cost function for [`MinCost`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost).
  * **basis** (*Sequence* *[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *]*) – the basis gates to translate into. To score in another basis, bind it with
    [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial) before passing the function to [`MinCost`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost).
* **Returns:**
  `(two-qubit depth, two-qubit gate count)`.
* **Return type:**
  [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), [int](https://docs.python.org/3/builtins/functions.html#int)]
