# device_cost

### device_cost(circuit, context, , basis=('cz', 'rz', 'sx', 'x'))

The two-qubit depth and count after a quick transpile onto the device, when one is known.

The circuit is transpiled into the basis of the context’s target, or into `basis` without one.
With a coupling map and the physical qubits, qubit `i` of the circuit is placed on device
qubit `physical_qubits[i]`, so that the router sees exactly the connectivity the real circuit
will see. With a coupling map only, the router chooses the placement. Without a coupling map,
nothing is routed.

A router may leave the qubits permuted at the end. The circuit that follows expects every qubit
back in its place, so a permutation is undone with SWAP gates on device edges before the circuit
is counted. Otherwise a circuit that the router leaves permuted would look cheaper than one that
needs no routing at all.

* **Parameters:**
  * **circuit** (*QuantumCircuit*) – the circuit to score.
  * **context** ([*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context)) – where it will run.
  * **basis** (*Sequence* *[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *]*) – the basis gates to translate into when the context has no target. To score in another
    basis, bind it with [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial) before passing the function to
    [`MinCost`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost).
* **Returns:**
  `(two-qubit depth, two-qubit gate count)`.
* **Raises:**
  **TranspilerError** – if the circuit cannot be routed onto the coupling map. [`MinCost`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost)
      drops such a circuit.
* **Return type:**
  [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), [int](https://docs.python.org/3/builtins/functions.html#int)]
