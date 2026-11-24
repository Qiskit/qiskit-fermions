# TwinFold

### *class* TwinFold

`plain` followed by CX layers that fold qubits occurring in exactly the same terms.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

It applies only to groups that have such twins (see `twin_fold_clifford()`), and offers
nothing otherwise.

**Variants:** `"forward"` and `"reversed"`, the two orders of the layers of weight-two
rotations. They commute, so both are exact; which one routes better depends on how the layers
meet the gates of the Clifford.

### Attributes

#### name *= 'twin-fold'*

### Methods

#### generate(problem, context)

* **Parameters:**
  * **problem** ([*Problem*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem))
  * **context** ([*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context))
* **Return type:**
  [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*QuantumCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit)]
