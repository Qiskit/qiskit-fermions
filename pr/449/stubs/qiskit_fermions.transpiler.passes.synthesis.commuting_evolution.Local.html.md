# Local

### *class* Local(, variants=('tree', 'ladder'))

Each term diagonalized on its own, with no shared Clifford.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Terms on disjoint qubits are packed into layers.

Cheap, and the stock recipe on groups without structure.

**Variants:** `"tree"` (log-depth parity trees) and `"ladder"` (nearest-neighbor parity
ladders in sorted qubit order).

* **Parameters:**
  **variants** (*Sequence* *[**ParityShape* *]*) – the variants to build.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `variants` is empty or names an unknown variant.

### Attributes

#### name *= 'local'*

### Methods

#### generate(problem, context)

* **Parameters:**
  * **problem** ([*Problem*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem))
  * **context** ([*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context))
* **Return type:**
  [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*QuantumCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit)]
