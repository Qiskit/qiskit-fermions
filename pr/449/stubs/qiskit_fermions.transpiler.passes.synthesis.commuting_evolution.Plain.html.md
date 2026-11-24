# Plain

### *class* Plain

GF(2) elimination of the terms with locally chosen pivots, then the diagonal part.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

**Variants:** `"elimination-order"` and `"layered"`, the two orders in which the CX gates of
the elimination can be emitted (one only, when they coincide).

### Attributes

#### name *= 'plain'*

### Methods

#### generate(problem, context)

* **Parameters:**
  * **problem** ([*Problem*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem))
  * **context** ([*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context))
* **Return type:**
  [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*QuantumCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit)]
