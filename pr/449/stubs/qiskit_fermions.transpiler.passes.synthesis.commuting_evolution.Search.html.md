# Search

### *class* Search(, variants=('star-layered', 'star-elimination-order', 'tree-colored-layered', 'tree-colored-elimination-order'))

The `plain` elimination after per-qubit bases found by a coordinate descent.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The descent minimizes the two-qubit gate count of the Clifford, then its two-qubit depth.

It is the most expensive candidate, and it pays off only for a gate that is not placed on the
device, which is the case when the high-level synthesis runs in the `init` stage: there it
often finds a shallower circuit than every other candidate, on structured groups too. Once the
gate is placed ([`Context.placed`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.placed)), the candidates that use the device’s connectivity do
better, so it offers nothing.

**Variants:** `"<shape>-<order>"` for each elimination shape and each CX order (see
[`Plain`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Plain.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Plain)), in the order of `VARIANTS`. The shape is `"star"`, a star fan-in in
natural row order, or `"tree-colored"`, a balanced fan-in tree with the rows eliminated color
class by color class. The descent ranks the bases with the layered order, which therefore comes
first for each shape. When the two orders of a shape give the same circuit, only the first one
requested is offered.

* **Parameters:**
  **variants** (*Sequence* *[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *]*) – the variants to build. Every shape among them costs one descent, shared by its two
  orders.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `variants` is empty or names an unknown variant.

### Attributes

#### name *= 'search'*

### Methods

#### generate(problem, context)

* **Parameters:**
  * **problem** ([*Problem*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem))
  * **context** ([*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context))
* **Return type:**
  [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*QuantumCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit)]
