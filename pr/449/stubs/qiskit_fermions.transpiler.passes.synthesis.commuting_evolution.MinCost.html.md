# MinCost

### *class* MinCost(cost=<function device_cost>)

A [`Selector`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Selector.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Selector) that chooses the variant of the lowest cost.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Ties go to the earlier variant.

Each variant is scored once. A variant whose cost raises
[`TranspilerError`](https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler#qiskit.transpiler.TranspilerError) (a routed cost that cannot route it) is
dropped, and when no variant is left the gate is declined. A single variant is returned without
being scored.

* **Parameters:**
  **cost** (*Callable* *[* *[**QuantumCircuit* *,* [*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context) *]* *,* *Any* *]*) – maps a circuit and its [`Context`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context) to a value; smaller is better. The values of
  all variants must be comparable with `<` among themselves, a number or a tuple of
  numbers, say. The default, [`device_cost()`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.device_cost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.device_cost), routes onto the device when one is
  known.

Configures the selector; see the class documentation for the argument.
