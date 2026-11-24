# Candidate

### *class* Candidate(\*args, \*\*kwargs)

One way of building `exp(-i t H)` for a group of mutually commuting Paulis.

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A candidate offers circuits; choosing between them, and between the candidates, is the
selector’s job. So a candidate never ranks its own output: when it can build the evolution in
several ways (an order of commuting gates, parity trees or parity ladders), it returns each of
them as a variant. A candidate whose variants are costly to build may let the user choose which
of them it builds, with a `variants` argument. It may run any search or heuristic of its own
on the way to a circuit.

A candidate must be exact (every circuit it returns implements the evolution, including the
global phase) and deterministic, and should keep no state between gates.

### Attributes

#### name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The candidate’s name. It labels its variants, and must be unique among the plugin’s
candidates.

### Methods

#### generate(problem, context)

The candidate’s circuits for this problem, by variant name.

The mapping is empty when the candidate has nothing to offer.

The order of the mapping is the order in which ties between the variants are broken. A
candidate with a single circuit may name its variant `""`.

* **Parameters:**
  * **problem** ([*Problem*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem))
  * **context** ([*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context))
* **Return type:**
  [*Mapping*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*QuantumCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit)]
