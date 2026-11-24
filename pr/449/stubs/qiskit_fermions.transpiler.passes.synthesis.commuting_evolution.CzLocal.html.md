# CzLocal

### *class* CzLocal

CZ layers on the allowed pairs between single-qubit layers.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

It needs the allowed CZ pairs ([`Context.edges`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.edges)): the device’s, once the gate’s qubits are
placed, or the user’s. Without them it offers nothing. Every CZ gate is on an allowed pair, and
so is every block of the pair-block variants.

**Variants:**

* `"diagonal"`: `U = L2 CZ(G) L1` sends every term to a Z-string, and the middle is a
  diagonal rotation stage. `G` and the letters come out of a linear system over GF(2); there
  is no variant when it has no solution.
* `"pair-blocks"`: `U = CZ(G) L1`, with `G` a matching, only has to send every term onto
  one qubit or one allowed pair, and the middle holds one generic two-qubit block per pair. This
  weaker target helps when the terms on a pair form an entangled set such as `{XX, YY, ZZ}`,
  which no single-qubit layer diagonalizes. The circuit needs no routing.
* `"fswap-pair-blocks"`: the same target, for the components of the group that one CZ layer
  cannot solve, with `U = CZ(F) L2 CZ(F) L1`, both CZ layers on the same matching `F`. Such
  a `U` is a layer of moves of the kind of the fermionic swap `fSWAP = SWAP . CZ` (two CX
  gates each): it brings a term onto a pair that one CZ layer cannot reach, by moving a mode
  next to its partner. The components that one CZ layer solves keep that solution, and the
  variant is offered only when some component needed the moves. The circuit needs no routing.

### Attributes

#### name *= 'cz-local'*

### Methods

#### generate(problem, context)

* **Parameters:**
  * **problem** ([*Problem*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem))
  * **context** ([*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context))
* **Return type:**
  [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*QuantumCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit)]
