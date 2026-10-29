# GivensDecompositionSlaterDeterminantSynthesis

### *class* GivensDecompositionSlaterDeterminantSynthesis(, minimize_2q_gate_count=False)

A [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) for transpiling [`PrepareSlaterDeterminant`](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant).

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

This plugin exploits the known reference occupation to synthesize the gate with the rectangular
[`givens_decomposition_slater()`](qiskit_fermions.linalg.givens_decomposition_slater.html.md#qiskit_fermions.linalg.givens_decomposition_slater): only the $m$ occupied orbitals
of the rotation need be realized, so at most $m (n - m)$ `XXPlusYYGate` rotations are
emitted (versus the $n (n - 1) / 2$ brick-wall plus $n$ phase gates that
[`GivensDecompositionOrbitalRotationSynthesis`](qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionOrbitalRotationSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionOrbitalRotationSynthesis) would use for the full
[`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation)). The reduced decomposition carries no diagonal phases – a global phase
and any rotation within the occupied space leave the prepared Slater determinant unchanged – so
the state is prepared correct up to a global phase.

#### NOTE
[`givens_decomposition_slater()`](qiskit_fermions.linalg.givens_decomposition_slater.html.md#qiskit_fermions.linalg.givens_decomposition_slater) is defined against a fixed
*leading*-$m$ reference (the first $m$ modes occupied). The emitted `XGate`s
therefore seed the leading modes of the decomposed block, **not** the qubits pointed to by
`occupation` – the Givens sweep then transports the occupation onto the physical target
orbitals. The positions of `occupation`’s occupied modes enter only through which columns of
`rotation_unitary` are selected (`rotation_unitary[:, occupied]`); their *count* $m$
determines the reference. The final prepared state is correct for any occupation, contiguous
or not.

#### NOTE
*Optionally* (when [`minimize_2q_gate_count`](#qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionSlaterDeterminantSynthesis.minimize_2q_gate_count) is set; it is **off** by default – see the
caution below) occupied orbitals that are basis vectors on a mode disjoint from every other
occupied orbital are *peeled off* before the decomposition: their `XGate` is placed directly
on the target mode and they are excluded from the Givens sweep. This avoids the chain of bare
`SWAP`s (`XXPlusYYGate` with `c = 0`) that the leading-reference sweep would otherwise
emit to transport such an orbital out to its mode. Only the genuinely-mixing remainder is then
decomposed; when the whole occupied space is a signed permutation every orbital peels and no
`XXPlusYYGate` is emitted at all.

A peeled mode may sit *between* the physical endpoints of a sweep rotation, making that
`XXPlusYYGate` act on physically non-adjacent qubits. Because a peeled mode is an occupied
unit-orbital, the Jordan-Wigner Z-string it would contribute is folded back into the sign of
the `XXPlusYYGate` phase when an odd number of peeled modes lies strictly between the
endpoints, so the prepared state stays correct regardless of how the peeled modes interleave
the mixing space.

#### NOTE
The occupied space (the genuinely-mixing remainder that survives peeling, when
[`minimize_2q_gate_count`](#qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionSlaterDeterminantSynthesis.minimize_2q_gate_count) is set; the full occupied space otherwise) is further
partitioned into disjoint-support **blocks**, each synthesized on its own contiguous mode
window. When it splits into several mixing clusters separated by empty (or peeled) modes, this
drops the bare `SWAP`s (`XXPlusYYGate` with `c = 0`) the single leading-reference sweep
would otherwise emit to transport occupation across the gaps between clusters. A single mixing
cluster spanning all kept modes is one block and reproduces the plain leading-reference sweep
exactly. This block splitting runs regardless of [`minimize_2q_gate_count`](#qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionSlaterDeterminantSynthesis.minimize_2q_gate_count).

#### CAUTION
The unit-orbital peeling above trades circuit metrics: it minimizes the emitted two-qubit gate
*count* (replacing a swept orbital by a direct `XGate`), but only under free (all-to-all)
connectivity. On a constrained coupling map it can *increase* the two-qubit gate *depth* after
routing – peeling occupied modes that the leading-reference sweep would have used as
nearest-neighbor stepping-stones can strand the residual mixing orbital across a wide,
physically non-adjacent window (a single long-range `XXPlusYYGate` that routing must then
bridge with many `SWAP`s). The full leading-reference sweep instead keeps every emitted
rotation nearest-neighbor along the mode order – more two-qubit gates, but no routing overhead.
The two are a genuine, non-dominated trade-off (neither ever beats the other on both metrics),
so which is preferable depends on the target: minimizing two-qubit *depth* generally matters
more than raw gate *count*, both on hardware and in simulation, so peeling is **off** by
default. Enable it via [`minimize_2q_gate_count`](#qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionSlaterDeterminantSynthesis.minimize_2q_gate_count) when raw gate count is the bottleneck
(e.g. richly-connected hardware). Both settings prepare the same state (up to a global phase).

#### WARNING
This transpilation pass plugin makes the following assumptions:

- an occupation-basis encoding (like Jordan-Wigner)
- a trivial fermion-to-qubit layout (i.e. no change in their register lengths)
- a 1-to-1 mapping of fermionic mode indices to qubit indices

Initializing this plugin can be done with the arguments listed below.

* **Parameters:**
  **minimize_2q_gate_count** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – which of two equivalent syntheses to emit, selected by the circuit
  metric to optimize. When `False` (the default), the full leading-reference Givens
  sweep is used: every emitted two-qubit rotation is nearest-neighbor along the mode
  order, minimizing the post-routing two-qubit *depth* on a constrained coupling map at
  the cost of more two-qubit gates. When `True`, disjoint unit-orbitals (occupied
  orbitals that are basis vectors on a mode disjoint from all other occupied orbitals)
  are peeled out of the decomposition and realized with a direct `XGate`, minimizing
  the two-qubit gate *count* but potentially emitting long-range rotations that inflate
  the routed depth (see the caution in the class docstring). Both settings prepare the
  same state (up to a global phase); the default favors depth because it generally
  matters more than gate count, both on hardware and in simulation.

### Attributes

#### minimize_2q_gate_count

Whether to peel disjoint unit-orbitals to minimize the two-qubit gate count (see
`__init__`). When `False` (default), the nearest-neighbor leading-reference sweep is used
instead, minimizing the routed two-qubit depth.

### Methods

#### run(in_node, out_dag, , f2q_layout)

Runs this transpilation plugin.

* **Parameters:**
  * **in_node** ([*DAGOpNode*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode)) – the input fermion-based circuit instruction. When this plugin gets called, the
    `in_node.op` attribute must be of type [`PrepareSlaterDeterminant`](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant).
  * **out_dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)) – the output qubit-based circuit.
  * **f2q_layout** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *[*[*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *,* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *]*) – the global transpilation [`F2QLayout`](qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout)
    setting.
* **Return type:**
  None

#### SEE ALSO
The documentation of [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) for more detailed explanations of the
arguments.

* **Raises:**
  [**NotImplementedError**](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError) – when `in_node` acts on fermionic modes that are spread across
      multiple [`FermionicRegister`](qiskit_fermions.circuit.FermionicRegister.html.md#qiskit_fermions.circuit.FermionicRegister) instances.
* **Parameters:**
  * **in_node** ([*DAGOpNode*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode))
  * **out_dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit))
  * **f2q_layout** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *[*[*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *,* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *]*)
* **Return type:**
  None
