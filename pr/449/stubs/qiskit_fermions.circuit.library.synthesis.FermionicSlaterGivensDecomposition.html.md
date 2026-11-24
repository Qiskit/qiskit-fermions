# FermionicSlaterGivensDecomposition

### *class* FermionicSlaterGivensDecomposition(, atol=1e-12)

The decomposition of a Slater determinant preparation into Givens rotations.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

This exploits the known reference occupation to synthesize the gate with the rectangular
[`givens_decomposition_slater()`](qiskit_fermions.linalg.givens_decomposition_slater.html.md#qiskit_fermions.linalg.givens_decomposition_slater): only the $m$ occupied orbitals
of the rotation need be realized, so at most $m (n - m)$ rotations are emitted (versus the
$n (n - 1) / 2$ rotations plus $n$ phases that [`FermionicGivensDecomposition`](qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition.html.md#qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition)
needs for the full [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation)). The reduced decomposition carries no diagonal
phases (a global phase and any rotation within the occupied space leave the prepared Slater
determinant unchanged), so the state is prepared correct up to a global phase.

The result is an [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) that sets the reference, followed by one atomic
[`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) per rotation (see [`FermionicGivensDecomposition`](qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition.html.md#qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition) for how a rotation
becomes one). Being fermionic, the circuit needs no knowledge of how the modes are encoded.

#### NOTE
[`givens_decomposition_slater()`](qiskit_fermions.linalg.givens_decomposition_slater.html.md#qiskit_fermions.linalg.givens_decomposition_slater) is defined against a fixed
*leading*-$m$ reference (the first $m$ modes occupied). The [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes)
therefore seeds the leading modes of each decomposed block, **not** the modes pointed to by
`occupation` – the rotations then transport the occupation onto the physical target
orbitals. The positions of `occupation`’s occupied modes enter only through which columns of
`rotation_unitary` are selected (`rotation_unitary[:, occupied]`); their *count*
$m$ determines the reference. The final prepared state is correct for any occupation,
contiguous or not. This is also why the circuit is only equivalent to the gate as a state
preparation from the vacuum.

#### NOTE
The occupied orbitals are partitioned into disjoint-support **blocks**, each synthesized on
its own contiguous mode window. When they split into several clusters separated by empty (or
merely occupied) modes, this drops the bare swaps (rotations with `c = 0`) that a single
leading-reference sweep would otherwise spend on the gaps between clusters, and an orbital
that is a basis vector needs no rotation at all. A single cluster spanning all modes is one
block and reproduces the plain leading-reference sweep exactly. Every rotation is between
adjacent modes, so each is a hop between adjacent qubits under Jordan-Wigner, which routes
cheaply on a constrained coupling map.

Initializing an instance of this synthesis method can be done with the arguments below.

* **Parameters:**
  **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – the absolute tolerance below which factors and entries are dropped. This is
  applied to the entries of the generator of a rotation, see
  [`FermionicGivensDecomposition`](qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition.html.md#qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition).

### Attributes

#### atol

The magnitude below which entries of a generator are dropped.

### Methods

#### synthesize(gate)

Synthesizes the gate into the reference and the mesh of atomic [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates.

See the class documentation for the decomposition this implements.

* **Parameters:**
  **gate** ([*PrepareSlaterDeterminant*](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant)) – the gate to synthesize.
* **Returns:**
  A [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) holding an [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) of the reference
  (absent if there is nothing to occupy), followed by one [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate per
  non-trivial rotation, in the order in which they are applied.
* **Return type:**
  [FermionicCircuit](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit)
