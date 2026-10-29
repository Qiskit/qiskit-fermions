# PrepareSlaterDeterminant

### *class* PrepareSlaterDeterminant(occupation, rotation_unitary)

Prepares a Slater determinant from an occupation and an orbital rotation.

Bases: [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate)

This gate is the composition of an [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) reference occupation followed by an
[`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation): it declares that the modes it acts on start in the determinant defined
by `occupation` and are then rotated by `rotation_unitary`. Because the initial occupation is
known, transpiling this gate can use the rectangular [`givens_decomposition_slater()`](qiskit_fermions.linalg.givens_decomposition_slater.html.md#qiskit_fermions.linalg.givens_decomposition_slater) for a
reduced-gate-count synthesis (see [`GivensDecompositionSlaterDeterminantSynthesis`](qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionSlaterDeterminantSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionSlaterDeterminantSynthesis)) rather
than the full square decomposition an [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) alone would require.

Following the [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) convention $a^\dagger_i \mapsto \sum_j U_{ji}
a^\dagger_j$, rotating the occupied modes maps them onto the corresponding columns of
`rotation_unitary`; those columns span the occupied space of the prepared Slater determinant.

### Simulation semantics

Under simulation this gate is **validate-then-rotate**, not a state producer: given an incoming
state vector it validates that the vector is confined to the subspace `occupation` defines (via
[`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes)) and then applies `rotation_unitary` (via [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation)).
It therefore requires a real reference state vector, prepared externally (e.g. via
[`ffsim.slater_determinant()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.slater_determinant.html#ffsim.slater_determinant)). The *producer* behavior – emitting the gates that set the
reference occupation – lives in the synthesis plugin, not the simulation path.

Applying (rather than dropping) the rotation is what guarantees that merging an
[`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) and an [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) into this single gate leaves the
simulated final state unchanged.

#### CAUTION
This is an early development prototype. Beware of changes to its interface without warning
during the pre-release development of this package.

#### SEE ALSO
[`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes), [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation), and
[`givens_decomposition_slater()`](qiskit_fermions.linalg.givens_decomposition_slater.html.md#qiskit_fermions.linalg.givens_decomposition_slater).

Initializing an instance of this gate can be done with the arguments listed below.

* **Parameters:**
  * **occupation** (*Sequence* *[*[*bool*](https://docs.python.org/3/builtins/functions.html#bool) *]*) – a sequence of booleans indicating the reference occupation for each mode this
    gate acts on.
  * **rotation_unitary** (*np.ndarray*) – the $n \times n$ unitary matrix $U$ defining the orbital
    rotation via $a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j$, where $n$ is
    the number of modes (`len(occupation)`). It must be square and unitary; this is the
    caller’s responsibility and is not verified.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `rotation_unitary` is not a square matrix whose dimension matches the
      length of `occupation`.

### Attributes

#### occupation

The reference occupation (one boolean per mode) the rotation is applied to.

#### rotation_unitary

The unitary matrix representing the orbital rotation coefficients.

### Protocol Methods

#### \_apply_unitary_placed_(vec, norb, nelec, copy, freg_indices)

Validates the reference occupation and applies the rotation, placing modes onto `vec`.

Delegates to the gate’s definition (`_build_definition()`): the leading
[`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) validates that `vec` is confined to the subspace `occupation`
defines and the following [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) rotates it. See those gates’
`_apply_unitary_placed_` for the full semantics (subspace confinement and the rejection of
spin-mixing rotations).

* **Parameters:**
  * **vec** ([*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) – the reference state vector to validate and rotate.
  * **norb** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of spatial orbitals of the *global* state vector.
  * **nelec** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – either a single integer for a spinless system, or a pair of integers storing the
    numbers of spin alpha and spin beta fermions.
  * **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to copy the vector before operating on it.
  * **freg_indices** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – the absolute (global) mode indices that this gate’s local modes map onto.
* **Returns:**
  The transformed vector.
* **Return type:**
  [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)
