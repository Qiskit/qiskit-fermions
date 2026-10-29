# OrbitalRotation

### *class* OrbitalRotation(rotation_unitary)

Implements an orbital rotation.

Bases: [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate)

Given an $n \times n$ unitary matrix $U$ (`rotation_unitary`), this gate
implements the single-particle basis change that maps the creation operators as

$$
a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j,
$$

which is equivalent to applying the many-body unitary

$$
\exp\left(\sum_{ij} \log(U)_{ij} \, a^\dagger_i a_j\right).
$$

The number of fermionic modes the gate acts on is the dimension $n$ of
`rotation_unitary`.

Initializing an instance of this gate can be done with the arguments listed below.

* **Parameters:**
  **rotation_unitary** (*np.ndarray*) – the $n \times n$ unitary matrix $U$ defining the orbital
  rotation via $a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j$. It must be
  square and unitary; this is the caller’s responsibility and is not verified.

### Attributes

#### rotation_unitary

The unitary matrix representing the orbital rotation coefficients.

### Methods

#### *classmethod* from_t1_amplitudes(t1)

Constructs an orbital rotation from $t_1$ (singles) amplitudes.

The rotation is the unitary $\exp(t_1 - t_1^\dagger)$, where the
$n_\text{occ} \times n_\text{virt}$ amplitude matrix $t_1$ is embedded into the
anti-Hermitian generator over all $n = n_\text{occ} + n_\text{virt}$ orbitals

$$
G = \begin{pmatrix} 0 & -t_1^* \\ t_1^\top & 0 \end{pmatrix},
$$

with the occupied orbitals ordered before the virtual ones. This is the single-excitation
orbital rotation entering the (L)UCJ ansatz when it is initialized from coupled-cluster
amplitudes; see [`UCJ`](qiskit_fermions.circuit.library.UCJ.html.md#qiskit_fermions.circuit.library.UCJ).

* **Parameters:**
  **t1** ([*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) – the $t_1$ amplitudes of shape `(nocc, nvrt)`, where `nocc` is the number
  of occupied orbitals and `nvrt` is the number of virtual orbitals.
* **Returns:**
  An [`OrbitalRotation`](#qiskit_fermions.circuit.library.OrbitalRotation) acting on $n = n_\text{occ} + n_\text{virt}$ modes,
  whose `rotation_unitary` is $\exp(t_1 - t_1^\dagger)$.
* **Return type:**
  [*OrbitalRotation*](#qiskit_fermions.circuit.library.OrbitalRotation)

### Protocol Methods

#### \_apply_unitary_placed_(vec, norb, nelec, copy, freg_indices)

Applies the orbital rotation after placing it onto the vector’s global modes.

The gate’s local [`rotation_unitary`](#qiskit_fermions.circuit.library.OrbitalRotation.rotation_unitary) (an $n \times n$ matrix acting on the gate’s
`num_modes` modes) is first embedded into the full register: an identity matrix of the
state vector’s mode count with `rotation_unitary` written into the rows/columns picked out
by `freg_indices`.

In the spinful case the embedded matrix must be block-diagonal across the alpha/beta split.
A rotation with nonzero alpha/beta off-diagonal blocks mixes the spin sectors, which does not
conserve the individual alpha/beta electron counts and hence maps amplitude out of the fixed
`(n_alpha, n_beta)` sector – an operation the fixed-sector state vector cannot represent.
Such a rotation is rejected with a [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError).

The embedded matrix is then applied via [`ffsim.apply_orbital_rotation()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.apply_orbital_rotation.html#ffsim.apply_orbital_rotation)’s Givens-rotation
kernel. Under the spinful block-spin convention (modes `0..norb` are alpha orbitals, modes
`norb..2*norb` are beta orbitals) the two diagonal blocks are the per-spin rotations passed
to ffsim as `(mat_a, mat_b)`.

* **Parameters:**
  * **vec** ([*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) – the state vector to act on.
  * **norb** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of spatial orbitals of the *global* state vector.
  * **nelec** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – either a single integer for a spinless system, or a pair of integers storing the
    numbers of spin alpha and spin beta fermions. An integer selects the spinless mode
    interpretation (the `norb` modes are orbitals); a pair selects the spinful
    `(orb, spin)` block-spin interpretation of the `2 * norb` modes.
  * **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to copy the vector before operating on it.
  * **freg_indices** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – the absolute (global) mode indices that this gate’s local modes map onto.
    The rotation is embedded onto these global modes before being applied.
* **Returns:**
  The transformed vector.
* **Raises:**
  * **MissingOptionalLibraryError** – if ffsim is not installed.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `nelec` is a spinful pair and the (placed) rotation mixes the alpha and
        beta spin sectors.
* **Return type:**
  [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)
