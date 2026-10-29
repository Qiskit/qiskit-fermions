# UCJ

### *class* UCJ(ucj_op)

Implements the (local) unitary cluster Jastrow ((L)UCJ) ansatz.

Bases: [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate)

A unitary cluster Jastrow operator has the form

$$
\left(\prod_{k=1}^{L} \mathcal{U}_k\, e^{i \mathcal{J}_k}\, \mathcal{U}_k^\dagger\right)
\mathcal{U}_\text{final}
$$

where each $\mathcal{U}_k$ is an [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation), each $\mathcal{J}_k$ is a
diagonal Coulomb operator

$$
\mathcal{J} = \frac12 \sum_{ij,\sigma\tau} \mathbf{J}^{\sigma\tau}_{ij}\,
n_{i\sigma}\, n_{j\tau},
$$

and $\mathcal{U}_\text{final}$ is an optional final orbital rotation. The number of terms
$L$ is the number of ansatz repetitions.

The operator itself is built by [ffsim](https://qiskit-community.github.io/ffsim/), and this
gate turns it into a [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit). That division of labor is deliberate: ffsim owns
the ansatz math (double factorization of $t_2$ amplitudes, the compressed variant, the
parameter-vector packing that a variational optimizer drives), while this gate expresses the
result as fermionic modes so that the transpiler can lower it through *any* fermion-to-qubit
encoding. Passing an ffsim operator through unchanged is what keeps the two consistent; see the
[ffsim relationship guide](../guides/ffsim.html.md#ffsim-relationship-explanation).

Accepts any of ffsim’s three UCJ operators, whose type fixes the spin variant and the number of
modes this gate acts on:

- [`UCJOpSpinBalanced`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCJOpSpinBalanced.html#ffsim.UCJOpSpinBalanced) acts on `2 * norb` block-spin modes, with one
  orbital rotation shared by both spin sectors and `[alpha-alpha, alpha-beta]` diagonal Coulomb
  matrices (beta-beta reuses alpha-alpha, beta-alpha reuses alpha-beta).
- [`UCJOpSpinUnbalanced`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCJOpSpinUnbalanced.html#ffsim.UCJOpSpinUnbalanced) acts on `2 * norb` block-spin modes, with
  independent `[alpha, beta]` rotations and `[alpha-alpha, alpha-beta, beta-beta]` matrices.
- [`UCJOpSpinless`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCJOpSpinless.html#ffsim.UCJOpSpinless) acts on `norb` spinless modes.

#### NOTE
ffsim’s [`UCJOpSpinless`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCJOpSpinless.html#ffsim.UCJOpSpinless) is also valid on a spinful sector, where its
tensors act on both spin sectors with no cross-spin term. A gate has to fix its width when it
is constructed, so this gate always reads that type as a single `norb`-mode register. Build
the two-register reading as an [`UCJOpSpinBalanced`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCJOpSpinBalanced.html#ffsim.UCJOpSpinBalanced) whose alpha-beta
block is zero, which is the same operator.

#### NOTE
ffsim does not support Windows (through its unconditional PySCF dependency), so this gate
requires the `ffsim` extra (`pip install "qiskit-fermions[ffsim]"`) and is unavailable
there. Use [WSL](https://learn.microsoft.com/windows/wsl/) on Windows.

#### CAUTION
This is an early development prototype. Beware of changes to its interface without warning
during the pre-release development of this package.

```pycon
>>> import ffsim
>>> from qiskit_fermions.circuit.library import UCJ
>>> ucj_op = ffsim.random.random_ucj_op_spin_balanced(3, n_reps=1, seed=1234)
>>> gate = UCJ(ucj_op)
>>> gate.norb, gate.num_modes, gate.n_reps
(3, 6, 1)
```

Initializing an instance of this gate can be done with the argument listed below.

* **Parameters:**
  **ucj_op** (*Any*) – the ffsim UCJ operator to build the circuit from, one of
  [`UCJOpSpinBalanced`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCJOpSpinBalanced.html#ffsim.UCJOpSpinBalanced),
  [`UCJOpSpinUnbalanced`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCJOpSpinUnbalanced.html#ffsim.UCJOpSpinUnbalanced) or
  [`UCJOpSpinless`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCJOpSpinless.html#ffsim.UCJOpSpinless). Its type determines the spin variant and the
  number of modes this gate acts on (see the class docstring).
* **Raises:**
  * **MissingOptionalLibraryError** – if `ffsim` is not installed.
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `ucj_op` is not one of ffsim’s three UCJ operator types.

### Attributes

#### n_reps

The number of ansatz repetitions.

#### norb

The number of spatial orbitals.

#### ucj_op

The ffsim UCJ operator this gate builds its circuit from.

### Protocol Methods

#### \_apply_unitary_placed_(vec, norb, nelec, copy, freg_indices)

Applies the ansatz after placing its modes onto the vector’s global modes.

This builds the gate’s definition (the per-repetition orbital rotations and diagonal Coulomb
evolutions) and applies it to `vec`, with the definition circuit placed onto the global
modes `freg_indices` (each of its instructions is relabeled onto the corresponding absolute
modes). See `_define()` for the exact gate sequence.

* **Parameters:**
  * **vec** ([*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) – the state vector to act on.
  * **norb** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of spatial orbitals of the *global* state vector.
  * **nelec** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – either a single integer for a spinless system, or a pair of integers storing the
    numbers of spin alpha and spin beta fermions.
  * **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to copy the vector before operating on it.
  * **freg_indices** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – the absolute (global) mode indices that this gate’s local modes map onto.
* **Returns:**
  The transformed vector.
* **Return type:**
  [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)
