# SupportsApplyUnitaryPlaced

### *class* SupportsApplyUnitaryPlaced(\*args, \*\*kwargs)

A package-specific extension of [`ffsim.SupportsApplyUnitary`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsApplyUnitary.html#ffsim.SupportsApplyUnitary).

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Where ffsim’s protocol applies an object to a whole state vector, this one adds the placement of
the object’s local modes onto the global ones it acts on. It has no ffsim equivalent:
[`_apply_unitary_placed_()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit._apply_unitary_placed_) is the only caller, and it dispatches to it
directly via `getattr` duck-typing. It exists here purely for typing and documentation
purposes, since every concrete [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) (as well as [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit)
itself) implements this method.

See [`FermionicCircuit._apply_unitary_placed_()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit._apply_unitary_placed_) for the full semantics, including how a
plain `_apply_unitary_` implementation (with no mode-placement argument) is honored only on the
identity placement.

### Protocol Methods

#### \_apply_unitary_placed_(vec, norb, nelec, copy, freg_indices)

Applies a unitary transformation to a state vector, after placing local modes onto global ones.

* **Parameters:**
  * **vec** (*np.ndarray*) – the state vector to apply the unitary transformation to.
  * **norb** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of spatial orbitals of the *global* state vector.
  * **nelec** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – either a single integer representing the number of fermions for a spinless
    system, or a pair of integers storing the numbers of spin alpha and spin beta
    fermions.
  * **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to copy the vector before operating on it.
  * **freg_indices** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – the absolute (global) mode indices that this object’s local modes map onto.
* **Returns:**
  The transformed vector.
* **Return type:**
  np.ndarray
