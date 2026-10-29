# FermionicCircuit

### *class* FermionicCircuit(num_modes)

A wrapper around [`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit) for expressing fermionic circuits.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

This class maintains a reduced API compared to the full API of the underlying
[`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit). This is done to avoid exposing (amongst other methods)
the ability to apply qubit-based gates onto a fermionic circuit, which would not be a
well-defined operation in the general case.

Initializing a circuit instance can be done with the arguments listed below.

* **Parameters:**
  **num_modes** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of fermionic modes on which this circuit acts.

### Attributes

#### metadata

Re-exposes [`metadata`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#metadata).

#### modes

The fermionic mode `bits` that this circuit acts upon.

#### register *: [FermionicRegister](qiskit_fermions.circuit.FermionicRegister.html.md#qiskit_fermions.circuit.FermionicRegister)*

The inner circuit’s [`FermionicRegister`](qiskit_fermions.circuit.FermionicRegister.html.md#qiskit_fermions.circuit.FermionicRegister).

### Methods

#### append(gate, fargs, cargs=None, , copy=True)

Appends a [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) to this circuit.

* **Parameters:**
  * **gate** ([*FermionicGate*](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate)) – the fermionic gate to apply.
  * **fargs** ([*Qubit*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Qubit) *|* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *|* [*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*slice*](https://docs.python.org/3/builtins/functions.html#slice) *|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence) *[*[*Qubit*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Qubit) *|* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – the fermionic modes on which this gate acts.
  * **cargs** (*None*) – 

    the classical bits on which this gate acts.

    #### WARNING
    No gates of this kind are currently supported.
  * **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – forwarded to [`append()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#append).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the provided `gate` is not an instance of [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate).
* **Return type:**
  None

#### count_ops()

Re-exposes [`count_ops()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#count_ops).

* **Return type:**
  [*OrderedDict*](https://docs.python.org/3/library/collections.html#collections.OrderedDict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [int](https://docs.python.org/3/builtins/functions.html#int)]

#### decompose(gates_to_decompose=None, reps=1)

Re-exposes [`decompose()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#decompose).

* **Parameters:**
  * **gates_to_decompose** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *|* [*type*](https://docs.python.org/3/builtins/functions.html#type) *[*[*Instruction*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Instruction) *]*  *|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence) *[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *|* [*type*](https://docs.python.org/3/builtins/functions.html#type) *[*[*Instruction*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Instruction) *]* *]*  *|* *None*)
  * **reps** ([*int*](https://docs.python.org/3/builtins/functions.html#int))
* **Return type:**
  [*FermionicCircuit*](#qiskit_fermions.circuit.FermionicCircuit)

#### draw(\*args, \*\*kwargs)

Directly exposes the inner circuit’s [`draw()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#draw) method.

* **Return type:**
  [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)

### Protocol Methods

#### \_apply_unitary_(vec, norb, nelec, copy)

Applies this circuit to an ffsim state vector, implementing ffsim’s protocol.

This walks the circuit in topological order and applies each instruction’s unitary effect to
the state vector via ffsim’s [`ffsim.SupportsApplyUnitary`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsApplyUnitary.html#ffsim.SupportsApplyUnitary) protocol. Each instruction
acting on a subset of the register has its fermionic modes relabeled to their absolute
(global) indices before being applied.

See [`_apply_unitary_placed_()`](#qiskit_fermions.circuit.FermionicCircuit._apply_unitary_placed_) for the details; this method assumes the circuit’s modes
are the vector’s modes `0..num_modes` (i.e. an identity mode placement).

* **Parameters:**
  * **vec** ([*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) – the state vector to apply this circuit to. An empty circuit returns it unchanged.
  * **norb** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of spatial orbitals.
  * **nelec** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – either a single integer representing the number of fermions for a spinless system,
    or a pair of integers storing the numbers of spin alpha and spin beta fermions.
  * **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to copy the vector before operating on it.
* **Returns:**
  The transformed vector.
* **Raises:**
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if a circuit instruction does not implement ffsim’s
        [`ffsim.SupportsApplyUnitary`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsApplyUnitary.html#ffsim.SupportsApplyUnitary) protocol.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if a circuit instruction declines to apply its unitary for the given
        `norb` and `nelec`; or if an instruction implementing only the plain
        `_apply_unitary_` protocol is placed on a non-identity mode subset.
* **Return type:**
  [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

#### \_apply_unitary_placed_(vec, norb, nelec, copy, freg_indices)

Applies this circuit after placing its modes onto the vector’s global modes.

This walks the circuit in topological order and applies each instruction’s unitary effect to
the state vector via ffsim’s [`ffsim.SupportsApplyUnitary`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsApplyUnitary.html#ffsim.SupportsApplyUnitary) protocol. Each
instruction’s own modes are first mapped through this circuit’s placement: a circuit-local
mode `m` maps to the
global mode `freg_indices[m]`, so a sub-instruction acting on circuit-local modes
`[m0, m1, ...]` is applied on the global modes `[freg_indices[m0], freg_indices[m1], ...]`.
With the identity placement `freg_indices == 0..num_modes` this is exactly
[`_apply_unitary_()`](#qiskit_fermions.circuit.FermionicCircuit._apply_unitary_); a subset placement lets this circuit act as the definition of a gate
placed on a subset of a larger register (e.g. [`UCJ`](qiskit_fermions.circuit.library.UCJ.html.md#qiskit_fermions.circuit.library.UCJ)). This placement-aware extension
is documented as [`SupportsApplyUnitaryPlaced`](qiskit_fermions.protocols.SupportsApplyUnitaryPlaced.html.md#qiskit_fermions.protocols.SupportsApplyUnitaryPlaced) – a package-specific protocol with no
ffsim equivalent.

An instruction is placed onto its absolute modes only if it implements the placement-aware
`_apply_unitary_placed_` extension ([`SupportsApplyUnitaryPlaced`](qiskit_fermions.protocols.SupportsApplyUnitaryPlaced.html.md#qiskit_fermions.protocols.SupportsApplyUnitaryPlaced)). An instruction
implementing only ffsim’s plain `_apply_unitary_` – which
has no mode argument and therefore acts on modes `0..k` of the vector – can only be honored
when its placement is the identity `[0, 1, ..., k-1]`; on any other subset the placement
cannot be expressed and the instruction is rejected rather than silently applied on the wrong
modes.

* **Parameters:**
  * **vec** ([*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) – the state vector to apply this circuit to. An empty circuit returns it unchanged.
  * **norb** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of spatial orbitals of the *global* state vector.
  * **nelec** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – either a single integer representing the number of fermions for a spinless system,
    or a pair of integers storing the numbers of spin alpha and spin beta fermions.
  * **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to copy the vector before operating on it.
  * **freg_indices** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – the absolute (global) mode indices that this circuit’s modes map onto.
* **Returns:**
  The transformed vector.
* **Raises:**
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if a circuit instruction does not implement ffsim’s
        [`ffsim.SupportsApplyUnitary`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsApplyUnitary.html#ffsim.SupportsApplyUnitary) protocol.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if a circuit instruction declines to apply its unitary for the given
        `norb` and `nelec`; or if an instruction implementing only the plain
        `_apply_unitary_` protocol is placed on a non-identity mode subset.
* **Return type:**
  [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)
