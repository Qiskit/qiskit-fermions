# FermionicCircuit

### *class* FermionicCircuit(num_modes)

A wrapper around [`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit) for expressing fermionic circuits.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

This class maintains a reduced API compared to the full API of the underlying
[`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit). This is done to avoid exposing (among other methods)
the ability to apply qubit-based gates onto a fermionic circuit, which would not be a
well-defined operation in the general case.

[`barrier()`](#qiskit_fermions.circuit.FermionicCircuit.barrier) is re-exposed because it is an exception to that rule: a barrier acts on modes
without interpreting them and carries no unitary content, so it is as well-defined on a fermionic
register as it is on a qubit one.

Measurements are likewise available, via [`measure()`](#qiskit_fermions.circuit.FermionicCircuit.measure) and [`measure_all()`](#qiskit_fermions.circuit.FermionicCircuit.measure_all). They are the one
non-unitary operation this circuit carries, so they are reached through those methods rather than
through [`append()`](#qiskit_fermions.circuit.FermionicCircuit.append), which stays restricted to unitary gates. Whether the
resulting bit is meaningful depends on the fermion-to-qubit encoding chosen later, during
transpilation, rather than on this circuit (see [`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure)).

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

#### add_register(\*cregs)

Adds classical registers to this circuit.

Only classical registers can be added: the fermionic mode register is fixed at construction
and is available as [`register`](#qiskit_fermions.circuit.FermionicCircuit.register).

* **Parameters:**
  **cregs** ([*ClassicalRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.ClassicalRegister)) – the classical registers to add.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if any of the provided registers is not a
      [`ClassicalRegister`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.ClassicalRegister).
* **Return type:**
  None

#### append(gate, fargs, cargs=None, , copy=True)

Appends a [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) to this circuit.

* **Parameters:**
  * **gate** ([*FermionicGate*](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate)) – the fermionic gate to apply.
  * **fargs** ([*Qubit*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Qubit) *|* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *|* [*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*slice*](https://docs.python.org/3/builtins/functions.html#slice) *|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence) *[*[*Qubit*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Qubit) *|* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – the fermionic modes on which this gate acts.
  * **cargs** (*None*) – 

    the classical bits on which this gate acts.

    #### WARNING
    No gates of this kind are currently supported. A [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) is unitary
    and so takes no classical bits; use [`measure()`](#qiskit_fermions.circuit.FermionicCircuit.measure) to add the one instruction that
    does.
  * **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – forwarded to [`append()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#append).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the provided `gate` is not an instance of [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate).
* **Return type:**
  None

#### barrier(\*fargs, label=None)

Re-exposes [`barrier()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#barrier).

A barrier carries no unitary effect. It marks a synchronization point that the optimization
passes will not fuse gates across, which is how you keep a pass such as
[`MergeOrbitalRotations`](qiskit_fermions.transpiler.passes.MergeOrbitalRotations.html.md#qiskit_fermions.transpiler.passes.MergeOrbitalRotations) or [`MergeSlaterDeterminantPreparation`](qiskit_fermions.transpiler.passes.MergeSlaterDeterminantPreparation.html.md#qiskit_fermions.transpiler.passes.MergeSlaterDeterminantPreparation) from combining
the gates on either side of it.

* **Parameters:**
  * **fargs** ([*Qubit*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Qubit) *|* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *|* [*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*slice*](https://docs.python.org/3/builtins/functions.html#slice) *|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence) *[*[*Qubit*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Qubit) *|* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – the fermionic modes on which to place the barrier. When omitted, the barrier spans
    every mode of this circuit.
  * **label** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *|* *None*) – the string label of the barrier.
* **Return type:**
  None

#### compose(other, fargs=None, , front=False, inplace=False)

Composes another [`FermionicCircuit`](#qiskit_fermions.circuit.FermionicCircuit) onto this one.

The instructions of `other` are inlined into this circuit, rather than wrapped into a single
gate. This mirrors [`compose()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#compose) with its default
`wrap=False`. The `wrap=True` counterpart is deliberately not offered: a dedicated
encapsulation mechanism is planned instead, so adding a second way to box up a block of
instructions here would leave two patterns solving the same problem.

* **Parameters:**
  * **other** ([*FermionicCircuit*](#qiskit_fermions.circuit.FermionicCircuit)) – the circuit to compose onto this one. It may act on fewer modes than this circuit,
    in which case `fargs` selects the modes it is placed on.
  * **fargs** ([*Qubit*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Qubit) *|* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *|* [*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*slice*](https://docs.python.org/3/builtins/functions.html#slice) *|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence) *[*[*Qubit*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Qubit) *|* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*  *|* *None*) – the fermionic modes of this circuit onto which `other` is placed. Defaults to all
    of this circuit’s modes, in order.
  * **front** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to compose `other` before rather than after this circuit’s instructions.
  * **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to modify this circuit rather than return a new one.
* **Returns:**
  The composed circuit, or `None` if `inplace` is set.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `other` is not an instance of [`FermionicCircuit`](#qiskit_fermions.circuit.FermionicCircuit). Only fermionic
        circuits are accepted because the plain
        [`compose()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#compose) would happily inline
        qubit-based instructions, bypassing the type guard that [`append()`](#qiskit_fermions.circuit.FermionicCircuit.append) enforces.
  * **CircuitError** – if `other` acts on more modes than this circuit, or `fargs` does not
        match its width.
* **Return type:**
  [*FermionicCircuit*](#qiskit_fermions.circuit.FermionicCircuit) | None

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

#### measure(fargs, cargs)

Measures the given fermionic modes into the given classical bits.

This appends one [`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure) per mode. The classical bits must already exist;
add them with [`add_register()`](#qiskit_fermions.circuit.FermionicCircuit.add_register), or use [`measure_all()`](#qiskit_fermions.circuit.FermionicCircuit.measure_all), which creates a register for
you.

The pairing of modes to classical bits is yours, and it is preserved through transpilation:
the mode a measurement acts on is permuted by a pass such as [`RelabelModes`](qiskit_fermions.transpiler.passes.RelabelModes.html.md#qiskit_fermions.transpiler.passes.RelabelModes), but the
classical bit it writes into is not.

* **Parameters:**
  * **fargs** ([*FermionicSpecifier*](qiskit_fermions.circuit.FermionicSpecifier.html.md#qiskit_fermions.circuit.FermionicSpecifier)) – the fermionic modes to measure.
  * **cargs** (*ClbitSpecifier*) – the classical bits to write the outcomes into.
* **Raises:**
  **CircuitError** – if the number of modes and classical bits differ, or if a specifier does not
      resolve against this circuit.
* **Return type:**
  None

#### measure_all(, add_bits=True, inplace=True)

Measures every mode of this circuit, in order, into one classical bit each.

Mode `i` is measured into classical bit `i`, so a sampled bitstring is indexed by mode.

#### NOTE
Unlike [`measure_all()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#measure_all), this inserts no
[`barrier()`](#qiskit_fermions.circuit.FermionicCircuit.barrier) before the measurements. Call [`barrier()`](#qiskit_fermions.circuit.FermionicCircuit.barrier) yourself beforehand if you
want one there.

* **Parameters:**
  * **add_bits** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to add a new `meas` [`ClassicalRegister`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.ClassicalRegister) sized to
    this circuit’s modes. When this is `False`, the existing classical bits are used
    instead, and calling this method again simply measures into them a second time.
  * **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to modify this circuit rather than return a new one.
* **Returns:**
  The measured circuit, or `None` if `inplace` is set.
* **Raises:**
  * **CircuitError** – if `add_bits` is set and this circuit already carries a register named
        `meas` (which is what a second `measure_all()` call runs into). Pass
        `add_bits=False` to measure into the existing bits instead.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `add_bits` is not set and this circuit has fewer classical bits than
        modes.
* **Return type:**
  [*FermionicCircuit*](#qiskit_fermions.circuit.FermionicCircuit) | None

#### repeat(reps, , insert_barriers=False)

Repeats this circuit `reps` times.

The repetitions are flattened: the returned circuit holds this circuit’s instructions
`reps` times in sequence, rather than one gate wrapping the circuit as
[`repeat()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit#repeat) produces. As a consequence
[`count_ops()`](#qiskit_fermions.circuit.FermionicCircuit.count_ops) scales with `reps`, and [`decompose()`](#qiskit_fermions.circuit.FermionicCircuit.decompose) decomposes the repeated body
rather than peeling off a repetition.

Flattening also leaves the gates that meet at a repetition boundary adjacent, so the
optimization passes can fuse them. Set `insert_barriers` to keep the repetitions apart
instead, which places a [`barrier()`](#qiskit_fermions.circuit.FermionicCircuit.barrier) that those passes do not fuse across.

#### NOTE
This is unrelated to the `reps` argument of [`FermionicSuzukiTrotter`](qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter), which
subdivides a single [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) into more, shorter factors. Repeating a circuit
holding one [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) merely multiplies its total evolution time.

#### SEE ALSO
[Work with fermionic circuits](../guides/circuit.html.md#fermionic-circuit-explanation) works through a repeated Trotter step, including the
boundary fusion and how to prevent it.

* **Parameters:**
  * **reps** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – how often this circuit should be repeated. Repeating it zero times yields an empty
    circuit, which is the identity.
  * **insert_barriers** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to place a [`barrier()`](#qiskit_fermions.circuit.FermionicCircuit.barrier) between consecutive repetitions. No
    barrier is placed after the final repetition, so the returned circuit ends on the same
    instruction it would without this argument.
* **Returns:**
  A new circuit containing `reps` repetitions of this circuit.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `reps` is negative.
* **Return type:**
  [*FermionicCircuit*](#qiskit_fermions.circuit.FermionicCircuit)

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
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if a circuit instruction other than a
        `Barrier` does not implement ffsim’s
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

A `Barrier` is skipped: it carries no unitary effect, so it
constrains transpilation without contributing to the state vector.

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
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if a circuit instruction other than a
        `Barrier` does not implement ffsim’s
        [`ffsim.SupportsApplyUnitary`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsApplyUnitary.html#ffsim.SupportsApplyUnitary) protocol.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if a circuit instruction declines to apply its unitary for the given
        `norb` and `nelec`; or if an instruction implementing only the plain
        `_apply_unitary_` protocol is placed on a non-identity mode subset.
* **Return type:**
  [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)
