# F2FSynthesisPlugin

### *class* F2FSynthesisPlugin(\*args, \*\*kwargs)

The protocol for plugins to the [`F2FSynthesis`](qiskit_fermions.transpiler.passes.F2FSynthesis.html.md#qiskit_fermions.transpiler.passes.F2FSynthesis) transpiler pass.

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A plugin rewrites a gate into a [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) of *smaller* [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate)
instances. Both sides of that rewrite live in fermionic space, so no encoding is involved: how
the smaller gates are then mapped to qubits is left to the fermion-to-qubit stage (see
[`F2QSynthesis`](qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis)). It is the counterpart of [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) one step earlier.

The gates a plugin emits should be marked [`atomic`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.atomic), and an
[`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) among them should carry an operator with a single group, so that the
fermion-to-qubit stage maps every one of them whole.

#### IMPORTANT
What the circuit has to be equivalent to is specified by the *gate* a plugin is registered
for, not by this protocol, and it may differ between gates: a circuit that implements the same
unitary is the common case, but a gate may settle for less (for example, the same state when
prepared from the vacuum). Consult the documentation of the gate whose method you implement,
and document any such caveat there.

#### IMPORTANT
This fermion-to-fermion step is **optional**. A gate can be handed to the fermion-to-qubit
stage whole, where a plugin of that stage registered for it can map it directly.

### Methods

#### synthesize(gate)

Synthesizes the provided gate.

* **Parameters:**
  **gate** ([*FermionicGate*](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate)) – the gate to synthesize, which is of the one type that this plugin is registered for
  (the first part of its entry-point name).
* **Returns:**
  A [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) on `gate.num_modes` modes implementing the gate, in the
  sense that the documentation of the gate defines.
* **Return type:**
  [FermionicCircuit](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit)
