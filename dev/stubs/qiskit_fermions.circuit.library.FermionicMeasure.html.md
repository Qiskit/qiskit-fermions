# FermionicMeasure

### *class* FermionicMeasure(\*args, \_force_mutable=False, \*\*kwargs)

A measurement of a single fermionic mode, writing its outcome into one classical bit.

Bases: [`FermionicInstruction`](qiskit_fermions.circuit.FermionicInstruction.html.md#qiskit_fermions.circuit.FermionicInstruction), [`Measure`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Measure)

This is the fermionic counterpart of [`Measure`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Measure), and the one
instruction on a [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) that is not a [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate): a measurement is
not unitary, and [`Gate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Gate) admits no classical bits. Place it with
[`FermionicCircuit.measure()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.measure) or [`FermionicCircuit.measure_all()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.measure_all) rather than constructing
it directly.

It subclasses [`Measure`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Measure) so that the circuit drawers, which dispatch
on `isinstance(op, Measure)`, render it with the usual measurement symbol rather than as a
generic boxed instruction.

#### IMPORTANT
What the measured bit *means* is decided by the fermion-to-qubit encoding, not by this
instruction. Under an occupation-basis encoding such as Jordan-Wigner, mode $i$ is carried
by a single qubit and the bit is that mode’s occupation, which is the reading
[`TrivialOccupationFermionicMeasureSynthesis`](qiskit_fermions.transpiler.passes.synthesis.TrivialOccupationFermionicMeasureSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.TrivialOccupationFermionicMeasureSynthesis) implements. A
[`F2QLayout`](qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout) relates whole registers whose two sides may differ
in size, so in general a mode has no single qubit to measure; an encoding for which that is the
case needs its own synthesis plugin, and this instruction carries no encoding assumption of its
own.

#### SEE ALSO
[`TrivialOccupationFermionicMeasureSynthesis`](qiskit_fermions.transpiler.passes.synthesis.TrivialOccupationFermionicMeasureSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.TrivialOccupationFermionicMeasureSynthesis) for the occupation-basis lowering, and
[`FermionicCircuit.measure_all()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.measure_all) for the convenient way to read out an entire circuit.

Initializing this instruction can be done with the arguments listed below.

* **Parameters:**
  **label** – the string label of this instruction.
