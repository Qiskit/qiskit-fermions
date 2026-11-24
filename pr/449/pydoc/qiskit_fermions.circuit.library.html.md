<a id="module-qiskit_fermions.circuit.library"></a>

# Circuit Library

This module provides a library of [`FermionicGate`](../stubs/qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) implementations.

Measurement is the one non-unitary operation a [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) can carry, so it is a
[`FermionicInstruction`](../stubs/qiskit_fermions.circuit.FermionicInstruction.html.md#qiskit_fermions.circuit.FermionicInstruction) rather than a [`FermionicGate`](../stubs/qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate):

Should an [`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate be decomposed in fermionic space (an optional step) how that is
done is determined by a synthesis method from the
[`synthesis`](qiskit_fermions.circuit.library.synthesis.html.md#module-qiskit_fermions.circuit.library.synthesis) module.
