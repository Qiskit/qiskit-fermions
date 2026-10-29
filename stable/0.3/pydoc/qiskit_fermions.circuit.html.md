<a id="module-qiskit_fermions.circuit"></a>

# Circuit Representation

With this module you can build [`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit) objects that are
interpreted as acting on fermionic modes rather than qubits.

The following objects are simple type aliases to make this re-interpretation more clear:

For actually expressing your circuits, you should use the [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) class along with
the [`library`](qiskit_fermions.circuit.library.html.md#module-qiskit_fermions.circuit.library) of [`FermionicGate`](../stubs/qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) implementations.
