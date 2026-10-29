# F2QLayout

### F2QLayout

A mapping of fermionic mode registers to quantum registers.

Users must provide a data structure of this type to the circuit transpiler. In a trivial case, (such
as under the Jordan-Wigner transformation (cf. [`jordan_wigner()`](qiskit_fermions.mappers.library.jordan_wigner.html.md#qiskit_fermions.mappers.library.jordan_wigner))) every fermionic mode gets
mapped to a single qubit. However, not all fermion-to-qubit mappings are of this occupation-based
nature, which is why we do not associate single fermionic modes with single qubits. Crucially, the
size of the registers on either side of this mapping may differ.

alias of [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`QuantumRegister`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister), [`QuantumRegister`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister)]
