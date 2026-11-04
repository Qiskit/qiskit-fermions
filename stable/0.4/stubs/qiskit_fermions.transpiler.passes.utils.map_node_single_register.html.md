# map_node_single_register

### map_node_single_register(in_node, f2q_layout)

Map the single fermionic mode register acted upon by the circuit operation.

Given a [`DAGOpNode`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode) and a
[`F2QLayout`](qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout) this function will:

1. ensure that all acted-upon fermionic modes belong to the same register
2. map the node’s indices to their global indices with respect to said fermionic mode register
3. map this fermionic mode register to the corresponding qubit register

#### CAUTION
The returned global fermionic mode register indices are knowingly dependent on the order of
the fermionic modes during the construction of their original
[`FermionicRegister`](qiskit_fermions.circuit.FermionicRegister.html.md#qiskit_fermions.circuit.FermionicRegister). This is unlikely to cause any issues but
makes the usage of [`RelabelModes`](qiskit_fermions.transpiler.passes.RelabelModes.html.md#qiskit_fermions.transpiler.passes.RelabelModes) possibly more generally.

* **Parameters:**
  * **in_node** ([*DAGOpNode*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode)) – the node whose indices to parse.
  * **f2q_layout** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *[*[*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *,* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *]*) – the mapping of fermionic to qubit registers.
* **Returns:**
  The list of the global fermionic mode indices and mapped qubit register instance.
* **Raises:**
  [**NotImplementedError**](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError) – when the provided node acts on fermionic modes that do not all belong
      to the same [`FermionicRegister`](qiskit_fermions.circuit.FermionicRegister.html.md#qiskit_fermions.circuit.FermionicRegister).
* **Return type:**
  [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[list](https://docs.python.org/3/builtins/stdtypes.html#list)[[int](https://docs.python.org/3/builtins/functions.html#int)], [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister)]
