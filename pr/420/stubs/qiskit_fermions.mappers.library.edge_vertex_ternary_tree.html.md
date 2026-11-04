# edge_vertex_ternary_tree

### edge_vertex_ternary_tree(op, encoding, num_qubits)

Map an [`EdgeVertexOperator`](qiskit_fermions.operators.EdgeVertexOperator.html.md#qiskit_fermions.operators.EdgeVertexOperator) to a [`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable) under a
ternary-tree encoding.

#### NOTE
Unlike under Jordan-Wigner, a vertex operator is not in general the weight-1 Pauli
$Z_l$: that is a property of the chain, where the two Majorana strings’ $Z$
chains cancel. On the balanced tree at $n = 4$ a vertex operator has weight 3.

* **Parameters:**
  * **op** – the edge-vertex operator to map.
  * **encoding** – the compiled encoding to map through.
  * **num_qubits** – the number of qubits for the resulting qubit operator. Must be at least the
    encoding’s mode count; a larger value pads with the identity.
* **Returns:**
  The mapped qubit operator, subject to the same simplification and grouping caveats as
  [`fermion_ternary_tree()`](qiskit_fermions.mappers.library.fermion_ternary_tree.html.md#qiskit_fermions.mappers.library.fermion_ternary_tree).
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `op` acts on a mode the encoding does not cover, or `num_qubits` is
        smaller than the encoding’s mode count.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `num_qubits` is negative or does not fit in 32 bits.
