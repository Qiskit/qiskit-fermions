# transfer_vertex_ternary_tree

### transfer_vertex_ternary_tree(op, encoding, num_qubits)

Map a [`TransferVertexOperator`](qiskit_fermions.operators.TransferVertexOperator.html.md#qiskit_fermions.operators.TransferVertexOperator) to a [`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable) under
a ternary-tree encoding.

* **Parameters:**
  * **op** – the transfer-vertex operator to map.
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
