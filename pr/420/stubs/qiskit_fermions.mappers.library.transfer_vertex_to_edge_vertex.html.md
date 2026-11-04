# transfer_vertex_to_edge_vertex

### transfer_vertex_to_edge_vertex(inter_op)

Map a [`TransferVertexOperator`](qiskit_fermions.operators.TransferVertexOperator.html.md#qiskit_fermions.operators.TransferVertexOperator) to an [`EdgeVertexOperator`](qiskit_fermions.operators.EdgeVertexOperator.html.md#qiskit_fermions.operators.EdgeVertexOperator).

* **Parameters:**
  **inter_op** – the transfer-vertex operator to map.
* **Returns:**
  The mapped edge-vertex operator.

## Definition

This function decomposes the transfer-vertex operators in terms of the edge-vertex operators as
defined [here](qiskit_fermions.operators.TransferVertexOperator.html.md#transfervertexoperator-definition).

## Usage

```pycon
>>> from qiskit_fermions.mappers.library import transfer_vertex_to_edge_vertex
>>> from qiskit_fermions.operators import TransferVertexOperator
>>> dir_op = TransferVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2, ((2, 1),): -2})
>>> undir_op = transfer_vertex_to_edge_vertex(dir_op)
>>> print(format(undir_op.simplify()))
  1.000000e0 +0.000000e0j * (V(0))
  0.000000e0 +1.000000e0j * (V(1) E(1,2))
 -0.000000e0 -1.000000e0j * (E(1,2) V(2))
```
