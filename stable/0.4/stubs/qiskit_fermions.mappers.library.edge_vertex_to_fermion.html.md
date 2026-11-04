# edge_vertex_to_fermion

### edge_vertex_to_fermion(inter_op)

Map an [`EdgeVertexOperator`](qiskit_fermions.operators.EdgeVertexOperator.html.md#qiskit_fermions.operators.EdgeVertexOperator) to a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator).

* **Parameters:**
  **inter_op** – the edge-vertex operator to map.
* **Returns:**
  The mapped fermionic operator.

## Definition

This function decomposes the edge-vertex operators in terms of the fermionic action operators
as defined [here](qiskit_fermions.operators.EdgeVertexOperator.html.md#edgevertexoperator-definition).

## Usage

```pycon
>>> from qiskit_fermions.mappers.library import edge_vertex_to_fermion
>>> from qiskit_fermions.operators import EdgeVertexOperator
>>> inter_op = EdgeVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2})
>>> fer_op = edge_vertex_to_fermion(inter_op)
>>> print(format(fer_op.normal_ordered().simplify()))
  1.000000e0 +0.000000e0j * ()
 -0.000000e0 +2.000000e0j * (-2 -1)
 -2.000000e0 +0.000000e0j * (+0 -0)
  0.000000e0 -2.000000e0j * (+1 -2)
 -0.000000e0 +2.000000e0j * (+2 -1)
 -0.000000e0 +2.000000e0j * (+2 +1)
```
