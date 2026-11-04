# edge_vertex_to_majorana

### edge_vertex_to_majorana(inter_op)

Map an [`EdgeVertexOperator`](qiskit_fermions.operators.EdgeVertexOperator.html.md#qiskit_fermions.operators.EdgeVertexOperator) to a [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator).

* **Parameters:**
  **inter_op** – the edge-vertex operator to map.
* **Returns:**
  The mapped majorana operator.

## Definition

This function decomposes the edge-vertex operators in terms of the majorana operators as
defined [here](qiskit_fermions.operators.EdgeVertexOperator.html.md#edgevertexoperator-definition).

## Usage

```pycon
>>> from qiskit_fermions.mappers.library import edge_vertex_to_majorana
>>> from qiskit_fermions.operators import EdgeVertexOperator
>>> inter_op = EdgeVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2})
>>> maj_op = edge_vertex_to_majorana(inter_op)
>>> print(format(maj_op.normal_ordered().simplify()))
  0.000000e0 +1.000000e0j * (γ'0 γ0)
  0.000000e0 +2.000000e0j * (γ2 γ1)
```
