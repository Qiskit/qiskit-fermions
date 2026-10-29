# transfer_vertex_to_majorana

### transfer_vertex_to_majorana(inter_op)

Map a [`TransferVertexOperator`](qiskit_fermions.operators.TransferVertexOperator.html.md#qiskit_fermions.operators.TransferVertexOperator) to a [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator).

* **Parameters:**
  **inter_op** – the transfer-vertex operator to map.
* **Returns:**
  The mapped majorana operator.

## Definition

This function decomposes the transfer-vertex operators in terms of the majorana operators as
defined [here](qiskit_fermions.operators.TransferVertexOperator.html.md#transfervertexoperator-definition).

## Usage

```pycon
>>> from qiskit_fermions.mappers.library import transfer_vertex_to_majorana
>>> from qiskit_fermions.operators import TransferVertexOperator
>>> inter_op = TransferVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2, ((2, 1),): -2})
>>> maj_op = transfer_vertex_to_majorana(inter_op)
>>> print(format(maj_op.normal_ordered().simplify()))
  0.000000e0 +1.000000e0j * (γ'0 γ0)
 -0.000000e0 -1.000000e0j * (γ2 γ'1)
 -0.000000e0 -1.000000e0j * (γ'2 γ1)
```
