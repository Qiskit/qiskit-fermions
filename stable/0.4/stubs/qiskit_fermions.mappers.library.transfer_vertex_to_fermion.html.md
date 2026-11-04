# transfer_vertex_to_fermion

### transfer_vertex_to_fermion(inter_op)

Map a [`TransferVertexOperator`](qiskit_fermions.operators.TransferVertexOperator.html.md#qiskit_fermions.operators.TransferVertexOperator) to a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator).

* **Parameters:**
  **inter_op** – the transfer-vertex operator to map.
* **Returns:**
  The mapped fermionic operator.

## Definition

This function decomposes the transfer-vertex operators in terms of the fermionic action
operators as defined [here](qiskit_fermions.operators.TransferVertexOperator.html.md#transfervertexoperator-definition).

## Usage

```pycon
>>> from qiskit_fermions.mappers.library import transfer_vertex_to_fermion
>>> from qiskit_fermions.operators import TransferVertexOperator
>>> inter_op = TransferVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2, ((2, 1),): -2})
>>> fer_op = transfer_vertex_to_fermion(inter_op)
>>> print(format(fer_op.normal_ordered().simplify()))
  1.000000e0 +0.000000e0j * ()
 -2.000000e0 +0.000000e0j * (-2 -1)
 -2.000000e0 +0.000000e0j * (+0 -0)
  2.000000e0 -0.000000e0j * (+2 +1)
```
