# fermion_operator

### fermion_operator(operator)

Converts an operator into a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator).

This is a thin, type-agnostic wrapper around the [`SupportsFermionOperator`](qiskit_fermions.protocols.SupportsFermionOperator.html.md#qiskit_fermions.protocols.SupportsFermionOperator) protocol
method, mirroring the free-function style of [`ffsim.fermion_operator()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.fermion_operator.html#ffsim.fermion_operator).

* **Parameters:**
  **operator** ([*SupportsFermionOperator*](qiskit_fermions.protocols.SupportsFermionOperator.html.md#qiskit_fermions.protocols.SupportsFermionOperator)) – the operator to convert, implementing [`SupportsFermionOperator`](qiskit_fermions.protocols.SupportsFermionOperator.html.md#qiskit_fermions.protocols.SupportsFermionOperator).
* **Returns:**
  The operator converted into a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator).
* **Return type:**
  [*FermionOperator*](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)

```pycon
>>> from qiskit_fermions.mappers.library import fermion_operator
>>> from qiskit_fermions.operators import MajoranaOperator
>>> maj_op = MajoranaOperator.from_dict({(0, 1): 1})
>>> fer_op = fermion_operator(maj_op)
>>> print(format(fer_op.normal_ordered().simplify()))
  0.000000e0 +1.000000e0j * ()
  0.000000e0 -2.000000e0j * (+0 -0)
```
