# majorana_operator

### majorana_operator(operator)

Converts an operator into a [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator).

This is a thin, type-agnostic wrapper around the [`SupportsMajoranaOperator`](qiskit_fermions.protocols.SupportsMajoranaOperator.html.md#qiskit_fermions.protocols.SupportsMajoranaOperator) protocol
method, mirroring the free-function style of [`fermion_operator()`](qiskit_fermions.mappers.library.fermion_operator.html.md#qiskit_fermions.mappers.library.fermion_operator).

* **Parameters:**
  **operator** ([*SupportsMajoranaOperator*](qiskit_fermions.protocols.SupportsMajoranaOperator.html.md#qiskit_fermions.protocols.SupportsMajoranaOperator)) – the operator to convert, implementing [`SupportsMajoranaOperator`](qiskit_fermions.protocols.SupportsMajoranaOperator.html.md#qiskit_fermions.protocols.SupportsMajoranaOperator).
* **Returns:**
  The operator converted into a [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator).
* **Return type:**
  [*MajoranaOperator*](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator)

```pycon
>>> from qiskit_fermions.mappers.library import majorana_operator
>>> from qiskit_fermions.operators import FermionOperator
>>> fer_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
>>> maj_op = majorana_operator(fer_op)
>>> print(format(maj_op.normal_ordered().simplify()))
 5.000000e-1 +0.000000e0j * ()
  0.000000e0-5.000000e-1j * (γ'0 γ0)
```
