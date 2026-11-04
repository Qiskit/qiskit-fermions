# SupportsMajoranaOperator

### *class* SupportsMajoranaOperator(\*args, \*\*kwargs)

An equivalent of the [`SupportsFermionOperator`](qiskit_fermions.protocols.SupportsFermionOperator.html.md#qiskit_fermions.protocols.SupportsFermionOperator) protocol, targeting Majorana operators.

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

See [`majorana_operator()`](qiskit_fermions.mappers.library.majorana_operator.html.md#qiskit_fermions.mappers.library.majorana_operator) for the type-agnostic helper function dispatching to the method
below.

```pycon
>>> from qiskit_fermions.operators import FermionOperator
>>> fer_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
>>> maj_op = fer_op._majorana_operator_()
>>> print(format(maj_op.normal_ordered().simplify()))
 5.000000e-1 +0.000000e0j * ()
  0.000000e0-5.000000e-1j * (γ'0 γ0)
```

### Protocol Methods

#### \_majorana_operator_()

Converts this operator into a [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator) instance.

* **Return type:**
  [MajoranaOperator](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator)
