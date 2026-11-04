# SupportsFermionOperator

### *class* SupportsFermionOperator(\*args, \*\*kwargs)

This package’s operator-conversion contract, converting an operator to a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator).

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

See [`fermion_operator()`](qiskit_fermions.mappers.library.fermion_operator.html.md#qiskit_fermions.mappers.library.fermion_operator) for the type-agnostic helper function dispatching to the method
below.

#### CAUTION
Despite the shared method name, this is *not* the same contract as
[`ffsim.SupportsFermionOperator`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsFermionOperator.html#ffsim.SupportsFermionOperator): this one returns a
[`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator), whereas ffsim’s returns an
[`ffsim.FermionOperator`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.FermionOperator.html#ffsim.FermionOperator). The two types are unrelated. Since
[`ffsim.fermion_operator()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.fermion_operator.html#ffsim.fermion_operator) dispatches purely on the method name, calling it on an operator
of this package returns *this* package’s type rather than ffsim’s; use
[`fermion_operator()`](qiskit_fermions.mappers.library.fermion_operator.html.md#qiskit_fermions.mappers.library.fermion_operator) when that is what you mean.

```pycon
>>> from qiskit_fermions.operators import MajoranaOperator
>>> maj_op = MajoranaOperator.from_dict({(0, 1): 1})
>>> fer_op = maj_op._fermion_operator_()
>>> print(format(fer_op.normal_ordered().simplify()))
  0.000000e0 +1.000000e0j * ()
  0.000000e0 -2.000000e0j * (+0 -0)
```

### Protocol Methods

#### \_fermion_operator_()

Converts this operator into a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator) instance.

* **Return type:**
  [FermionOperator](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)
