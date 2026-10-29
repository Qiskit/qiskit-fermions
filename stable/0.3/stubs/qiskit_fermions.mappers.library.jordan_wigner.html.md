# jordan_wigner

### jordan_wigner(operator, num_qubits)

Map an operator to a `SparseObservable` under the Jordan-Wigner transformation.

This is the type-agnostic entry point to the Jordan-Wigner transformation. It dispatches on the
concrete type of `operator` to the matching direct implementation, listed under
[Library implementations](../pydoc/qiskit_fermions.mappers.library.html.md#jordan-wigner-mappers) in the [`library`](../pydoc/qiskit_fermions.mappers.library.html.md#module-qiskit_fermions.mappers.library) documentation. An
operator type without one raises a [`TypeError`](https://docs.python.org/3/builtins/exceptions.html#TypeError).

* **Parameters:**
  * **operator** ([*OperatorTrait*](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait)) – the operator to map.
  * **num_qubits** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of qubits for the resulting qubit operator. This must be strictly
    greater than the largest mode index in `operator` (any additional qubits are padded
    with the identity). Note that for a [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator) the bound is counted in
    *fermionic* modes, i.e. in the largest Majorana index divided by two.
* **Returns:**
  The mapped qubit operator.
* **Raises:**
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `operator` is of a type for which no Jordan-Wigner implementation exists.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `num_qubits` is too small to hold the operator’s support.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `num_qubits` is negative or does not fit in 32 bits.
* **Return type:**
  [*SparseObservable*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable)

```pycon
>>> from qiskit_fermions.mappers.library import jordan_wigner
>>> from qiskit_fermions.operators import FermionOperator
>>> fop = FermionOperator.from_dict({((True, 0), (False, 0)): 0.1})
>>> jordan_wigner(fop, 2).simplify()
<SparseObservable with 2 terms on 2 qubits: (0.05+0j)() + (-0.05+0j)(Z_0)>
```

Every operator type has a direct implementation, so the dispatch is total over them. Anything
else – an object that is not a fermionic operator at all – raises a [`TypeError`](https://docs.python.org/3/builtins/exceptions.html#TypeError):

```pycon
>>> from qiskit.quantum_info import SparseObservable
>>> jordan_wigner(SparseObservable.identity(2), 2)
Traceback (most recent call last):
...
TypeError: jordan_wigner does not support operator type 'SparseObservable'.
```
