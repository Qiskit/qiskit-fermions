# map_fermion_action_generators

### map_fermion_action_generators(operator, map_action, identity, compose=None)

Map a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator) to another operator type.

This is a generic function to aid in implementing new mappers for [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)
instances. At its core, it simply iterates over the terms of the operator, mapping each
encountered [`FermionAction`](qiskit_fermions.operators.fermion_action.FermionAction.html.md#qiskit_fermions.operators.fermion_action.FermionAction) with the user-provided `map_action` function. In
combination with the user-provided `identity` generator, this allows mapping to arbitrary
output types.

#### NOTE
The output type `T` must support multiplication by a scalar via `__mul__`.
If `compose=None` it must also support composition of two instances via `__and__`.
[`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable) does not, which is why the example below
names [`compose()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable#compose) explicitly; a type with an
`__and__` (such as [`SparsePauliOp`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparsePauliOp)) can rely on the default.

#### NOTE
The mapping written out below is a minimal illustration of this function rather than a
replacement for [`fermion_jordan_wigner()`](qiskit_fermions.mappers.library.fermion_jordan_wigner.html.md#qiskit_fermions.mappers.library.fermion_jordan_wigner), which is parallelized and bounds the memory it
uses while assembling the result.

```pycon
>>> from qiskit_fermions.mappers import map_fermion_action_generators
>>> from qiskit_fermions.operators import FermionAction, FermionOperator, ann, cre
>>> from qiskit.quantum_info import SparseObservable
>>>
>>> def jordan_wigner(action: FermionAction) -> SparseObservable:
...     act, idx = action
...     qubits = list(range(idx + 1))
...     return SparseObservable.from_sparse_list(
...         [
...             ("Z" * idx + "X", qubits, 0.5),
...             ("Z" * idx + "Y", qubits, -0.5j if act else 0.5j),
...         ],
...         num_qubits=num_qubits,
...     )
>>>
>>> num_qubits = 2
>>> def identity() -> SparseObservable:
...     return SparseObservable.identity(num_qubits)
>>>
>>> op = FermionOperator.from_dict({(cre(0), ann(1)): 2.0})
>>> qop = map_fermion_action_generators(
...     op, jordan_wigner, identity, compose=SparseObservable.compose
... )
>>> print(sorted(qop.simplify().to_sparse_list()))
[('XX', [0, 1], (0.5+0j)), ('XY', [0, 1], 0.5j), ('YX', [0, 1], -0.5j), ('YY', [0, 1], (0.5+0j))]
```

* **Parameters:**
  * **operator** ([*FermionOperator*](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)) – the operator to be mapped.
  * **map_action** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*bool*](https://docs.python.org/3/builtins/functions.html#bool) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]* *]* *,* *T* *]*) – the function to map a single [`FermionAction`](qiskit_fermions.operators.fermion_action.FermionAction.html.md#qiskit_fermions.operators.fermion_action.FermionAction) to the desired output type.
  * **identity** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[* *]* *,* *T* *]*) – the function to generate the multiplicative identity instance of the output type.
  * **compose** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[**T* *,* *T* *]* *,* *T* *]*  *|* *None*) – an optional function to implement the compositiion logic of two output type
    instances. If this is not provided, it will default to using [`operator.and_()`](https://docs.python.org/3/library/operator.html#operator.and_).
* **Returns:**
  The mapped operator.
* **Return type:**
  *T*
