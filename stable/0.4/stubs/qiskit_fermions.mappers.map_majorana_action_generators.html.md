# map_majorana_action_generators

### map_majorana_action_generators(operator, map_action, identity, compose=None)

Map a [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator) to another operator type.

This is a generic function to aid in implementing new mappers for [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator)
instances. At its core, it simply iterates over the terms of the operator, mapping each
encountered [`MajoranaAction`](qiskit_fermions.operators.majorana_action.MajoranaAction.html.md#qiskit_fermions.operators.majorana_action.MajoranaAction) with the user-provided `map_action` function. In
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
replacement for [`majorana_jordan_wigner()`](qiskit_fermions.mappers.library.majorana_jordan_wigner.html.md#qiskit_fermions.mappers.library.majorana_jordan_wigner), which is parallelized and bounds the memory
it uses while assembling the result.

```pycon
>>> from qiskit_fermions.mappers import map_majorana_action_generators
>>> from qiskit_fermions.operators import MajoranaAction, MajoranaOperator, gamma
>>> from qiskit.quantum_info import SparseObservable
>>>
>>> def jordan_wigner(mode: MajoranaAction) -> SparseObservable:
...     idx = mode // 2
...     qubits = list(range(idx + 1))
...     pauli = "Y" if mode % 2 else "X"
...     return SparseObservable.from_sparse_list(
...         [("Z" * idx + pauli, qubits, 1.0)],
...         num_qubits=num_qubits,
...     )
>>>
>>> num_qubits = 2
>>> def identity() -> SparseObservable:
...     return SparseObservable.identity(num_qubits)
>>>
>>> op = MajoranaOperator.from_dict({
...     (0, 2): 0.5,
...     (1, 3): 0.5,
...     (0, 3): 0.5j,
...     (1, 2): -0.5j,
... })
>>> qop = map_majorana_action_generators(
...     op, jordan_wigner, identity, compose=SparseObservable.compose
... )
>>> print(sorted(qop.simplify().to_sparse_list()))
[('XX', [0, 1], (0.5-0j)), ('XY', [0, 1], 0.5j), ('YX', [0, 1], -0.5j), ('YY', [0, 1], (0.5+0j))]
```

* **Parameters:**
  * **operator** ([*MajoranaOperator*](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator)) – the operator to be mapped.
  * **map_action** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *]* *,* *T* *]*) – the function to map a single [`MajoranaAction`](qiskit_fermions.operators.majorana_action.MajoranaAction.html.md#qiskit_fermions.operators.majorana_action.MajoranaAction) to the desired output type.
  * **identity** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[* *]* *,* *T* *]*) – the function to generate the multiplicative identity instance of the output type.
  * **compose** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[**T* *,* *T* *]* *,* *T* *]*  *|* *None*) – an optional function to implement the compositiion logic of two output type
    instances. If this is not provided, it will default to using [`operator.and_()`](https://docs.python.org/3/library/operator.html#operator.and_).
* **Returns:**
  The mapped operator.
* **Return type:**
  *T*
