# ternary_tree

### ternary_tree(operator, encoding, num_qubits)

Map an operator to a `SparseObservable` under a ternary-tree encoding.

This is the type-agnostic entry point to the ternary-tree mappers, mirroring
[`jordan_wigner()`](qiskit_fermions.mappers.library.jordan_wigner.html.md#qiskit_fermions.mappers.library.jordan_wigner). It dispatches on the concrete type of `operator` to the matching direct
implementation, listed under [Ternary-tree encodings](../pydoc/qiskit_fermions.mappers.library.html.md#ternary-tree-mappers) in the
[`library`](../pydoc/qiskit_fermions.mappers.library.html.md#module-qiskit_fermions.mappers.library) documentation. An operator type without one raises a
[`TypeError`](https://docs.python.org/3/builtins/exceptions.html#TypeError).

* **Parameters:**
  * **operator** ([*OperatorTrait*](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait)) – the operator to map.
  * **encoding** ([*TernaryTreeEncoding*](qiskit_fermions.mappers.TernaryTreeEncoding.html.md#qiskit_fermions.mappers.TernaryTreeEncoding)) – the compiled encoding to map through. Compile it once with
    [`TernaryTreeEncoding`](qiskit_fermions.mappers.TernaryTreeEncoding.html.md#qiskit_fermions.mappers.TernaryTreeEncoding) and reuse it across every operator
    mapped through the same tree.
  * **num_qubits** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of qubits for the resulting qubit operator. Must be at least the
    encoding’s mode count; a larger value pads with the identity.
* **Returns:**
  The mapped qubit operator, subject to the same simplification and grouping caveats as
  [`fermion_ternary_tree()`](qiskit_fermions.mappers.library.fermion_ternary_tree.html.md#qiskit_fermions.mappers.library.fermion_ternary_tree).
* **Raises:**
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `operator` is of a type for which no ternary-tree implementation exists.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `operator` acts on a mode the encoding does not cover, or `num_qubits` is
        smaller than the encoding’s mode count.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `num_qubits` is negative or does not fit in 32 bits.
* **Return type:**
  [*SparseObservable*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable)

```pycon
>>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
>>> from qiskit_fermions.mappers.library import ternary_tree
>>> from qiskit_fermions.operators import FermionOperator
>>> fop = FermionOperator.from_dict({((True, 0), (False, 0)): 0.1})
>>> enc = TernaryTreeEncoding(TernaryTree.breadth_first(2, 3))
>>> ternary_tree(fop, enc, 2).simplify()
<SparseObservable with 2 terms on 2 qubits: (0.05+0j)() + (-0.05+0j)(Z_1 Z_0)>
```

A chain along `Z` reproduces [`jordan_wigner()`](qiskit_fermions.mappers.library.jordan_wigner.html.md#qiskit_fermions.mappers.library.jordan_wigner), which is what makes the tree a
generalisation of it rather than an alternative to it:

```pycon
>>> from qiskit_fermions.mappers.library import jordan_wigner
>>> chain = TernaryTreeEncoding(TernaryTree.chain(2, "Z"))
>>> bool((ternary_tree(fop, chain, 2) - jordan_wigner(fop, 2)).simplify().num_terms == 0)
True
```

Every operator type has a direct implementation, so the dispatch is total over them. Anything
else (an object that is not a fermionic operator at all) raises a [`TypeError`](https://docs.python.org/3/builtins/exceptions.html#TypeError):

```pycon
>>> from qiskit.quantum_info import SparseObservable
>>> ternary_tree(SparseObservable.identity(2), enc, 2)
Traceback (most recent call last):
...
TypeError: ternary_tree does not support operator type 'SparseObservable'.
```
