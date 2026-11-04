# fermion_ternary_tree

### fermion_ternary_tree(op, encoding, num_qubits)

Map a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator) to a [`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable) under a
ternary-tree encoding.

* **Parameters:**
  * **op** – the fermionic operator to map.
  * **encoding** – the compiled encoding to map through.
  * **num_qubits** – the number of qubits for the resulting qubit operator. Must be at least the
    encoding’s mode count; a larger value pads with the identity.
* **Returns:**
  The mapped qubit operator. The result is not guaranteed to be fully simplified; call
  [`simplify()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable#simplify) to combine any remaining duplicate
  terms. Duplicates are merged as the result is assembled, to bound the memory required, so
  the exact number of terms returned may vary with the number of threads used. Any grouping
  carried by `op` is not reflected in the result, as a
  [`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable) does not track groups.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `op` acts on a mode the encoding does not cover, or `num_qubits` is
        smaller than the encoding’s mode count.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `num_qubits` is negative or does not fit in 32 bits.

## Usage

On a chain along `Z` this reproduces [`fermion_jordan_wigner()`](qiskit_fermions.mappers.library.fermion_jordan_wigner.html.md#qiskit_fermions.mappers.library.fermion_jordan_wigner) exactly:

```pycon
>>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
>>> from qiskit_fermions.mappers.library import (
...     fermion_jordan_wigner, fermion_ternary_tree,
... )
>>> from qiskit_fermions.operators import FermionOperator
>>> fop = FermionOperator.from_dict({((True, 0), (False, 0)): 0.1})
>>> enc = TernaryTreeEncoding(TernaryTree.chain(2, "Z"))
>>> fermion_ternary_tree(fop, enc, 2).simplify()
<SparseObservable with 2 terms on 2 qubits: (0.05+0j)() + (-0.05+0j)(Z_0)>
>>> bool((fermion_ternary_tree(fop, enc, 2) - fermion_jordan_wigner(fop, 2)).simplify().num_terms == 0)
True
```

Other trees trade that encoding’s Pauli weight, linear in the number of modes, for a shallower
one.
