# edge_vertex_jordan_wigner

### edge_vertex_jordan_wigner(op, num_qubits)

Map an [`EdgeVertexOperator`](qiskit_fermions.operators.EdgeVertexOperator.html.md#qiskit_fermions.operators.EdgeVertexOperator) to a [`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable) under the
Jordan-Wigner transformation. <sup>[1](#id3)</sup>

Fermionic mode $j$ is mapped to qubit $j$ of the resulting
[`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable), following Qiskit’s little-endian qubit ordering.

* **Parameters:**
  * **op** – the edge-vertex operator to map.
  * **num_qubits** – the number of qubits for the resulting qubit operator. This must be strictly
    greater than the largest mode index in `op` (any additional qubits are padded with the
    identity).
* **Returns:**
  The mapped qubit operator. The result is not guaranteed to be fully simplified; call
  [`simplify()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable#simplify) to combine any remaining duplicate
  terms. Duplicates are merged as the result is assembled, to bound the memory required, so
  the exact number of terms returned may vary with the number of threads used. This does not
  affect the operator that the result represents.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `num_qubits` is too small to hold the operator’s support, i.e. if it is
      not larger than the largest mode index acted upon by `op`.

## Definition

Writing $l_\text{min}$ and $l_\text{max}$ for the smaller and larger of the two
indices, the generalized edge operators map onto single Pauli strings,

$$
\begin{align}
V_l = E_{ll} &\rightarrow \sigma^Z_l \, , \nonumber \\
E_{lr} &\rightarrow \mp \, \sigma^Y_{l_\text{min}}
        \left( \bigotimes_{l_\text{min} \lt k \lt l_\text{max}} \sigma^Z_k \right)
        \sigma^X_{l_\text{max}} \nonumber
\end{align}
$$

where the sign is negative for $l \lt r$ and positive otherwise. The $\sigma^Z$
chains of the two underlying Majorana operators cancel below the lower index, which is why the
$\sigma^Z$ string spans only the modes strictly *between* the two endpoints.

#### NOTE
Reversing the two indices leaves the Pauli string unchanged and flips only the sign, which is
the antisymmetry $E_{lr} = -E_{rl}$. Contrast [`transfer_vertex_jordan_wigner()`](qiskit_fermions.mappers.library.transfer_vertex_jordan_wigner.html.md#qiskit_fermions.mappers.library.transfer_vertex_jordan_wigner),
where the coefficient is the same for both orientations and the Pauli letters change instead.

#### NOTE
These Pauli strings differ from those in Eq. (10) of <sup>[2](#id4)</sup> by an exchange of $\sigma^X$
and $\sigma^Y$ on the two endpoints. This is a single-qubit basis choice – both
conventions satisfy every defining relation of the algebra – and the one used here is the one
consistent with [`edge_vertex_to_fermion()`](qiskit_fermions.mappers.library.edge_vertex_to_fermion.html.md#qiskit_fermions.mappers.library.edge_vertex_to_fermion), so that mapping an operator directly agrees
with converting it to a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator) first.

Mapping directly also avoids an intermediate blowup: each fermionic action maps onto a *two*-term
sum, so routing a term built from $L$ edge operators through a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)
inflates a single Pauli string into up to $4^L$ terms before merging them back down. The
saving grows with the length of the terms; for single-operator terms the two routes cost about the
same.

## Usage

```pycon
>>> from qiskit_fermions.mappers.library import edge_vertex_jordan_wigner
>>> from qiskit_fermions.operators import EdgeVertexOperator
>>> eop = EdgeVertexOperator.from_dict({((0, 0),): 2.0, ((0, 1),): 0.5})
>>> edge_vertex_jordan_wigner(eop, 2).simplify()
<SparseObservable with 2 terms on 2 qubits: (2+0j)(Z_0) + (-0.5+0j)(X_1 Y_0)>
```

* <a id='id3'>**[1]**</a> P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot, Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651, [doi:10.1007/BF01331938](https://link.springer.com/article/10.1007/BF01331938).
* <a id='id4'>**[2]**</a> L. Gandon et al., Fermionic quantum simulation with flow sets, [arXiv:2512.11418](https://arxiv.org/abs/2512.11418).
