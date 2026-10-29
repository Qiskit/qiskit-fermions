<a id="qf-mapper-library"></a>

# Mapper Library

The C API provides efficient implementations of commonly used operator representation mapper
routines.

| [`qf_ferm_op_jordan_wigner()`](#c.qf_ferm_op_jordan_wigner)          | Map a [`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator) to a<br/>[QkObs](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-obs) under<br/>the Jordan-Wigner transformation.               |
|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`qf_maj_op_jordan_wigner()`](#c.qf_maj_op_jordan_wigner)           | Map a [`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator) to a<br/>[QkObs](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-obs) under<br/>the Jordan-Wigner transformation.             |
| [`qf_edge_op_jordan_wigner()`](#c.qf_edge_op_jordan_wigner)          | Map a [`QfEdgeVertexOperator`](qf-edge-op.html.md#c.QfEdgeVertexOperator) to a<br/>[QkObs](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-obs) under<br/>the Jordan-Wigner transformation.         |
| [`qf_transfer_op_jordan_wigner()`](#c.qf_transfer_op_jordan_wigner)      | Map a [`QfTransferVertexOperator`](qf-transfer-op.html.md#c.QfTransferVertexOperator) to a<br/>[QkObs](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-obs) under<br/>the Jordan-Wigner transformation. |
| [`qf_fermion_to_majorana()`](#c.qf_fermion_to_majorana)            | Map a [`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator) to a<br/>[`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator).                                 |
| [`qf_majorana_to_fermion()`](#c.qf_majorana_to_fermion)            | Map a [`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator) to a<br/>[`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator).                                 |
| [`qf_edge_vertex_to_fermion()`](#c.qf_edge_vertex_to_fermion)         | Map a [`QfEdgeVertexOperator`](qf-edge-op.html.md#c.QfEdgeVertexOperator) to a<br/>[`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator).                             |
| [`qf_edge_vertex_to_majorana()`](#c.qf_edge_vertex_to_majorana)        | Map a [`QfEdgeVertexOperator`](qf-edge-op.html.md#c.QfEdgeVertexOperator) to a<br/>[`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator).                           |
| [`qf_transfer_vertex_to_fermion()`](#c.qf_transfer_vertex_to_fermion)     | Map a [`QfTransferVertexOperator`](qf-transfer-op.html.md#c.QfTransferVertexOperator) to a<br/>[`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator).                     |
| [`qf_transfer_vertex_to_majorana()`](#c.qf_transfer_vertex_to_majorana)    | Map a [`QfTransferVertexOperator`](qf-transfer-op.html.md#c.QfTransferVertexOperator) to a<br/>[`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator).                   |
| [`qf_transfer_vertex_to_edge_vertex()`](#c.qf_transfer_vertex_to_edge_vertex) | Map a [`QfTransferVertexOperator`](qf-transfer-op.html.md#c.QfTransferVertexOperator) to a<br/>[`QfEdgeVertexOperator`](qf-edge-op.html.md#c.QfEdgeVertexOperator).               |

---

### [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*qf_edge_vertex_to_fermion(const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*edge_op)

Map a `QfEdgeVertexOperator` to a `QfFermionOperator`. 

## Definition

This function decomposes the edge and vertex operators in terms of the fermionic creation and
annihilation operators, as defined [here](qf-edge-op.html.md#qf-edge-op-definition).

## Example

```c
// define some kind of edge-vertex operator
QfEdgeVertexOperator *edge_op = qf_edge_op_one();

// and map it to a fermionic operator
QfFermionOperator *fer_op = qf_edge_vertex_to_fermion(edge_op);
```

 #### NOTE
The mapped operator is not simplified. Because each generator expands into a sum of fermionic
terms, the result generally contains duplicate terms; call
[`qf_ferm_op_normal_ordered()`](qf-ferm-op.html.md#c.qf_ferm_op_normal_ordered) followed by [`qf_ferm_op_simplify()`](qf-ferm-op.html.md#c.qf_ferm_op_simplify) to reduce it.

* **Parameters:**
  * **edge_op** – A pointer to the edge-vertex operator to be mapped.
* **Returns:**
  A pointer to the mapped fermionic operator.

### [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*qf_edge_vertex_to_majorana(const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*edge_op)

Map a `QfEdgeVertexOperator` to a `QfMajoranaOperator`. 

## Definition

This function decomposes the edge and vertex operators in terms of the Majorana operators, as
defined [here](qf-edge-op.html.md#qf-edge-op-definition).

## Example

```c
// define some kind of edge-vertex operator
QfEdgeVertexOperator *edge_op = qf_edge_op_one();

// and map it to a Majorana operator
QfMajoranaOperator *maj_op = qf_edge_vertex_to_majorana(edge_op);
```

 #### NOTE
The mapped operator is not simplified; see [`qf_maj_op_normal_ordered()`](qf-maj-op.html.md#c.qf_maj_op_normal_ordered) and
[`qf_maj_op_simplify()`](qf-maj-op.html.md#c.qf_maj_op_simplify).

* **Parameters:**
  * **edge_op** – A pointer to the edge-vertex operator to be mapped.
* **Returns:**
  A pointer to the mapped Majorana operator.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ferm_op_jordan_wigner(const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op, uint32_t num_qubits, QkObs \*\*out)

Applies the Jordan-Wigner transformation to an operator. 

Map a [`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator) to a
[QkObs](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-obs) under the Jordan-Wigner
transformation. [[JW-ferm]](#jw-ferm)

## Definition

The Jordan-Wigner transformation maps fermionic creation and annihilation operators to spin (or
in this case, qubit) operators:

$$
a^\dagger_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^-_j ~~\text{and}~~
a_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^+_j \, ,
$$

where $a^\dagger_j$ ($a_j$) is the fermionic creation (annihilation) operator
acting on the $j$-th spin-less fermionic mode, $\sigma^P$ with
$P \in \{X,Y,Z\}$ are the spin-$\frac{1}{2}$ Pauli operators and
$\sigma^\pm = (\sigma^X \pm \mathrm{i} \sigma^Y) / 2$.

This mapping preserves the fermionic anti-commutation relations by introducing a chain of
$\sigma^Z$ operators on all qubits preceding the acted-upon index $j$.

## Memory usage

The result is not guaranteed to be fully simplified: duplicate terms are merged as it is
assembled, to bound the memory required, but some may remain. Call `qk_obs_canonicalize` if you
need them all combined. The exact number of terms returned may therefore vary with the number of
threads used, which does not affect the operator that the result represents.

This mapping is parallelized for speed, and that choice costs memory. Each worker thread
accumulates into an observable of its own, and the terms are handed to whichever thread is free
rather than partitioned by which Pauli strings they produce, so every thread ends up holding
roughly a full copy of the mapped operator. Peak memory therefore grows with the number of
threads: expect on the order of the mapped operator’s size *times* the thread count, plus the
input operator.

If memory matters more than wall-clock time, reduce the thread count through rayon’s
`RAYON_NUM_THREADS` environment variable – peak memory falls roughly in proportion, and the
mapping takes correspondingly longer – or map the operator in batches and add the partial results
together yourself, which bounds the peak by the batch size at the cost of repeating the merging
work.

## Example

```c
// define some kind of fermionic operator
QfFermionOperator *hamil = qf_ferm_op_one();

// and map it to a qubit operator
QkObs *result;
QfExitCode exit = qf_ferm_op_jordan_wigner(hamil, 4, &result);

assert(exit == QfExitCode_Success);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator to be mapped. 
  * **num_qubits** – The number of qubits of the resulting operator. This must be strictly greater than the largest mode index acted upon by `op`. 
  * **out** – A pointer to where the created qubit operator will be written on success. It is left untouched if the transformation fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred. In particular, a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` is returned if `num_qubits` is too small to hold the operator’s support.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_maj_op_jordan_wigner(const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op, uint32_t num_qubits, QkObs \*\*out)

Applies the Jordan-Wigner transformation to a Majorana operator. 

Map a [`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator) to a
[QkObs](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-obs) under the Jordan-Wigner
transformation. [[JW-maj]](#jw-maj)

## Definition

With the [`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator) convention that even indices carry
$\gamma_j = a^\dagger_j + a_j$ and odd ones $\gamma'_j = i(a^\dagger_j - a_j)$, the
Majorana index $m$ acts on the fermionic mode $\lfloor m/2 \rfloor$ and maps onto a
single Pauli string,

$$
\gamma_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^X_j ~~\text{and}~~
\gamma'_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^Y_j \, .
$$

This also avoids an intermediate blowup relative to converting to a
[`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator) first: each fermionic action maps onto a *two*-term sum, so that
route inflates a single Pauli string into up to $4^L$ terms for a term built from $L$
Majorana operators. The saving grows with the length of the terms; for single-operator terms the
two routes cost about the same.

## Memory usage

See [`qf_ferm_op_jordan_wigner()`](#c.qf_ferm_op_jordan_wigner); the same parallelization and merging behaviour applies.

## Example

```c
// define some kind of Majorana operator
QfMajoranaOperator *hamil = qf_maj_op_one();

// and map it to a qubit operator
QkObs *result;
QfExitCode exit = qf_maj_op_jordan_wigner(hamil, 4, &result);

assert(exit == QfExitCode_Success);
```

 * **Parameters:**
  * **op** – A pointer to the Majorana operator to be mapped. 
  * **num_qubits** – The number of qubits of the resulting operator. Note that this is counted in *fermionic* modes, so it must be strictly greater than the largest Majorana index acted upon by `op` divided by two. 
  * **out** – A pointer to where the created qubit operator will be written on success. It is left untouched if the transformation fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred. In particular, a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` is returned if `num_qubits` is too small to hold the operator’s support.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_edge_op_jordan_wigner(const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op, uint32_t num_qubits, QkObs \*\*out)

Applies the Jordan-Wigner transformation to an edge-vertex operator. 

Map a [`QfEdgeVertexOperator`](qf-edge-op.html.md#c.QfEdgeVertexOperator) to a
[QkObs](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-obs) under the Jordan-Wigner
transformation. [[JW-edge]](#jw-edge)

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

where the sign is negative for $l \lt r$ and positive otherwise. The $\sigma^Z$ chains
of the two underlying Majorana operators cancel below the lower index, which is why the
$\sigma^Z$ string spans only the modes strictly *between* the two endpoints.

## Memory usage

See [`qf_ferm_op_jordan_wigner()`](#c.qf_ferm_op_jordan_wigner); the same parallelization and merging behaviour applies.

## Example

```c
// define some kind of edge-vertex operator
QfEdgeVertexOperator *hamil = qf_edge_op_one();

// and map it to a qubit operator
QkObs *result;
QfExitCode exit = qf_edge_op_jordan_wigner(hamil, 4, &result);

assert(exit == QfExitCode_Success);
```

 #### NOTE
Reversing the two indices leaves the Pauli string unchanged and flips only the sign, which is
the antisymmetry $E_{lr} = -E_{rl}$. Contrast
[`qf_transfer_op_jordan_wigner()`](#c.qf_transfer_op_jordan_wigner), where the coefficient is the same for both
orientations and the Pauli letters change instead.

#### NOTE
These Pauli strings differ from those in Eq. (10) of [[Gandon-edge]](#gandon-edge) by an exchange of $\sigma^X$
and $\sigma^Y$ on the two endpoints. This is a single-qubit basis choice – both
conventions satisfy every defining relation of the algebra – and the one used here is the one
consistent with [`qf_edge_vertex_to_fermion()`](#c.qf_edge_vertex_to_fermion), so that mapping an operator directly
agrees with converting it to a [`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator) first.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator to be mapped. 
  * **num_qubits** – The number of qubits of the resulting operator. This must be strictly greater than the largest mode index acted upon by `op`. 
  * **out** – A pointer to where the created qubit operator will be written on success. It is left untouched if the transformation fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred. In particular, a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` is returned if `num_qubits` is too small to hold the operator’s support.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_transfer_op_jordan_wigner(const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op, uint32_t num_qubits, QkObs \*\*out)

Applies the Jordan-Wigner transformation to a transfer-vertex operator. 

Map a [`QfTransferVertexOperator`](qf-transfer-op.html.md#c.QfTransferVertexOperator) to a
[QkObs](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-obs) under the Jordan-Wigner
transformation. [[JW-transfer]](#jw-transfer)

## Definition

Writing $l_\text{min}$ and $l_\text{max}$ for the smaller and larger of the two
indices, the generalized transfer operators map onto single Pauli strings,

$$
\begin{align}
V_l = T_{ll} &\rightarrow \sigma^Z_l \, , \nonumber \\
T_{lr} &\rightarrow -\frac{1}{2} \, \sigma^P_{l_\text{min}}
        \left( \bigotimes_{l_\text{min} \lt k \lt l_\text{max}} \sigma^Z_k \right)
        \sigma^P_{l_\text{max}} \nonumber
\end{align}
$$

where $P = X$ for $l \lt r$ and $P = Y$ otherwise.

## Memory usage

See [`qf_ferm_op_jordan_wigner()`](#c.qf_ferm_op_jordan_wigner); the same parallelization and merging behaviour applies.

## Example

```c
// define some kind of transfer-vertex operator
QfTransferVertexOperator *hamil = qf_transfer_op_one();

// and map it to a qubit operator
QkObs *result;
QfExitCode exit = qf_transfer_op_jordan_wigner(hamil, 4, &result);

assert(exit == QfExitCode_Success);
```

 #### NOTE
The index order works the opposite way round to [`qf_edge_op_jordan_wigner()`](#c.qf_edge_op_jordan_wigner): the
coefficient is $-1/2$ for **both** orientations and it is the Pauli letters that swap.
$T_{lr}$ and $T_{rl}$ are genuinely different operators, with no antisymmetry
relating them.

#### NOTE
As for [`qf_edge_op_jordan_wigner()`](#c.qf_edge_op_jordan_wigner), these Pauli strings differ from Eq. (10) of [[Gandon-transfer]](#gandon-transfer) by
a single-qubit basis choice; the convention used here is the one consistent with
[`qf_transfer_vertex_to_fermion()`](#c.qf_transfer_vertex_to_fermion).

* **Parameters:**
  * **op** – A pointer to the transfer-vertex operator to be mapped. 
  * **num_qubits** – The number of qubits of the resulting operator. This must be strictly greater than the largest mode index acted upon by `op`. 
  * **out** – A pointer to where the created qubit operator will be written on success. It is left untouched if the transformation fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred. In particular, a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` is returned if `num_qubits` is too small to hold the operator’s support.

### [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*qf_fermion_to_majorana(const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*fer_op)

Map a `QfFermionOperator` to a `QfMajoranaOperator`. 

## Definition

This function implements the simple transformation:

$$
a^\dagger_j \rightarrow \frac{1}{2} (\gamma_j - i \gamma'_j) ~~\text{and}~~
a_j \rightarrow \frac{1}{2} (\gamma_j + i \gamma'_j)
$$

where $a^\dagger_j$ ($a_j$) is the fermionic creation (annihilation) operator
acting on the $j$-th spin-less fermionic mode, and $\gamma_j$/$\gamma'_j$ are
the two Majorana fermion operators. In the case of the [`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator) these
will be stored on the even and odd Majorana modes, respectively.

## Example

```c
// define some kind of fermionic operator
QfFermionOperator *fer_op = qf_ferm_op_one();

// and map it to a majorana operator
QfMajoranaOperator *maj_op = qf_fermion_to_majorana(fer_op);
```

 * **Parameters:**
  * **fer_op** – A pointer to the fermionic operator to be mapped.
* **Returns:**
  A pointer to the mapped majorana operator.

### [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*qf_majorana_to_fermion(const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*maj_op)

Map a `QfMajoranaOperator` to a `QfFermionOperator`. 

## Definition

This function implements the simple transformation:

$$
\gamma_j \rightarrow a^\dagger_j + a_j ~~\text{and}~~
\gamma'_j \rightarrow i (a^\dagger_j - a_j)
$$

where $\gamma_j$/$\gamma'_j$ are the two Majorana fermion operators (stored on the
even and odd modes, respectively), and $a^\dagger_j$ ($a_j$) is the fermionic
creation (annihilation) operator acting on the $j$-th spin-less fermionic mode.

## Example

```c
// define some kind of majorana operator
QfMajoranaOperator *maj_op = qf_maj_op_one();

// and map it to a fermion operator
QfFermionOperator *fer_op = qf_majorana_to_fermion(maj_op);
```

 * **Parameters:**
  * **maj_op** – A pointer to the majorana operator to be mapped.
* **Returns:**
  A pointer to the mapped fermion operator.

### [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*qf_transfer_vertex_to_fermion(const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*transfer_op)

Map a `QfTransferVertexOperator` to a `QfFermionOperator`. 

## Definition

This function decomposes the transfer and vertex operators in terms of the fermionic creation
and annihilation operators, as defined [here](qf-transfer-op.html.md#qf-transfer-op-definition).

## Example

```c
// define some kind of transfer-vertex operator
QfTransferVertexOperator *transfer_op = qf_transfer_op_one();

// and map it to a fermionic operator
QfFermionOperator *fer_op = qf_transfer_vertex_to_fermion(transfer_op);
```

 #### NOTE
The mapped operator is not simplified. Because each generator expands into a sum of fermionic
terms, the result generally contains duplicate terms; call
[`qf_ferm_op_normal_ordered()`](qf-ferm-op.html.md#c.qf_ferm_op_normal_ordered) followed by [`qf_ferm_op_simplify()`](qf-ferm-op.html.md#c.qf_ferm_op_simplify) to reduce it.

* **Parameters:**
  * **transfer_op** – A pointer to the transfer-vertex operator to be mapped.
* **Returns:**
  A pointer to the mapped fermionic operator.

### [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*qf_transfer_vertex_to_majorana(const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*transfer_op)

Map a `QfTransferVertexOperator` to a `QfMajoranaOperator`. 

## Definition

This function decomposes the transfer and vertex operators in terms of the Majorana operators,
as defined [here](qf-transfer-op.html.md#qf-transfer-op-definition).

## Example

```c
// define some kind of transfer-vertex operator
QfTransferVertexOperator *transfer_op = qf_transfer_op_one();

// and map it to a Majorana operator
QfMajoranaOperator *maj_op = qf_transfer_vertex_to_majorana(transfer_op);
```

 #### NOTE
The mapped operator is not simplified; see [`qf_maj_op_normal_ordered()`](qf-maj-op.html.md#c.qf_maj_op_normal_ordered) and
[`qf_maj_op_simplify()`](qf-maj-op.html.md#c.qf_maj_op_simplify).

* **Parameters:**
  * **transfer_op** – A pointer to the transfer-vertex operator to be mapped.
* **Returns:**
  A pointer to the mapped Majorana operator.

### [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*qf_transfer_vertex_to_edge_vertex(const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*transfer_op)

Map a `QfTransferVertexOperator` to a `QfEdgeVertexOperator`. 

## Definition

This function rewrites each transfer operator in terms of the edge and vertex operators. A
vertex operator maps to itself, whereas a transfer operator $T_{jk}$ becomes a
length-two product of an edge and a vertex operator.

## Example

```c
// define some kind of transfer-vertex operator
QfTransferVertexOperator *transfer_op = qf_transfer_op_one();

// and map it to an edge-vertex operator
QfEdgeVertexOperator *edge_op = qf_transfer_vertex_to_edge_vertex(transfer_op);
```

 #### NOTE
This is the one mapper that stays within the interaction-operator representations, so it is
the natural route to compare the two: mapping a transfer operator to an edge-vertex operator
and then on to a fermionic one agrees with mapping it to a fermionic operator directly.

* **Parameters:**
  * **transfer_op** – A pointer to the transfer-vertex operator to be mapped.
* **Returns:**
  A pointer to the mapped edge-vertex operator.
