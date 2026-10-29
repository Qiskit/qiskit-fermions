# QfEdgeVertexOperator

### struct QfEdgeVertexOperator

An edge-vertex operator.

#### NOTE
This is an opaque data structure to the C API whose internals are implemented entirely in Rust.
The remainder of this page describes the design and related functions to work with this struct.

---

<a id="qf-edge-op-definition"></a>

## Definition

This operator is defined in terms of the edge-vertex ($E_{jk}$, $V_j$) operators:

$$
\begin{align}
V_j    &= -i \gamma_{2j-1} \gamma_{2j}
        = 1 - 2 a^\dagger_j a_j \, , \nonumber \\
E_{jk} &= -i \gamma_{2j-1} \gamma_{2k-1}
        = -i (a_j a_k + a_j a^\dagger_k + a^\dagger_j a_k + a^\dagger_j a^\dagger_k)
        = -E_{kj} \nonumber
\end{align}
$$

which fulfill the following mixed fermionic-bosonic commutation relations for $j \neq k \neq
l \neq m$: <sup>[1](#id2)</sup>

$$
\begin{align}
\left\{ E_{jk}, V_k \right\} &= 0 \nonumber \\
\left\{ E_{jk}, E_{kl} \right\} &= 0 \nonumber \\
\left[ V_k, V_l \right] &= 0 \nonumber \\
\left[ E_{jk}, V_l \right] &= 0 \nonumber \\
\left[ E_{jk}, E_{lm} \right] &= 0 \nonumber \, .
\end{align}
$$

In summary, edge and vertex operators commute, unless they share *exactly one* index, in which case
they anticommute.

#### NOTE
The relations above are stated for $j \neq k \neq l \neq m$, so they do not cover two edge
operators spanning the *same* pair of modes. Those commute: since $E_{kj} = -E_{jk}$, such
a pair is collinear, and every operator commutes with itself. This is why the condition is
“exactly one” shared index rather than “at least one”.

We can abuse the notation a little bit and define $V_j = E_{jj}$ which reflects how the
internal data structure of this operator works. This makes the definition of the entire operator the
following:

$$
\text{\texttt{EdgeVertexOperator}} = \sum_i c_i \bigotimes_{lr} E_{lr} \, ,
$$

where $lr$ indexes the involved operator terms and $c_i$ is the (complex) coefficient
making up the linear combination of products. The indices $l$ and $r$ can take any value
between 0 and the number of fermionic modes acted upon by the operator minus 1.

We will refer to $E_{lr}$ as generalized edge operators.

---

<a id="qf-edge-op-implementation"></a>

## Implementation

This struct stores the terms and coefficients in multiple sparse vectors, akin to the
[compressed sparse row format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
commonly used for sparse matrices. More concretely, a single operator contains 4 arrays:

| `coeffs`        | A vector of complex coefficients consisting of two 64-bit floating point numbers.   |
|-----------------|-------------------------------------------------------------------------------------|
| `left_indices`  | A vector of 32-bit integers storing the left fermionic mode indices ($l$).          |
| `right_indices` | A vector of 32-bit integers storing the right fermionic mode indices ($r$).         |
| `boundaries`    | A vector of integers indicating the boundaries between terms.                       |

Fermionic modes indexed by `left_indices` and `right_indices` are considered spinless. The two
index arrays always have the same length, since every generator is identified by exactly one
`(left, right)` pair; this is why the constructor takes a single `num_indices` length for both.

#### NOTE
You can access **read-only borrows** of these internal arrays via their respective functions:

- [`qf_edge_op_get_coeffs()`](#c.qf_edge_op_get_coeffs)
- [`qf_edge_op_get_left_indices()`](#c.qf_edge_op_get_left_indices)
- [`qf_edge_op_get_right_indices()`](#c.qf_edge_op_get_right_indices)
- [`qf_edge_op_get_boundaries()`](#c.qf_edge_op_get_boundaries)

The returned pointers stay valid only until the operator is modified or freed, and must not be
freed by the caller.

This data structure allows for very efficient construction and manipulation of operators.
However, it implies that duplicate terms might be contained in an operator at any moment.
These must be resolved manually through the use of [`qf_edge_op_simplify()`](#c.qf_edge_op_simplify).

### Construction

A new operator can be constructed directly by specifying the corresponding arrays outlined above.
Alternatively, an empty [`QfEdgeVertexOperator`](#c.QfEdgeVertexOperator) can be initialized with
[`qf_edge_op_zero()`](#c.qf_edge_op_zero) and terms can be added iteratively via [`qf_edge_op_add_term()`](#c.qf_edge_op_add_term).

| [`qf_edge_op_new()`](#c.qf_edge_op_new)      | Constructs a new operator from the provided arrays.   |
|------------------------------------------------------------------------|-------------------------------------------------------|
| [`qf_edge_op_zero()`](#c.qf_edge_op_zero)     | Constructs the additive identity operator.            |
| [`qf_edge_op_one()`](#c.qf_edge_op_one)      | Constructs the multiplicative identity operator.      |
| [`qf_edge_op_add_term()`](#c.qf_edge_op_add_term) | Adds a term to an existing `QfEdgeVertexOperator`.    |

#### NOTE
A [`QfEdgeVertexOperator`](#c.QfEdgeVertexOperator) can be freed with [`qf_edge_op_free()`](#c.qf_edge_op_free).

### Arithmetics

The following functions provide arithmetic manipulation:

| [`qf_edge_op_add()`](#c.qf_edge_op_add)     | Adds two operators together.              |
|-----------------------------------------------------------------------|-------------------------------------------|
| [`qf_edge_op_mul()`](#c.qf_edge_op_mul)     | Multiplies an operator by a scalar.       |
| [`qf_edge_op_compose()`](#c.qf_edge_op_compose) | Composes two operators with each other.   |
| [`qf_edge_op_adjoint()`](#c.qf_edge_op_adjoint) | Returns the Hermitian conjugate operator. |

### Manipulation

The following functions provide operator manipulation logic:

| [`qf_edge_op_ichop()`](#c.qf_edge_op_ichop)          | Removes terms with small coefficient magnitudes.          |
|------------------------------------------------------------------------------|-----------------------------------------------------------|
| [`qf_edge_op_simplify()`](#c.qf_edge_op_simplify)       | Returns an equivalent but simplified operator.            |
| [`qf_edge_op_normal_ordered()`](#c.qf_edge_op_normal_ordered) | Returns an equivalent operator with normal ordered terms. |
| [`qf_edge_op_relabel_modes()`](#c.qf_edge_op_relabel_modes)  | Relabels the modes of an operator.                        |

### Properties

The following functions exist to check certain properties of an operator.

| [`qf_edge_op_is_hermitian()`](#c.qf_edge_op_is_hermitian)   | Returns whether an operator is Hermitian.             |
|------------------------------------------------------------------------------|-------------------------------------------------------|
| [`qf_edge_op_len()`](#c.qf_edge_op_len)            | Returns the number of terms in this operator.         |
| [`qf_edge_op_min_num_modes()`](#c.qf_edge_op_min_num_modes)  | Returns the minimum number of modes of this operator. |

### Mapping

The following functions map this operator into another representation:

| [`qf_edge_vertex_to_fermion()`](qf-mappers-library.html.md#c.qf_edge_vertex_to_fermion)   | Maps to a [`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator).   |
|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [`qf_edge_vertex_to_majorana()`](qf-mappers-library.html.md#c.qf_edge_vertex_to_majorana) | Maps to a [`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator). |

---
* <a id='id2'>**[1]**</a> [https://arxiv.org/abs/2512.11418](https://arxiv.org/abs/2512.11418)

## Members

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_new(uint64_t num_terms, uint64_t num_indices, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*coeffs, const uint32_t \*left_indices, const uint32_t \*right_indices, const uint32_t \*boundaries)

Constructs a new operator. 

Any of the pointer arguments can be `NULL` if and only if their corresponding length is zero.

A generator with equal indices is a vertex operator, $V_j = E_{jj}$; one with differing
indices is a (generalized) edge operator, $E_{jk}$.

### Example

```c
// this builds `1.0 + (-1.0) * V(0) E(0,1)`
uint64_t num_terms = 2;
uint64_t num_indices = 2;
uint32_t left_indices[2] = {0, 0};
uint32_t right_indices[2] = {0, 1};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {-1.0, 0.0}};
uint32_t boundaries[3] = {0, 0, 2};
QfEdgeVertexOperator *op = qf_edge_op_new(num_terms, num_indices, coeffs,
                                          left_indices, right_indices, boundaries);
```

 * **Parameters:**
  * **num_terms** – The number of terms in the operator. 
  * **num_indices** – The number of generators summed over all terms. Both index arrays have this same length, since every generator is identified by exactly one `(left, right)` pair. 
  * **coeffs** – A pointer to an array of term coefficients. The length of this array should be `num_terms`. 
  * **left_indices** – A pointer to an array of left-hand mode indices over all terms. The length of this array should be `num_indices`. 
  * **right_indices** – A pointer to an array of right-hand mode indices over all terms. The length of this array should be `num_indices`. 
  * **boundaries** – A pointer to an array of the boundaries between terms. The length of this array should be `num_terms + 1`.
* **Returns:**
  A pointer to the created operator.

### void qf_edge_op_free([QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op)

Frees an existing operator. 

### Example

```c
QfEdgeVertexOperator *op = qf_edge_op_one();
qf_edge_op_free(op);
```

 * **Parameters:**
  * **op** – A pointer to the edge-vertex operator to be freed.

### void qf_edge_op_get_coeffs(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*\*coeffs_out, uint64_t \*coeffs_len)

Provides read-only access to the operator’s coefficients. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-edge-op-implementation).

### Example

```c
uint64_t num_terms = 2;
QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[3] = {0, 0, 0};
QfEdgeVertexOperator *op =
    qf_edge_op_new(num_terms, 0, coeffs, NULL, NULL, boundaries);

QkComplex64 *coeffs_out;
uint64_t coeffs_len;

qf_edge_op_get_coeffs(op, &coeffs_out, &coeffs_len);

assert(coeffs_len == 2);
```

 #### NOTE
This function borrows the operator’s internal buffer rather than copying it. The returned
pointer stays valid only until the operator is modified or freed; do not free it yourself.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose coefficients to access. 
  * **coeffs_out** – A pointer to the array of complex values into which to write the coefficients. 
  * **coeffs_len** – A pointer to the integer into which to write the length of the output array.

### void qf_edge_op_get_left_indices(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, uint32_t \*\*left_indices_out, uint64_t \*left_indices_len)

Provides read-only access to the operator’s left-hand mode indices. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-edge-op-implementation).

### Example

```c
uint64_t num_terms = 1;
uint64_t num_indices = 2;
uint32_t left_indices[2] = {0, 0};
uint32_t right_indices[2] = {0, 1};
QkComplex64 coeffs[1] = {{1.0, 0.0}};
uint32_t boundaries[2] = {0, 2};
QfEdgeVertexOperator *op = qf_edge_op_new(num_terms, num_indices, coeffs,
                                          left_indices, right_indices, boundaries);

uint32_t *left_out;
uint64_t left_len;

qf_edge_op_get_left_indices(op, &left_out, &left_len);

assert(left_len == 2);
```

 #### NOTE
This function borrows the operator’s internal buffer rather than copying it. The returned
pointer stays valid only until the operator is modified or freed; do not free it yourself.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose left indices to access. 
  * **left_indices_out** – A pointer to the array of integers into which to write the indices. 
  * **left_indices_len** – A pointer to the integer into which to write the length of the output array.

### void qf_edge_op_get_right_indices(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, uint32_t \*\*right_indices_out, uint64_t \*right_indices_len)

Provides read-only access to the operator’s right-hand mode indices. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-edge-op-implementation).

### Example

```c
uint64_t num_terms = 1;
uint64_t num_indices = 2;
uint32_t left_indices[2] = {0, 0};
uint32_t right_indices[2] = {0, 1};
QkComplex64 coeffs[1] = {{1.0, 0.0}};
uint32_t boundaries[2] = {0, 2};
QfEdgeVertexOperator *op = qf_edge_op_new(num_terms, num_indices, coeffs,
                                          left_indices, right_indices, boundaries);

uint32_t *right_out;
uint64_t right_len;

qf_edge_op_get_right_indices(op, &right_out, &right_len);

assert(right_len == 2);
```

 #### NOTE
This function borrows the operator’s internal buffer rather than copying it. The returned
pointer stays valid only until the operator is modified or freed; do not free it yourself.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose right indices to access. 
  * **right_indices_out** – A pointer to the array of integers into which to write the indices. 
  * **right_indices_len** – A pointer to the integer into which to write the length of the output array.

### void qf_edge_op_get_boundaries(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, size_t \*\*boundaries_out, uint64_t \*boundaries_len)

Provides read-only access to the indices indicating the boundaries between operator terms. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-edge-op-implementation).

### Example

```c
uint64_t num_terms = 1;
uint64_t num_indices = 2;
uint32_t left_indices[2] = {0, 0};
uint32_t right_indices[2] = {0, 1};
QkComplex64 coeffs[1] = {{1.0, 0.0}};
uint32_t boundaries[2] = {0, 2};
QfEdgeVertexOperator *op = qf_edge_op_new(num_terms, num_indices, coeffs,
                                          left_indices, right_indices, boundaries);

size_t *boundaries_out;
uint64_t boundaries_len;

qf_edge_op_get_boundaries(op, &boundaries_out, &boundaries_len);

assert(boundaries_len == 2);
```

 #### NOTE
This function borrows the operator’s internal buffer rather than copying it. The returned
pointer stays valid only until the operator is modified or freed; do not free it yourself.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose boundaries to access. 
  * **boundaries_out** – A pointer to the array of integers into which to write the boundaries. 
  * **boundaries_len** – A pointer to the integer into which to write the length of the output array.

### uint32_t qf_edge_op_num_support(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op)

Gets the size of the support of an operator. 

Use this to size the output buffer of [`qf_edge_op_get_support()`](#c.qf_edge_op_get_support), which does not report a
length of its own.

### Example

```c
uint32_t left_indices[3] = {2, 0, 2};
uint32_t right_indices[3] = {2, 1, 2};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[3] = {0, 2, 3};
QfEdgeVertexOperator *op =
    qf_edge_op_new(2, 3, coeffs, left_indices, right_indices, boundaries);

uint32_t num_support = qf_edge_op_num_support(op);

assert(num_support == 3);
```

 * **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose support size to get.
* **Returns:**
  The number of distinct mode indices acted upon by the operator.

### void qf_edge_op_get_support(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, uint32_t \*support_out)

Gets the support of an operator, i.e. the mode indices acted upon by the operator. 

The indices are written in **ascending** order, and every index appears exactly once no matter
how many terms act upon it.

### Example

```c
uint32_t left_indices[3] = {2, 0, 2};
uint32_t right_indices[3] = {2, 1, 2};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[3] = {0, 2, 3};
QfEdgeVertexOperator *op =
    qf_edge_op_new(2, 3, coeffs, left_indices, right_indices, boundaries);

uint32_t num_support = qf_edge_op_num_support(op);
uint32_t *support_out = malloc(num_support * sizeof(uint32_t));

qf_edge_op_get_support(op, support_out);

assert(support_out[0] == 0);
assert(support_out[1] == 1);
assert(support_out[2] == 2);

free(support_out);
```

 #### NOTE
Unlike the other getters, this does not hand out a pointer into the operator: the support is
computed on demand rather than stored, so the caller provides the output buffer. Query its
required length with [`qf_edge_op_num_support()`](#c.qf_edge_op_num_support) first.

#### NOTE
Both the left and the right indices of every generalized edge operator contribute to
the support.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose support to get. 
  * **support_out** – A pointer to the integer array into which to write the support. It must be sized to at least :c:func:`[qf_edge_op_num_support](#group__qf__edge__op_1gaea12120b7fc5eec643d9d81b0791fde7)` elements.

### uint32_t qf_edge_op_min_num_modes(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op)

Gets the smallest number of modes that can host an operator. 

This is what determines, for example, the number of qubits a mapping requires. Note that it is
not the number of distinct modes, which [`qf_edge_op_num_support()`](#c.qf_edge_op_num_support) returns.

### Example

```c
uint32_t left_indices[3] = {2, 0, 2};
uint32_t right_indices[3] = {2, 1, 2};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[3] = {0, 2, 3};
QfEdgeVertexOperator *op =
    qf_edge_op_new(2, 3, coeffs, left_indices, right_indices, boundaries);

uint32_t min_num_modes = qf_edge_op_min_num_modes(op);

assert(min_num_modes == 3);
```

 * **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose minimum number of modes to get.
* **Returns:**
  One more than the largest mode index acted upon by the operator, or 0 if it acts on no mode.

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_zero(void)

Constructs the additive identity operator. 

Adding the operator that is constructed by this method to another one has no effect.

### Example

```c
QfEdgeVertexOperator *zero = qf_edge_op_zero();

QfEdgeVertexOperator *op_plus_zero = qf_edge_op_add(op, zero);

assert(qf_edge_op_equal(op, op_plus_zero));
```

 * **Returns:**
  A pointer to the created operator.

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_one(void)

Constructs the multiplicative identity operator. 

Composing the operator that is constructed by this method with another one has no effect.

### Example

```c
QfEdgeVertexOperator *one = qf_edge_op_one();

QfEdgeVertexOperator *op_times_one = qf_edge_op_compose(op, one);

assert(qf_edge_op_equal(op, op_times_one));
```

 * **Returns:**
  A pointer to the created operator.

### bool qf_edge_op_has_groups(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op)

Checks whether an operator has its `groups` attribute set. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfEdgeVertexOperator *op = ...;

bool has_groups = qf_edge_op_has_groups(op);
```

 * **Parameters:**
  * **op** – A pointer to the edge-vertex operator to check.
* **Returns:**
  Whether the operator tracks group indices.

### uint32_t qf_edge_op_num_groups(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op)

Gets the number of groups from an operator. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfEdgeVertexOperator *op = ...;

uint32_t num_groups = qf_edge_op_num_groups(op);
```

 #### NOTE
The number of groups is evaluated lazily as the largest occurring group index plus 1.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose number of groups to get.
* **Returns:**
  The number of group indices from the operator’s `groups` attribute.

### void qf_edge_op_get_groups(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, uint32_t \*\*groups_out, uint64_t \*groups_len)

Gets the group indices for all operator terms. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfEdgeVertexOperator *op = ...;
uint32_t *groups_out;
uint64_t groups_len;

qf_edge_op_get_groups(op, &groups_out, &groups_len);
```

 #### NOTE
This function borrows the operator’s internal buffer rather than copying it. The returned
pointer stays valid only until the operator is modified or freed; do not free it yourself.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose group indices to get. 
  * **groups_out** – A pointer to the integer array into which to write the group indices. 
  * **groups_len** – A pointer to the integer into which to write the length of the output array.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_edge_op_set_groups([QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, const uint32_t \*groups_in, uint64_t groups_len)

Sets the `groups` attribute of the provided operator. 

The length is validated here because nothing downstream re-checks it: too few indices would
silently drop the trailing terms wherever terms are iterated together with their groups, and too
many would make [`qf_edge_op_num_groups()`](#c.qf_edge_op_num_groups) report groups that no term carries.

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfEdgeVertexOperator *op = ...;

uint32_t num_terms = 4;
uint32_t groups_in[4] = {0, 1, 0, 1};
qf_edge_op_set_groups(op, groups_in, num_terms);
```

 * **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose `groups` attribute to write. 
  * **groups_in** – A pointer to the `groups` integer array to write into the operator. 
  * **groups_len** – The number of terms in the `groups_in` array.
* **Returns:**
  An exit code.
  * `[QfExitCode_Success](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a3e14e3ca9bf1cb919c6913131132da37)` upon success
  * `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if `groups_len` differs from the number of terms in the operator

### void qf_edge_op_del_groups([QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op)

Deletes the `groups` attribute from the provided operator. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfEdgeVertexOperator *op = ...;

qf_edge_op_del_groups(op);
```

 * **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose `groups` attribute to delete.

### void qf_edge_op_split_out_groups(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, const uint32_t \*group_indices, uint64_t num_indices, [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*\*group_ops_out)

Splits this operator into a list of new operators based on its `groups` attribute. 

A duplicate index in `group_indices` is written once per occurrence in `group_ops_out`.
Requesting only a small number of groups out of a much larger total is significantly cheaper
than requesting all of them, since terms belonging to a group that is not requested are
skipped rather than appended anywhere.

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfEdgeVertexOperator *op = ...;

uint32_t groups_in[4] = {0, 1, 0, 1};
qf_edge_op_set_groups(op, groups_in, 4);

// build every group, in index order
QfEdgeVertexOperator *group_ops[2];
qf_edge_op_split_out_groups(op, NULL, 0, group_ops);

// build only group 1
uint32_t group_indices[1] = {1};
QfEdgeVertexOperator *group_op[1];
qf_edge_op_split_out_groups(op, group_indices, 1, group_op);
```

 * **Parameters:**
  * **op** – A pointer to the edge-vertex operator whose `groups` to split out. 
  * **group_indices** – A pointer to the array of group indices for which to build operators, in the desired output order. May be `NULL`, in which case every group is built, in index order (equivalent to passing every index from `0` to :c:func:`[qf_edge_op_num_groups](#group__qf__edge__op_1ga2030322b215b9a1bc95942784d9c6337)` `- 1`). 
  * **num_indices** – The number of indices in the `group_indices` array. Ignored if `group_indices` is `NULL`. 
  * **group_ops_out** – A pointer to the array of :c:struct:`QfEdgeVertexOperator` into which to write the operators for each requested group. Must be sized to `num_indices` when `group_indices` is non-`NULL`, or to :c:func:`[qf_edge_op_num_groups](#group__qf__edge__op_1ga2030322b215b9a1bc95942784d9c6337)` when it is `NULL`.

### void qf_edge_op_add_term([QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, uint64_t num_indices, const uint32_t \*left_indices, const uint32_t \*right_indices, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*coeff)

Adds a term to an existing operator. 

Any of the pointer arguments can be `NULL` if and only if their corresponding length is zero.

#### CAUTION
This function resets the operator’s `groups` attribute to `NULL`.

### Example

```c
QfEdgeVertexOperator *one = qf_edge_op_one();

QfEdgeVertexOperator *op = qf_edge_op_zero();
QkComplex64 coeff = {1.0, 0.0};

qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);

assert(qf_edge_op_equal(op, one));
```

 * **Parameters:**
  * **op** – A pointer to the edge-vertex operator to be modified. 
  * **num_indices** – The length of both index arrays. 
  * **left_indices** – A pointer to an array of left-hand mode indices. The length of this array should be `num_indices`. 
  * **right_indices** – A pointer to an array of right-hand mode indices. The length of this array should be `num_indices`. 
  * **coeff** – A pointer to the complex coefficient.

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_add(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*left, const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*right)

Adds two operators together. 

### Example

```c
QfEdgeVertexOperator *one = qf_edge_op_one();
QfEdgeVertexOperator *zero = qf_edge_op_zero();

QfEdgeVertexOperator *result = qf_edge_op_add(one, zero);

assert(qf_edge_op_equal(result, one));
```

 * **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator.
* **Returns:**
  A pointer to the resulting operator.

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_scaled_add(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*left, const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*right, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*factor)

Adds two operators together, scaling the coefficients of the right one. 

This fuses the scaling into the addition, which avoids building a fully scaled copy of `right`
just to append it. Passing a factor of `-1` therefore subtracts, which is why no separate
subtraction function is provided.

### Example

```c
QfEdgeVertexOperator *one = qf_edge_op_one();
QfEdgeVertexOperator *other = qf_edge_op_one();

// Subtracting via a factor of -1 leaves the two terms cancelling each other out.
QkComplex64 factor = {-1.0, 0.0};
QfEdgeVertexOperator *result = qf_edge_op_scaled_add(one, other, &factor);

QfEdgeVertexOperator *simplified = qf_edge_op_simplify(result, 1e-10);
assert(qf_edge_op_len(simplified) == 0);
```

 #### NOTE
Like [`qf_edge_op_add()`](#c.qf_edge_op_add), this appends the terms of `right` without combining them with
those of `left`, so the result tracks no group indices. Call [`qf_edge_op_simplify()`](#c.qf_edge_op_simplify) to
collect equal terms afterwards.

* **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator. 
  * **factor** – A pointer to the factor to scale the right operator’s coefficients with.
* **Returns:**
  A pointer to the resulting operator, `left + factor * right`.

### void qf_edge_op_add_inplace([QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*left, const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*right)

Adds an operator into another one, in place. 

The in-place counterpart of [`qf_edge_op_add()`](#c.qf_edge_op_add), which saves copying `left` into a freshly
allocated result.

#### CAUTION
This function resets the operator’s `groups` attribute to `NULL`.

### Example

```c
QfEdgeVertexOperator *left = qf_edge_op_one();
QfEdgeVertexOperator *right = qf_edge_op_one();

qf_edge_op_add_inplace(left, right);

assert(qf_edge_op_len(left) == 2);
```

 * **Parameters:**
  * **left** – A pointer to the operator to add into. 
  * **right** – A pointer to the operator to add.

### void qf_edge_op_scaled_add_inplace([QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*left, const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*right, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*factor)

Adds an operator into another one in place, scaling the coefficients of the right one. 

The in-place counterpart of [`qf_edge_op_scaled_add()`](#c.qf_edge_op_scaled_add), computing `left + factor * right`
without allocating a result. As there, a factor of `-1` subtracts.

#### CAUTION
This function resets the operator’s `groups` attribute to `NULL`.

### Example

```c
QfEdgeVertexOperator *left = qf_edge_op_one();
QfEdgeVertexOperator *right = qf_edge_op_one();

// Subtract, leaving two terms that cancel each other out.
QkComplex64 factor = {-1.0, 0.0};
qf_edge_op_scaled_add_inplace(left, right, &factor);

QfEdgeVertexOperator *simplified = qf_edge_op_simplify(left, 1e-10);
assert(qf_edge_op_len(simplified) == 0);
```

 * **Parameters:**
  * **left** – A pointer to the operator to add into. 
  * **right** – A pointer to the operator to add. 
  * **factor** – A pointer to the factor to scale the right operator’s coefficients with.

### void qf_edge_op_mul_inplace([QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*scalar)

Multiplies an operator by a scalar, in place. 

The in-place counterpart of [`qf_edge_op_mul()`](#c.qf_edge_op_mul), which saves copying the operator into a
freshly allocated result.

### Example

```c
QfEdgeVertexOperator *op = qf_edge_op_one();
QkComplex64 scalar = {2.0, 0.0};

qf_edge_op_mul_inplace(op, &scalar);
```

 #### NOTE
Unlike the two in-place additions above, this **preserves** the `groups` attribute: scaling
the coefficients leaves the number of terms, and hence the one-index-per-term invariant,
untouched.

* **Parameters:**
  * **op** – A pointer to the operator to scale. 
  * **scalar** – A pointer to the scalar.

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_mul(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*scalar)

Multiplies an operator by a scalar. 

### Example

```c
QfEdgeVertexOperator *one = qf_edge_op_one();
QkComplex64 coeff = {2.0, 0.0};
QfEdgeVertexOperator *result = qf_edge_op_mul(one, &coeff);

QfEdgeVertexOperator *expected = qf_edge_op_zero();
qf_edge_op_add_term(expected, 0, NULL, NULL, &coeff);

assert(qf_edge_op_equal(result, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator. 
  * **scalar** – A pointer to the scalar.
* **Returns:**
  A pointer to the resulting operator.

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_compose(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*left, const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*right)

Composes two operators with each other. 

### Example

```c
QfEdgeVertexOperator *one = qf_edge_op_one();
QfEdgeVertexOperator *zero = qf_edge_op_zero();

QfEdgeVertexOperator *result = qf_edge_op_compose(one, zero);

assert(qf_edge_op_equal(result, zero));
```

 #### NOTE
The composition of two operators tracks no groups, even when both operands do.

* **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator.
* **Returns:**
  A pointer to `left.compose(right)`, which equals the operator `result = right @ left` in terms of the matrix multiplication `@`. In other words, `right` is applied first. To obtain `left @ right`, swap the arguments.

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_adjoint(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op)

Returns the Hermitian conjugate (or adjoint) of an operator. 

This affects the terms and coefficients as follows:

* the generators in each term reverse their order
* the coefficients are complex conjugated

### Example

```c
QfEdgeVertexOperator *op = qf_edge_op_zero();
QkComplex64 coeff = {0.0, 1.0};
qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);

QfEdgeVertexOperator *adjoint = qf_edge_op_adjoint(op);

QfEdgeVertexOperator *expected = qf_edge_op_zero();
QkComplex64 coeff_adj = {0.0, -1.0};
qf_edge_op_add_term(expected, 0, NULL, NULL, &coeff_adj);

assert(qf_edge_op_equal(adjoint, expected));
```

 #### NOTE
Reversing the operator string is essential: the edge and vertex generators anticommute when
they share exactly one index, so `BA != AB` in general.

* **Parameters:**
  * **op** – A pointer to the operator.
* **Returns:**
  A pointer to the created operator.

### void qf_edge_op_ichop([QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, double atol)

Removes terms whose coefficient magnitude lies below the provided threshold. 

#### CAUTION
This functions truncates coefficients greedily! If the acted upon operator might contain
separate coefficients for duplicate terms consider calling [`qf_edge_op_simplify()`](#c.qf_edge_op_simplify)
instead!

### Example

```c
QfEdgeVertexOperator *op = qf_edge_op_zero();
QkComplex64 coeff = {1e-8, 0.0};
qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);

qf_edge_op_ichop(op, 1e-6);

QfEdgeVertexOperator *expected = qf_edge_op_zero();

assert(qf_edge_op_equal(op, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator. 
  * **atol** – The absolute tolerance for coefficient truncation.

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_simplify(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, double atol)

Returns an equivalent but simplified operator. 

The simplification process first sums all coefficients that belong to equal terms and then
only retains those whose total coefficient exceeds the specified tolerance (just like
[`qf_edge_op_ichop()`](#c.qf_edge_op_ichop)).

When an operator has been arithmetically manipulated or constructed in a way that does not
guarantee unique terms, this method should be called before applying any method that
filters numerically small coefficients to avoid loss of information.

### Example

```c
QfEdgeVertexOperator *op = qf_edge_op_zero();
QkComplex64 coeff = {1e-5, 0.0};
for (int i = 0; i < 100; i++) {
  qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);
}

QfEdgeVertexOperator *canon = qf_edge_op_simplify(op, 1e-4);

assert(qf_edge_op_len(canon) == 1);
```

 #### NOTE
This groups terms by their exact stored index arrays. Two terms that are mathematically
equal but stored with different generator orientations (recall `E_kj = -E_jk`) are not
recognized as duplicates; call [`qf_edge_op_normal_ordered()`](#c.qf_edge_op_normal_ordered) first to bring them into
a common form.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator to be simplified. 
  * **atol** – The absolute tolerance for coefficient truncation.
* **Returns:**
  An equivalent but simplified operator.

### [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*qf_edge_op_normal_ordered(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, bool ascending, bool reduce)

Returns an equivalent operator with normal ordered terms. 

The `ascending` flag fixes the orientation convention of the generalized edge operators:
because $E_{kj} = -E_{jk}$, every edge operator has two representations, and this picks
$j < k$ (`true`) or $j > k$ (`false`), absorbing the sign into the coefficient.
Vertex operators $V_j = E_{jj}$ are unaffected.

The `reduce` flag additionally contracts adjacent generators, which is what makes the result
a genuine canonical form. The rules are $V_j V_j = 1$, $E_{jk} E_{jk} = 1$,
$E_{jk} E_{kj} = -1$, and the *fusion* rule
$E_{ab} E_{bc} = -i E_{ac}$ for distinct $a, b, c$.

#### SEE ALSO
[`qf_transfer_op_normal_ordered()`](qf-transfer-op.html.md#c.qf_transfer_op_normal_ordered), which takes no `ascending` flag because the
transfer operators $T_{jk}$ and $T_{kj}$ are genuinely *different* operators
rather than two representations of one.

### Example

```c
// `E(1,0)` normal-orders to `-E(0,1)` when ascending
QfEdgeVertexOperator *op = qf_edge_op_zero();
uint32_t left[1] = {1};
uint32_t right[1] = {0};
QkComplex64 coeff = {1.0, 0.0};
qf_edge_op_add_term(op, 1, left, right, &coeff);

QfEdgeVertexOperator *ordered = qf_edge_op_normal_ordered(op, true, true);

QfEdgeVertexOperator *expected = qf_edge_op_zero();
uint32_t left_exp[1] = {0};
uint32_t right_exp[1] = {1};
QkComplex64 coeff_exp = {-1.0, 0.0};
qf_edge_op_add_term(expected, 1, left_exp, right_exp, &coeff_exp);

assert(qf_edge_op_equal(ordered, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator. 
  * **ascending** – Whether the indices of each edge operator should ascend or descend. 
  * **reduce** – Whether to contract adjacent generators via the algebra’s identities.
* **Returns:**
  A pointer to the created operator.

### bool qf_edge_op_is_hermitian(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, double atol)

Checks whether an operator is Hermitian. 

### Example

```c
// `V(0) E(0,1)` is *not* Hermitian: the two generators share the index 0 and therefore
// anticommute.
QfEdgeVertexOperator *op = qf_edge_op_zero();
uint32_t left[2] = {0, 0};
uint32_t right[2] = {0, 1};
QkComplex64 coeff = {1.0, 0.0};
qf_edge_op_add_term(op, 2, left, right, &coeff);

assert(!qf_edge_op_is_hermitian(op, 1e-10));
```

 #### NOTE
This check is implemented using [`qf_edge_op_equiv()`](#c.qf_edge_op_equiv) on the
[`qf_edge_op_normal_ordered()`](#c.qf_edge_op_normal_ordered) difference of `op` and its
[`qf_edge_op_adjoint()`](#c.qf_edge_op_adjoint) and [`qf_edge_op_zero()`](#c.qf_edge_op_zero).

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator to be checked. 
  * **atol** – The absolute tolerance upto which coefficients are considered equal.
* **Returns:**
  Whether the provided operator is Hermitian.

### bool qf_edge_op_equal(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*left, const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*right)

Compare two operators for equality. 

Equality in this context means an exact match of the internal data arrays.

### Example

```c
QfEdgeVertexOperator *one = qf_edge_op_one();
QfEdgeVertexOperator *zero = qf_edge_op_zero();

assert(qf_edge_op_equal(one, one));
assert(!qf_edge_op_equal(one, zero));
```

 * **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator.
* **Returns:**
  Whether the two operators are equal.

### bool qf_edge_op_equiv(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*left, const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*right, double atol)

Compare two operators for equivalence. 

Equivalence in this context means approximate equality up to the specified absolute tolerance. To be more precise, this method returns `True`, when all the absolute values of the coefficients in the difference `other - self` are below the specified threshold `atol`.

### Example

```c
QfEdgeVertexOperator *zero = qf_edge_op_zero();

QfEdgeVertexOperator *op = qf_edge_op_zero();
QkComplex64 coeff = {1e-7, 0.0};
qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);

assert(qf_edge_op_equiv(op, zero, 1e-6));
assert(!qf_edge_op_equiv(op, zero, 1e-8));
```

 * **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator. 
  * **atol** – The absolute tolerance for coefficient equivalence.
* **Returns:**
  Whether the two operators are equivalent.

### size_t qf_edge_op_len(const [QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op)

Returns the length (or number of terms) of the provided operator. 

### Example

```c
QfEdgeVertexOperator *op = qf_edge_op_zero();
uint32_t left[2] = {0, 1};
uint32_t right[2] = {1, 2};
QkComplex64 coeff = {1.0, 0.0};
qf_edge_op_add_term(op, 2, left, right, &coeff);

assert(qf_edge_op_len(op) == 1);
```

 * **Parameters:**
  * **op** – A pointer to the edge-vertex operator.
* **Returns:**
  The length (or number of terms) of the operator.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_edge_op_relabel_modes([QfEdgeVertexOperator](#c.QfEdgeVertexOperator) \*op, uint64_t num_modes, const uint32_t \*permutation)

Relabels the modes of the provided operator. 

### Example

```c
QfEdgeVertexOperator *op = qf_edge_op_zero();
uint32_t left[2] = {0, 2};
uint32_t right[2] = {1, 3};
QkComplex64 coeff = {1.0, 0.0};
qf_edge_op_add_term(op, 2, left, right, &coeff);

uint32_t permutation[4] = {3, 2, 1, 0};

QfExitCode exit = qf_edge_op_relabel_modes(op, 4, permutation);

assert(exit == QfExitCode_Success);
```

 #### NOTE
Both index arrays are relabelled. Unlike most operations, this preserves the operator’s
`groups` attribute, since relabelling permutes mode indices without reordering, splitting
or merging terms.

* **Parameters:**
  * **op** – A pointer to the edge-vertex operator. 
  * **num_modes** – The number of mode indices in the provided permutation list. 
  * **permutation** – The index permutation list.
* **Returns:**
  An exit code.
  * `[QfExitCode_Success](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a3e14e3ca9bf1cb919c6913131132da37)` upon success
  * `[QfExitCode_DuplicateIndexError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a01f24cac283f08bcfa375ff2a59772ef)` if duplicate indices were found in the permutation
  * `[QfExitCode_IndexError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1ae5c84289c900faf721fde93e2f249e3b)` for any other index errors, such as invalid indices.
