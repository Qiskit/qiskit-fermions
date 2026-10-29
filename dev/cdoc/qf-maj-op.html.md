# QfMajoranaOperator

### struct QfMajoranaOperator

A Majorana fermion operator.

#### NOTE
This is an opaque data structure to the C API whose internals are implemented entirely in Rust.
The remainder of this page describes the design and related functions to work with this struct.

---

## Definition

This operator is defined by a linear combination of products of Majorana operators <sup>[1](#id2)</sup>, which
can be defined in terms of the standard fermionic second-quantization creation and annihilation
operators (see also [`.QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator)):

$$
\gamma = a^\dagger + a ~~\text{and}~~ \gamma' = i(a^\dagger - a)
$$

The key property that a Majorana fermion is its own antiparticle becomes immediately apparent:

$$
\gamma_i = \gamma_i^\dagger ~~\text{and}~~ \gamma_i^2 = (\gamma_i^\dagger)^2 = 1
$$

This result in the following anti-commutation relations for $2n$ Majorana fermions:

$$
\left\{\gamma_i,\gamma_j\right\} = 2\delta_{ij}
$$

This makes the definition of the entire operator the following:

$$
\text{\texttt{MajoranaOperator}} = \sum_i c_i \bigotimes_j \hat{\gamma_j} \, ,
$$

where $c_i$ is the (complex) coefficient making up the linear combination of products of
$\gamma_j$. The index $j$ can take any value between 0 and the number of majorana
fermionic modes acted upon by the operator minus 1.

---

<a id="qf-maj-op-implementation"></a>

## Implementation

This struct stores the terms and coefficients in multiple sparse vectors, akin to the
[compressed sparse row format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
commonly used for sparse matrices. More concretely, a single operator contains 4 arrays:

| `coeffs`     | A vector of complex coefficients consisting of two 64-bit floating point numbers.   |
|--------------|-------------------------------------------------------------------------------------|
| `modes`      | A vector of 32-bit integers storing the majorana mode indices acted upon.           |
| `boundaries` | A vector of integers indicating the boundaries in `actions` and `indices`.          |

The integers in `modes` index the Majorana modes, $j$. When using the convenience
function [`gamma()`](../stubs/qiskit_fermions.operators.gamma.html.md#qiskit_fermions.operators.gamma), even (odd) indices are used for $\gamma$ ($\gamma'$).

#### NOTE
You can access **read-only copies** of these internal arrays via their respective functions:

- [`qf_maj_op_get_coeffs()`](#c.qf_maj_op_get_coeffs)
- [`qf_maj_op_get_modes()`](#c.qf_maj_op_get_modes)
- [`qf_maj_op_get_boundaries()`](#c.qf_maj_op_get_boundaries)

This data structure allows for very efficient construction and manipulation of operators.
However, it implies that duplicate terms might be contained in an operator at any moment.
These must be resolved manually through the use of [`qf_maj_op_simplify()`](#c.qf_maj_op_simplify).

### Construction

A new operator can be constructed directly by specifying the corresponding arrays outlined above.
Alternatively, an empty [`QfMajoranaOperator`](#c.QfMajoranaOperator) can be initialized with
[`qf_maj_op_zero()`](#c.qf_maj_op_zero) and terms can be added iteratively via [`qf_maj_op_add_term()`](#c.qf_maj_op_add_term).

| [`qf_maj_op_new()`](#c.qf_maj_op_new)      | Constructs a new operator from the provided arrays.   |
|-----------------------------------------------------------------------|-------------------------------------------------------|
| [`qf_maj_op_zero()`](#c.qf_maj_op_zero)     | Constructs the additive identity operator.            |
| [`qf_maj_op_one()`](#c.qf_maj_op_one)      | Constructs the multiplicative identity operator.      |
| [`qf_maj_op_add_term()`](#c.qf_maj_op_add_term) | Adds a term to an existing `QfMajoranaOperator`.      |

#### NOTE
A [`QfMajoranaOperator`](#c.QfMajoranaOperator) can be freed with [`qf_maj_op_free()`](#c.qf_maj_op_free).

### Arithmetics

The following functions provide arithmetic manipulation:

| [`qf_maj_op_add()`](#c.qf_maj_op_add)     | Adds two operators together.              |
|----------------------------------------------------------------------|-------------------------------------------|
| [`qf_maj_op_mul()`](#c.qf_maj_op_mul)     | Multiplies an operator by a scalar.       |
| [`qf_maj_op_compose()`](#c.qf_maj_op_compose) | Composes two operators with each other.   |
| [`qf_maj_op_adjoint()`](#c.qf_maj_op_adjoint) | Returns the Hermitian conjugate operator. |

### Manipulation

The following functions provide operator manipulation logic:

| [`qf_maj_op_ichop()`](#c.qf_maj_op_ichop)          | Removes terms with small coefficient magnitudes.          |
|-----------------------------------------------------------------------------|-----------------------------------------------------------|
| [`qf_maj_op_simplify()`](#c.qf_maj_op_simplify)       | Returns an equivalent but simplified operator.            |
| [`qf_maj_op_normal_ordered()`](#c.qf_maj_op_normal_ordered) | Returns an equivalent operator with normal ordered terms. |
| [`qf_maj_op_relabel_modes()`](#c.qf_maj_op_relabel_modes)  | Relabels the modes of an operator.                        |

### Properties

The following functions exist to check certain properties of an operator.

| [`qf_maj_op_is_hermitian()`](#c.qf_maj_op_is_hermitian)   | Returns whether an operator is Hermitian.               |
|-----------------------------------------------------------------------------|---------------------------------------------------------|
| [`qf_maj_op_max_rank()`](#c.qf_maj_op_max_rank)       | Returns the maximum rank of the terms in this operator. |
| [`qf_maj_op_min_num_modes()`](#c.qf_maj_op_min_num_modes)  | Returns the minimum number of modes of this operator.   |
| [`qf_maj_op_is_even()`](#c.qf_maj_op_is_even)        | Returns whether an operator is even.                    |

---
* <a id='id2'>**[1]**</a> [https://en.wikipedia.org/wiki/Majorana_fermion](https://en.wikipedia.org/wiki/Majorana_fermion)

## Members

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_new(uint64_t num_terms, uint64_t num_modes, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*coeffs, const uint32_t \*modes, const uint32_t \*boundaries)

Constructs a new operator. 

Any of the pointer arguments can be `NULL` if and only if their corresponding length is zero.

### Example

```c
uint64_t num_terms = 3;
uint64_t num_modes = 4;
uint32_t modes[4] = {0, 1, 2, 3};
QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[4] = {0, 0, 2, 4};
QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs,
                                       modes, boundaries);
```

 * **Parameters:**
  * **num_terms** – The number of terms in the operator. 
  * **num_modes** – The number of modes summed over all terms. 
  * **coeffs** – A pointer to an array of term coefficients. The length of this array should be `num_terms`. 
  * **modes** – A pointer to an array of modes over all terms. The length of this array should be `num_modes`. 
  * **boundaries** – A pointer to an array of the boundaries between terms. The length of this array should be `num_terms + 1`.

### void qf_maj_op_free([QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Frees an existing operator. 

### Example

```c
QfMajoranaOperator *op = qf_maj_op_one();
qf_maj_op_free(op);
```

 * **Parameters:**
  * **op** – A pointer to the Majorana operator to be freed.

### void qf_maj_op_get_coeffs(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*\*coeffs_out, uint64_t \*coeffs_len)

Provides read-only access to the operator’s coefficients. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-maj-op-implementation).

### Example

```c
uint64_t num_terms = 2;
uint64_t num_modes = 0;
QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[3] = {0, 0, 0};
QfMajoranaOperator *op =
    qf_maj_op_new(num_terms, num_modes, coeffs, NULL, boundaries);

QkComplex64 *coeffs_out;
uint64_t *coeffs_len;

qf_maj_op_get_coeffs(op, &coeffs_out, &coeffs_len);

assert(coeffs_len == 2);
```

 #### NOTE
This function returns a **copy** of the internal data.

* **Parameters:**
  * **op** – A pointer to the majorana operator whose coefficients to access. 
  * **coeffs_out** – A pointer to the array of complex values into which to write the coefficients. 
  * **coeffs_len** – A pointer to the integer into which to write the length of the output array.

### void qf_maj_op_get_modes(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, uint32_t \*\*modes_out, uint64_t \*modes_len)

Provides read-only access to the operator’s acted-upon mode indices. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-maj-op-implementation).

### Example

```c
uint64_t num_terms = 2;
bool actions[2] = {true, false};
uint32_t modes[2] = {0, 1};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[3] = {0, 0, 2};
QfMajoranaOperator *op =
    qf_maj_op_new(num_terms, num_actions, coeffs, modes, boundaries);

QkComplex64 *modes_out;
uint64_t *modes_len;

qf_maj_op_get_modes(op, &modes_out, &modes_len);

assert(modes_len == 2);
```

 #### NOTE
This function returns a **copy** of the internal data.

* **Parameters:**
  * **op** – A pointer to the majorana operator whose modes to access. 
  * **modes_out** – A pointer to the array of boolean values into which to write the modes. 
  * **modes_len** – A pointer to the integer into which to write the length of the output array.

### void qf_maj_op_get_boundaries(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, size_t \*\*boundaries_out, uint64_t \*boundaries_len)

Provides read-only access to the indices indicating the boundaries between operator terms. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-maj-op-implementation).

### Example

```c
uint64_t num_terms = 2;
uint64_t num_modes = 2;
uint32_t modes[2] = {0, 1};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[3] = {0, 0, 2};
QfMajoranaOperator *op =
    qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

QkComplex64 *boundaries_out;
uint64_t *boundaries_len;

qf_maj_op_get_boundaries(op, &boundaries_out, &boundaries_len);

assert(boundaries_len == 3);
```

 #### NOTE
This function returns a **copy** of the internal data.

* **Parameters:**
  * **op** – A pointer to the majorana operator whose boundaries to access. 
  * **boundaries_out** – A pointer to the array of boolean values into which to write the boundaries. 
  * **boundaries_len** – A pointer to the integer into which to write the length of the output array.

### uint32_t qf_maj_op_num_support(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Gets the size of the support of an operator. 

Use this to size the output buffer of [`qf_maj_op_get_support()`](#c.qf_maj_op_get_support), which does not report a
length of its own.

### Example

```c
uint32_t modes[3] = {2, 0, 2};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[3] = {0, 2, 3};
QfMajoranaOperator *op = qf_maj_op_new(2, 3, coeffs, modes, boundaries);

uint32_t num_support = qf_maj_op_num_support(op);

assert(num_support == 2);
```

 * **Parameters:**
  * **op** – A pointer to the Majorana operator whose support size to get.
* **Returns:**
  The number of distinct mode indices acted upon by the operator.

### void qf_maj_op_get_support(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, uint32_t \*support_out)

Gets the support of an operator, i.e. the Majorana mode indices acted upon by the operator. 

The indices are written in **ascending** order, and every index appears exactly once no matter
how many terms act upon it.

### Example

```c
uint32_t modes[3] = {2, 0, 2};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[3] = {0, 2, 3};
QfMajoranaOperator *op = qf_maj_op_new(2, 3, coeffs, modes, boundaries);

uint32_t num_support = qf_maj_op_num_support(op);
uint32_t *support_out = malloc(num_support * sizeof(uint32_t));

qf_maj_op_get_support(op, support_out);

assert(support_out[0] == 0);
assert(support_out[1] == 2);

free(support_out);
```

 #### NOTE
Unlike the other getters, this does not hand out a pointer into the operator: the support is
computed on demand rather than stored, so the caller provides the output buffer. Query its
required length with [`qf_maj_op_num_support()`](#c.qf_maj_op_num_support) first.

#### NOTE
These are the *Majorana* mode indices, not the fermionic ones: they are not divided
by two. A Majorana operator on fermionic mode `j` therefore reports the indices
`2 * j` and/or `2 * j + 1`.

* **Parameters:**
  * **op** – A pointer to the Majorana operator whose support to get. 
  * **support_out** – A pointer to the integer array into which to write the support. It must be sized to at least :c:func:`[qf_maj_op_num_support](#group__qf__maj__op_1gad63df29c4932342caac0842fff4964eb)` elements.

### uint32_t qf_maj_op_min_num_modes(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Gets the smallest number of fermionic modes that can host an operator. 

Majorana index `m` belongs to fermionic mode `m / 2`, so this is `max_index / 2 + 1` and
counts fermionic modes (which is what determines, for example, the number of qubits a mapping
requires), not Majorana indices.

### Example

```c
uint32_t modes[3] = {2, 0, 2};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[3] = {0, 2, 3};
QfMajoranaOperator *op = qf_maj_op_new(2, 3, coeffs, modes, boundaries);

uint32_t min_num_modes = qf_maj_op_min_num_modes(op);

assert(min_num_modes == 2);
```

 * **Parameters:**
  * **op** – A pointer to the Majorana operator whose minimum number of modes to get.
* **Returns:**
  One more than the largest fermionic mode acted upon by the operator, or 0 if it acts on no mode.

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_zero(void)

Constructs the additive identity operator. 

Adding the operator that is constructed by this method to another one has no effect.

### Example

```c
QfMajoranaOperator *zero = qf_maj_op_zero();

QfMajoranaOperator *op_plus_zero = qf_maj_op_add(op, zero);

assert(qf_maj_op_equal(op, op_plus_zero));
```

 * **Returns:**
  A pointer to the created operator.

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_one(void)

Constructs the multiplicative identity operator. 

Composing the operator that is constructed by this method with another one has no effect.

### Example

```c
QfMajoranaOperator *one = qf_maj_op_one();

QfMajoranaOperator *op_times_one = qf_maj_op_compose(op, one);

assert(qf_maj_op_equal(op, op_times_one));
```

 * **Returns:**
  A pointer to the created operator.

### bool qf_maj_op_has_groups(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Checks whether this operator tracks group indices. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfMajoranaOperator *op = ...;

bool has_groups = qf_maj_op_has_groups(op);
```

 * **Parameters:**
  * **op** – A pointer to the majorana operator to be checked.
* **Returns:**
  Whether the provided operator has a `groups` attribute.

### uint32_t qf_maj_op_num_groups(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Gets the number of groups from an operator. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfMajoranaOperator *op = ...;

uint32_t num_groups = qf_maj_op_num_groups(op);
```

 #### NOTE
The number of groups is evaluated lazily as the largest occurring group index plus 1.

* **Parameters:**
  * **op** – A pointer to the majorana operator whose number of groups to get.
* **Returns:**
  The number of group indices from the operator’s `groups` attribute.

### void qf_maj_op_get_groups(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, uint32_t \*\*groups_out, uint64_t \*groups_len)

Gets the group indices for all operator terms. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfMajoranaOperator *op = ...;
uint32_t *groups_out;
uint32_t groups_len;

qf_maj_op_get_groups(op, &groups_out, &groups_len);
```

 * **Parameters:**
  * **op** – A pointer to the majorana operator whose group indices to get. 
  * **groups_out** – A pointer to the integer array into which to write the group indices. 
  * **groups_len** – A pointer to the integer into which to write the length of the output array.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_maj_op_set_groups([QfMajoranaOperator](#c.QfMajoranaOperator) \*op, const uint32_t \*groups_in, uint64_t groups_len)

Sets the `groups` attribute of the provided operator. 

The length is validated here because nothing downstream re-checks it: too few indices would
silently drop the trailing terms wherever terms are iterated together with their groups, and too
many would make [`qf_maj_op_num_groups()`](#c.qf_maj_op_num_groups) report groups that no term carries.

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfMajoranaOperator *op = ...;

uint32_t num_terms = 4;
uint32_t groups_in[4] = {0, 1, 0, 1};
qf_maj_op_set_groups(op, groups_in, num_terms);
```

 * **Parameters:**
  * **op** – A pointer to the majorana operator whose `groups` attribute to write. 
  * **groups_in** – A pointer to the `groups` integer array to write into the operator. 
  * **groups_len** – The number of terms in the `groups_in` array.
* **Returns:**
  An exit code.
  * `[QfExitCode_Success](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a3e14e3ca9bf1cb919c6913131132da37)` upon success
  * `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if `groups_len` differs from the number of terms in the operator

### void qf_maj_op_del_groups([QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Deletes the `groups` attribute from the provided operator. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfMajoranaOperator *op = ...;

qf_maj_op_del_groups(op);
```

 * **Parameters:**
  * **op** – A pointer to the majorana operator whose `groups` attribute to delete.

### void qf_maj_op_split_out_groups(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, const uint32_t \*group_indices, uint64_t num_indices, [QfMajoranaOperator](#c.QfMajoranaOperator) \*\*group_ops_out)

Splits this operator into a list of new operators based on its `groups` attribute. 

A duplicate index in `group_indices` is written once per occurrence in `group_ops_out`.
Requesting only a small number of groups out of a much larger total is significantly cheaper
than requesting all of them, since terms belonging to a group that is not requested are
skipped rather than appended anywhere.

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
uint64_t num_terms = 4;
uint64_t num_modes = 8;
uint32_t modes[8] = {0, 1, 2, 3, 1, 0, 3, 2};
QkComplex64 coeffs[4] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[5] = {0, 2, 4, 6, 8};
QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

uint32_t groups_in[4] = {0, 1, 0, 1};
qf_maj_op_set_groups(op, groups_in, num_terms);

// build every group, in index order
QfMajoranaOperator *group_ops[2];
qf_maj_op_split_out_groups(op, NULL, 0, group_ops);

// build only group 1
uint32_t group_indices[1] = {1};
QfMajoranaOperator *group_op[1];
qf_maj_op_split_out_groups(op, group_indices, 1, group_op);
```

 * **Parameters:**
  * **op** – A pointer to the majorana operator whose `groups` to split out. 
  * **group_indices** – A pointer to the array of group indices for which to build operators, in the desired output order. May be `NULL`, in which case every group is built, in index order (equivalent to passing every index from `0` to :c:func:`[qf_maj_op_num_groups](#group__qf__maj__op_1ga28217dc17b8b482364bfa9c2519ff37f)` `- 1`). 
  * **num_indices** – The number of indices in the `group_indices` array. Ignored if `group_indices` is `NULL`. 
  * **group_ops_out** – A pointer to the array of :c:struct:`QfMajoranaOperator` into which to write the operators for each requested group. Must be sized to `num_indices` when `group_indices` is non-`NULL`, or to :c:func:`[qf_maj_op_num_groups](#group__qf__maj__op_1ga28217dc17b8b482364bfa9c2519ff37f)` when it is `NULL`.

### void qf_maj_op_add_term([QfMajoranaOperator](#c.QfMajoranaOperator) \*op, uint64_t num_modes, const uint32_t \*modes, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*coeff)

Adds a term to an existing operator. 

Any of the pointer arguments can be `NULL` if and only if their corresponding length is zero.

#### CAUTION
This function resets the operator’s `groups` attribute to `NULL`.

### Example

```c
QfMajoranaOperator *one = qf_maj_op_one();

QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes[0] = {};
QkComplex64 coeff = {1.0, 0.0};

qf_maj_op_add_term(op, 0, modes, &coeff);

assert(qf_maj_op_equal(op, one));
```

 * **Parameters:**
  * **op** – A pointer to the Majorana operator to be modified. 
  * **num_modes** – The length of the modes array. 
  * **modes** – A pointer to an array of mode indices. The length of this array should be `num_modes`. 
  * **coeff** – A pointer to the complex coefficient.

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_add(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*left, const [QfMajoranaOperator](#c.QfMajoranaOperator) \*right)

Adds two operators together. 

### Example

```c
QfMajoranaOperator *one = qf_maj_op_one();
QfMajoranaOperator *zero = qf_maj_op_zero();

QfMajoranaOperator *result = qf_maj_op_add(one, zero);

assert(qf_maj_op_equal(result, one));
```

 * **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator.
* **Returns:**
  A pointer to the resulting operator.

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_scaled_add(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*left, const [QfMajoranaOperator](#c.QfMajoranaOperator) \*right, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*factor)

Adds two operators together, scaling the coefficients of the right one. 

This fuses the scaling into the addition, which avoids building a fully scaled copy of `right`
just to append it. Passing a factor of `-1` therefore subtracts, which is why no separate
subtraction function is provided.

### Example

```c
QfMajoranaOperator *one = qf_maj_op_one();
QfMajoranaOperator *other = qf_maj_op_one();

// Subtracting via a factor of -1 leaves the two terms cancelling each other out.
QkComplex64 factor = {-1.0, 0.0};
QfMajoranaOperator *result = qf_maj_op_scaled_add(one, other, &factor);

QfMajoranaOperator *simplified = qf_maj_op_simplify(result, 1e-10);
assert(qf_maj_op_len(simplified) == 0);
```

 #### NOTE
Like [`qf_maj_op_add()`](#c.qf_maj_op_add), this appends the terms of `right` without combining them with
those of `left`, so the result tracks no group indices. Call [`qf_maj_op_simplify()`](#c.qf_maj_op_simplify) to
collect equal terms afterwards.

* **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator. 
  * **factor** – A pointer to the factor to scale the right operator’s coefficients with.
* **Returns:**
  A pointer to the resulting operator, `left + factor * right`.

### void qf_maj_op_add_inplace([QfMajoranaOperator](#c.QfMajoranaOperator) \*left, const [QfMajoranaOperator](#c.QfMajoranaOperator) \*right)

Adds an operator into another one, in place. 

The in-place counterpart of [`qf_maj_op_add()`](#c.qf_maj_op_add), which saves copying `left` into a freshly
allocated result.

#### CAUTION
This function resets the operator’s `groups` attribute to `NULL`.

### Example

```c
QfMajoranaOperator *left = qf_maj_op_one();
QfMajoranaOperator *right = qf_maj_op_one();

qf_maj_op_add_inplace(left, right);

assert(qf_maj_op_len(left) == 2);
```

 * **Parameters:**
  * **left** – A pointer to the operator to add into. 
  * **right** – A pointer to the operator to add.

### void qf_maj_op_scaled_add_inplace([QfMajoranaOperator](#c.QfMajoranaOperator) \*left, const [QfMajoranaOperator](#c.QfMajoranaOperator) \*right, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*factor)

Adds an operator into another one in place, scaling the coefficients of the right one. 

The in-place counterpart of [`qf_maj_op_scaled_add()`](#c.qf_maj_op_scaled_add), computing `left + factor * right`
without allocating a result. As there, a factor of `-1` subtracts.

#### CAUTION
This function resets the operator’s `groups` attribute to `NULL`.

### Example

```c
QfMajoranaOperator *left = qf_maj_op_one();
QfMajoranaOperator *right = qf_maj_op_one();

// Subtract, leaving two terms that cancel each other out.
QkComplex64 factor = {-1.0, 0.0};
qf_maj_op_scaled_add_inplace(left, right, &factor);

QfMajoranaOperator *simplified = qf_maj_op_simplify(left, 1e-10);
assert(qf_maj_op_len(simplified) == 0);
```

 * **Parameters:**
  * **left** – A pointer to the operator to add into. 
  * **right** – A pointer to the operator to add. 
  * **factor** – A pointer to the factor to scale the right operator’s coefficients with.

### void qf_maj_op_mul_inplace([QfMajoranaOperator](#c.QfMajoranaOperator) \*op, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*scalar)

Multiplies an operator by a scalar, in place. 

The in-place counterpart of [`qf_maj_op_mul()`](#c.qf_maj_op_mul), which saves copying the operator into a
freshly allocated result.

### Example

```c
QfMajoranaOperator *op = qf_maj_op_one();
QkComplex64 scalar = {2.0, 0.0};

qf_maj_op_mul_inplace(op, &scalar);
```

 #### NOTE
Unlike the two in-place additions above, this **preserves** the `groups` attribute: scaling
the coefficients leaves the number of terms, and hence the one-index-per-term invariant,
untouched.

* **Parameters:**
  * **op** – A pointer to the operator to scale. 
  * **scalar** – A pointer to the scalar.

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_mul(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*scalar)

Multiplies an operator by a scalar. 

### Example

```c
QfMajoranaOperator *one = qf_maj_op_one();
QkComplex64 coeff = {2.0, 0.0};
QfMajoranaOperator *result = qf_maj_op_mul(one, &coeff);

QfMajoranaOperator *expected = qf_maj_op_zero();
uint32_t modes[0] = {};
qf_maj_op_add_term(expected, 0, modes, &coeff);

assert(qf_maj_op_equal(result, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator. 
  * **scalar** – A pointer to the scalar.
* **Returns:**
  A pointer to the resulting operator.

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_compose(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*left, const [QfMajoranaOperator](#c.QfMajoranaOperator) \*right)

Composes two operators with each other. 

### Example

```c
QfMajoranaOperator *one = qf_maj_op_one();
QfMajoranaOperator *zero = qf_maj_op_zero();

QfMajoranaOperator *result = qf_maj_op_compose(one, zero);

assert(qf_maj_op_equal(result, zero));
```

 #### NOTE
The composition of two operators tracks no groups, even when both operands do.

* **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator.
* **Returns:**
  A pointer to `left.compose(right)`, which equals the operator `result = right @ left` in terms of the matrix multiplication `@`. In other words, `right` is applied first. To obtain `left @ right`, swap the arguments.

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_adjoint(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Returns the Hermitian conjugate (or adjoint) of an operator. 

This affects the terms and coefficients as follows:

* the actions in each term reverse their order
* the coefficients are complex conjugated

### Example

```c
QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes[0] = {};
QkComplex64 coeff = {0.0, 1.0};
qf_maj_op_add_term(op, 0, modes, &coeff);

QfMajoranaOperator *adjoint = qf_maj_op_adjoint(op);

QfMajoranaOperator *expected = qf_maj_op_zero();
QkComplex64 coeff_adj = {0.0, -1.0};
qf_maj_op_add_term(expected, 0, modes, &coeff_adj);

assert(qf_maj_op_equal(adjoint, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator.
* **Returns:**
  A pointer to the created operator.

### void qf_maj_op_ichop([QfMajoranaOperator](#c.QfMajoranaOperator) \*op, double atol)

Removes terms whose coefficient magnitude lies below the provided threshold. 

#### CAUTION
This functions truncates coefficients greedily! If the acted upon operator might contain
separate coefficients for duplicate terms consider calling [`qf_maj_op_simplify()`](#c.qf_maj_op_simplify)
instead!

### Example

```c
QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes[0] = {};
QkComplex64 coeff = {1e-8};
qf_maj_op_add_term(op, 0, modes, &coeff);

qf_maj_op_ichop(op, 1e-6);

QfMajoranaOperator *expected = qf_maj_op_zero();

assert(qf_maj_op_equal(op, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator. 
  * **atol** – The absolute tolerance for coefficient truncation.

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_simplify(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, double atol)

Returns an equivalent but simplified operator. 

The simplification process first sums all coefficients that belong to equal terms and then
only retains those whose total coefficient exceeds the specified tolerance (just like
[`qf_maj_op_ichop()`](#c.qf_maj_op_ichop)).

When an operator has been arithmetically manipulated or constructed in a way that does not
guarantee unique terms, this method should be called before applying any method that
filters numerically small coefficients to avoid loss of information. See the example below
which showcases how [`qf_maj_op_ichop()`](#c.qf_maj_op_ichop) can truncate terms that sum to a total
coefficient magnitude which should not be truncated:

```c
uint64_t num_terms = 100000;
uint64_t num_modes = 0;
uint32_t modes[0] = {};
QkComplex64 coeffs[100000];
uint32_t boundaries[100001];
for (int i = 0; i < 100000; i++) {
  coeffs[i].re = 1e-5;
  coeffs[i].im = 0.0;
  boundaries[i] = 0;
}
boundaries[100000] = 0;
QfMajoranaOperator *op =
    qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

QfMajoranaOperator *canon = qf_maj_op_simplify(op, 1e-4);

QfMajoranaOperator *one = qf_maj_op_one();
bool canon_is_equal = qf_maj_op_equiv(canon, one, 1e-6);

qf_maj_op_ichop(op, 1e-4);

QfMajoranaOperator *zero = qf_maj_op_zero();
bool ichop_is_equal = qf_maj_op_equiv(op, zero, 1e-6);
```

 * **Parameters:**
  * **op** – A pointer to the Majorana operator to be simplified. 
  * **atol** – The absolute tolerance for coefficient truncation.
* **Returns:**
  An equivalent but simplified operator.

### [QfMajoranaOperator](#c.QfMajoranaOperator) \*qf_maj_op_normal_ordered(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, bool ascending, bool reduce)

Returns an equivalent operator with normal ordered terms. 

The normal order of an operator term is defined such that all actions are ordered lexicographically. Whether they ascend or descend depends on the value of the `ascending` parameter.

### Example

```c
QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes[4] = {0, 2, 1, 3};
QkComplex64 coeff = {1.0, 0.0};
qf_maj_op_add_term(op, 4, modes, &coeff);

QfMajoranaOperator *normal_ordered = qf_maj_op_normal_ordered(op, false, true);

QkComplex64 coeff_minus = {-1.0, 0.0};
QfMajoranaOperator *expected = qf_maj_op_zero();
uint32_t modes_exp[4] = {3, 2, 1, 0};
qf_maj_op_add_term(expected, 4, modes_exp, &coeff_minus);

assert(qf_maj_op_equal(normal_ordered, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator. 
  * **ascending** – Whether indices should ascend or descend. 
  * **reduce** – Whether to reduce each term to its minimal form by removing actions that square to the identity.
* **Returns:**
  A pointer to the created operator.

### bool qf_maj_op_is_hermitian(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op, double atol)

Checks whether an operator is Hermitian. 

### Example

```c
QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes1[2] = {0, 1};
QkComplex64 coeff1 = {0.0, 1.00001};
qf_maj_op_add_term(op, 2, modes1, &coeff1);
uint32_t modes2[2] = {0, 1};
QkComplex64 coeff2 = {0.0, -1};
qf_maj_op_add_term(op, 2, modes2, &coeff2);

assert(qf_maj_op_is_hermitian(op, 1e-4));
assert(!qf_maj_op_is_hermitian(op, 1e-8));
```

 #### NOTE
This check is implemented using [`qf_maj_op_equiv()`](#c.qf_maj_op_equiv) on the
[`qf_maj_op_normal_ordered()`](#c.qf_maj_op_normal_ordered) difference of `op` and its
[`qf_maj_op_adjoint()`](#c.qf_maj_op_adjoint) and [`qf_maj_op_zero()`](#c.qf_maj_op_zero).

* **Parameters:**
  * **op** – A pointer to the Majorana operator to be checked. 
  * **atol** – The absolute tolerance upto which coefficients are considered equal.
* **Returns:**
  Whether the provided operator is Hermitian.

### uint32_t qf_maj_op_max_rank(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Checks the maximum rank of an operator. 

### Example

```c
QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes[4] = {0, 1, 2, 3};
QkComplex64 coeff = {1.0, 0.0};
qf_maj_op_add_term(op, 4, modes, &coeff);

assert(qf_maj_op_max_rank(op), 4);
```

 #### NOTE
The length of the longest term can depend on the operator’s form which means that (for
example) operator simplification or normal-ordering can result in a different maximum rank.

* **Parameters:**
  * **op** – A pointer to the Majorana operator to be checked.
* **Returns:**
  The maximum rank of the operator.

### bool qf_maj_op_is_even(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Checks whether an operator is even. 

### Example

```c
QfMajoranaOperator *op = qf_maj_op_zero();
QkComplex64 coeff = {1.0, 0.0};
uint32_t modes1[2] = {0, 1};
qf_maj_op_add_term(op, 2, modes1, &coeff);

assert(qf_maj_op_is_even(op));

uint32_t modes2[1] = {2};
qf_maj_op_add_term(op, 2, modes2, &coeff);

assert(!qf_maj_op_is_even(op));
```

 #### NOTE
An operator is considered even when all of its terms contain an even number of actions.

* **Parameters:**
  * **op** – A pointer to the Majorana operator to be checked.
* **Returns:**
  Whether the provided operator is even.

### bool qf_maj_op_equal(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*left, const [QfMajoranaOperator](#c.QfMajoranaOperator) \*right)

Compare two operators for equality. 

### Example

```c
QfMajoranaOperator *one = qf_maj_op_one();
QfMajoranaOperator *zero = qf_maj_op_zero();

assert(qf_maj_op_equal(one, one));
assert(!qf_maj_op_equal(one, zero));
```

 * **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator.
* **Returns:**
  Whether the two operators are equal.

### bool qf_maj_op_equiv(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*left, const [QfMajoranaOperator](#c.QfMajoranaOperator) \*right, double atol)

Compare two operators for equivalence. 

Equivalence in this context means approximate equality up to the specified absolute tolerance. To be more precise, this method returns `True`, when all the absolute values of the coefficients in the difference `other - self` are below the specified threshold `atol`.

### Example

```c
QfMajoranaOperator *zero = qf_maj_op_zero();

QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes[0] = {};
QkComplex64 coeff = {1e-7, 0.0};
qf_maj_op_add_term(op, 0, modes, &coeff);

assert(qf_maj_op_equiv(op, zero, 1e-6));
assert(!qf_maj_op_equiv(op, zero, 1e-8));
```

 * **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator. 
  * **atol** – The absolute tolerance for coefficient equivalence.
* **Returns:**
  Whether the two operators are equivalent.

### size_t qf_maj_op_len(const [QfMajoranaOperator](#c.QfMajoranaOperator) \*op)

Returns the length (or number of terms) of the provided operator. 

### Example

```c
QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes[4] = {0, 1, 2, 3};
QkComplex64 coeff = {1.0, 0.0};
qf_maj_op_add_term(op, 4, modes, &coeff);

assert(qf_maj_op_len(op) == 1);
```

 * **Parameters:**
  * **op** – A pointer to the Majorana operator.
* **Returns:**
  The length (or number of terms) of the operator.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_maj_op_relabel_modes([QfMajoranaOperator](#c.QfMajoranaOperator) \*op, uint64_t num_modes, const uint32_t \*permutation)

Relabels the modes of the provided operator. 

### Example

```c
QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes[4] = {0, 1, 2, 3};
QkComplex64 coeff = {1.0, 0.0};
qf_maj_op_add_term(op, 4, modes, &coeff);

uint32_t permutation[4] = {3, 2, 1, 0};

QfExitCode exit = qf_maj_op_relabel_modes(op, 4, permutation);

assert(exit == QfExitCode_Success);
```

 * **Parameters:**
  * **op** – A pointer to the Majorana operator. 
  * **num_modes** – The number of mode indices in the provided permutation list. 
  * **permutation** – The index permutation list.
* **Returns:**
  An exit code.
  * `[QfExitCode_Success](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a3e14e3ca9bf1cb919c6913131132da37)` upon success
  * `[QfExitCode_DuplicateIndexError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a01f24cac283f08bcfa375ff2a59772ef)` if duplicate indices were found in the permutation
  * `[QfExitCode_IndexError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1ae5c84289c900faf721fde93e2f249e3b)` for any other index errors, such as invalid indices.
