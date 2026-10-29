# QfFermionOperator

### struct QfFermionOperator

A spin-less fermionic operator.

#### NOTE
This is an opaque data structure to the C API whose internals are implemented entirely in Rust.
The remainder of this page describes the design and related functions to work with this struct.

---

## Definition

This operator is defined by a linear combination of products of fermionic creation and
annihilation operators acting on spin-less fermionic modes. That is to say, the individual
terms fulfill the following anti-commutation relations: <sup>[1](#id2)</sup>

$$
\left\{a^\dagger_i, a^\dagger_j\right\} =
\left\{a_i, a_j\right\} = 0,~~\text{and}~~
\left\{a_i, a^\dagger_j\right\} = \delta_{ij} \, ,
$$

where $i$ and $j$ do not distinguish the spin species of the fermionic
modes they are indexing.

This makes the definition of the entire operator the following:

$$
\text{\texttt{QfFermionOperator}} = \sum_i c_i \bigotimes_j \hat{A_j} \, ,
$$

where $\hat{A_j} \in \{ a_j, a^\dagger_j \}$ and $c_i$ is the (complex) coefficient
making up the linear combination of products. The index $j$ can take any value between 0
and the number of fermionic modes acted upon by the operator minus 1.

---

<a id="qf-ferm-op-implementation"></a>

## Implementation

This struct stores the terms and coefficients in multiple sparse vectors, akin to the
[compressed sparse row format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
commonly used for sparse matrices. More concretely, a single operator contains 4 arrays:

| `coeffs`     | A vector of complex coefficients consisting of two 64-bit floating point numbers.   |
|--------------|-------------------------------------------------------------------------------------|
| `actions`    | A vector of booleans storing the nature of the second-quantization actions.         |
| `modes`      | A vector of 32-bit integers storing the fermionic mode indices acted upon.          |
| `boundaries` | A vector of integers indicating the boundaries in `actions` and `modes`.            |

Entries in `actions` indicate creation (annihilation) operators by `True` (`False`).
Fermionic modes indexed by `modes` are considered spinless.

#### NOTE
You can access **read-only copies** of these internal arrays via their respective functions:

- [`qf_ferm_op_get_coeffs()`](#c.qf_ferm_op_get_coeffs)
- [`qf_ferm_op_get_actions()`](#c.qf_ferm_op_get_actions)
- [`qf_ferm_op_get_modes()`](#c.qf_ferm_op_get_modes)
- [`qf_ferm_op_get_boundaries()`](#c.qf_ferm_op_get_boundaries)

This data structure allows for very efficient construction and manipulation of operators.
However, it implies that duplicate terms might be contained in an operator at any moment.
These must be resolved manually through the use of [`qf_ferm_op_simplify()`](#c.qf_ferm_op_simplify).

### Construction

A new operator can be constructed directly by specifying the corresponding arrays outlined above.
Alternatively, an empty [`QfFermionOperator`](#c.QfFermionOperator) can be initialized with
[`qf_ferm_op_zero()`](#c.qf_ferm_op_zero) and terms can be added iteratively via [`qf_ferm_op_add_term()`](#c.qf_ferm_op_add_term).

| [`qf_ferm_op_new()`](#c.qf_ferm_op_new)      | Constructs a new operator from the provided arrays.   |
|------------------------------------------------------------------------|-------------------------------------------------------|
| [`qf_ferm_op_zero()`](#c.qf_ferm_op_zero)     | Constructs the additive identity operator.            |
| [`qf_ferm_op_one()`](#c.qf_ferm_op_one)      | Constructs the multiplicative identity operator.      |
| [`qf_ferm_op_add_term()`](#c.qf_ferm_op_add_term) | Adds a term to an existing `QfFermionOperator`.       |

#### NOTE
A [`QfFermionOperator`](#c.QfFermionOperator) can be freed with [`qf_ferm_op_free()`](#c.qf_ferm_op_free).

### Arithmetics

The following functions provide arithmetic manipulation:

| [`qf_ferm_op_add()`](#c.qf_ferm_op_add)     | Adds two operators together.              |
|-----------------------------------------------------------------------|-------------------------------------------|
| [`qf_ferm_op_mul()`](#c.qf_ferm_op_mul)     | Multiplies an operator by a scalar.       |
| [`qf_ferm_op_compose()`](#c.qf_ferm_op_compose) | Composes two operators with each other.   |
| [`qf_ferm_op_adjoint()`](#c.qf_ferm_op_adjoint) | Returns the Hermitian conjugate operator. |

### Manipulation

The following functions provide operator manipulation logic:

| [`qf_ferm_op_ichop()`](#c.qf_ferm_op_ichop)          | Removes terms with small coefficient magnitudes.          |
|------------------------------------------------------------------------------|-----------------------------------------------------------|
| [`qf_ferm_op_simplify()`](#c.qf_ferm_op_simplify)       | Returns an equivalent but simplified operator.            |
| [`qf_ferm_op_normal_ordered()`](#c.qf_ferm_op_normal_ordered) | Returns an equivalent operator with normal ordered terms. |
| [`qf_ferm_op_relabel_modes()`](#c.qf_ferm_op_relabel_modes)  | Relabels the modes of an operator.                        |

### Properties

The following functions exist to check certain properties of an operator.

| [`qf_ferm_op_is_hermitian()`](#c.qf_ferm_op_is_hermitian)              | Returns whether an operator is Hermitian.                  |
|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`qf_ferm_op_max_rank()`](#c.qf_ferm_op_max_rank)                  | Returns the maximum rank of the terms in this operator.    |
| [`qf_ferm_op_conserves_particle_number()`](#c.qf_ferm_op_conserves_particle_number) | Returns whether an operator is particle-number conserving. |

---
* <a id='id2'>**[1]**</a> [https://en.wikipedia.org/wiki/Second_quantization#Fermion_creation_and_annihilation_operators](https://en.wikipedia.org/wiki/Second_quantization#Fermion_creation_and_annihilation_operators)

## Members

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_new(uint64_t num_terms, uint64_t num_actions, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*coeffs, const bool \*actions, const uint32_t \*modes, const uint32_t \*boundaries)

Constructs a new operator. 

Any of the pointer arguments can be `NULL` if and only if their corresponding length is zero.

### Example

```c
uint64_t num_terms = 3;
uint64_t num_actions = 4;
bool actions[4] = {true, false, true, false};
uint32_t modes[4] = {0, 1, 2, 3};
QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[4] = {0, 0, 2, 4};
QfFermionOperator *op = qf_ferm_op_new(num_terms, num_actions, coeffs,
                                       actions, modes, boundaries);
```

 * **Parameters:**
  * **num_terms** – The number of terms in the operator. 
  * **num_actions** – The number of actions summed over all terms. 
  * **coeffs** – A pointer to an array of term coefficients. The length of this array should be `num_terms`. 
  * **actions** – A pointer to an array of actions over all terms. The length of this array should be `num_actions`. 
  * **modes** – A pointer to an array of action modes over all terms. The length of this array should be `num_actions`. 
  * **boundaries** – A pointer to an array of the boundaries between terms. The length of this array should be `num_terms + 1`.

### void qf_ferm_op_free([QfFermionOperator](#c.QfFermionOperator) \*op)

Frees an existing operator. 

### Example

```c
QfFermionOperator *op = qf_ferm_op_one();
qf_ferm_op_free(op);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator to be freed.

### void qf_ferm_op_get_coeffs(const [QfFermionOperator](#c.QfFermionOperator) \*op, [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*\*coeffs_out, uint64_t \*coeffs_len)

Provides read-only access to the operator’s coefficients. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-ferm-op-implementation).

### Example

```c
uint64_t num_terms = 2;
uint64_t num_actions = 0;
QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[3] = {0, 0, 0};
QfFermionOperator *op =
    qf_ferm_op_new(num_terms, num_actions, coeffs, NULL, NULL, boundaries);

QkComplex64 *coeffs_out;
uint64_t *coeffs_len;

qf_ferm_op_get_coeffs(op, &coeffs_out, &coeffs_len);

assert(coeffs_len == 2);
```

 #### NOTE
This function returns a **copy** of the internal data.

* **Parameters:**
  * **op** – A pointer to the fermionic operator whose coefficients to access. 
  * **coeffs_out** – A pointer to the array of complex values into which to write the coefficients. 
  * **coeffs_len** – A pointer to the integer into which to write the length of the output array.

### void qf_ferm_op_get_actions(const [QfFermionOperator](#c.QfFermionOperator) \*op, bool \*\*actions_out, uint64_t \*actions_len)

Provides read-only access to the operator’s actions. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-ferm-op-implementation).

### Example

```c
uint64_t num_terms = 2;
uint64_t num_actions = 2;
bool actions[2] = {true, false};
uint32_t modes[2] = {0, 1};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[3] = {0, 0, 2};
QfFermionOperator *op =
    qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

QkComplex64 *actions_out;
uint64_t *actions_len;

qf_ferm_op_get_actions(op, &actions_out, &actions_len);

assert(actions_len == 2);
```

 #### NOTE
This function returns a **copy** of the internal data.

* **Parameters:**
  * **op** – A pointer to the fermionic operator whose actions to access. 
  * **actions_out** – A pointer to the array of boolean values into which to write the actions. 
  * **actions_len** – A pointer to the integer into which to write the length of the output array.

### void qf_ferm_op_get_modes(const [QfFermionOperator](#c.QfFermionOperator) \*op, uint32_t \*\*modes_out, uint64_t \*modes_len)

Provides read-only access to the operator’s acted-upon mode indices. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-ferm-op-implementation).

### Example

```c
uint64_t num_terms = 2;
uint64_t num_actions = 2;
bool actions[2] = {true, false};
uint32_t modes[2] = {0, 1};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[3] = {0, 0, 2};
QfFermionOperator *op =
    qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

QkComplex64 *modes_out;
uint64_t *modes_len;

qf_ferm_op_get_modes(op, &modes_out, &modes_len);

assert(modes_len == 2);
```

 #### NOTE
This function returns a **copy** of the internal data.

* **Parameters:**
  * **op** – A pointer to the fermionic operator whose modes to access. 
  * **modes_out** – A pointer to the array of boolean values into which to write the modes. 
  * **modes_len** – A pointer to the integer into which to write the length of the output array.

### void qf_ferm_op_get_boundaries(const [QfFermionOperator](#c.QfFermionOperator) \*op, size_t \*\*boundaries_out, uint64_t \*boundaries_len)

Provides read-only access to the indices indicating the boundaries between operator terms. 

#### SEE ALSO
The explanation of the internal data structure, [here](#qf-ferm-op-implementation).

### Example

```c
uint64_t num_terms = 2;
uint64_t num_actions = 2;
bool actions[2] = {true, false};
uint32_t modes[2] = {0, 1};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
uint32_t boundaries[3] = {0, 0, 2};
QfFermionOperator *op =
    qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

QkComplex64 *boundaries_out;
uint64_t *boundaries_len;

qf_ferm_op_get_boundaries(op, &boundaries_out, &boundaries_len);

assert(boundaries_len == 3);
```

 #### NOTE
This function returns a **copy** of the internal data.

* **Parameters:**
  * **op** – A pointer to the fermionic operator whose boundaries to access. 
  * **boundaries_out** – A pointer to the array of boolean values into which to write the boundaries. 
  * **boundaries_len** – A pointer to the integer into which to write the length of the output array.

### uint32_t qf_ferm_op_num_support(const [QfFermionOperator](#c.QfFermionOperator) \*op)

Gets the size of the support of an operator. 

Use this to size the output buffer of [`qf_ferm_op_get_support()`](#c.qf_ferm_op_get_support), which does not report a
length of its own.

### Example

```c
bool actions[3] = {true, false, true};
uint32_t modes[3] = {2, 0, 2};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[3] = {0, 2, 3};
QfFermionOperator *op = qf_ferm_op_new(2, 3, coeffs, actions, modes, boundaries);

uint32_t num_support = qf_ferm_op_num_support(op);

assert(num_support == 2);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator whose support size to get.
* **Returns:**
  The number of distinct mode indices acted upon by the operator.

### void qf_ferm_op_get_support(const [QfFermionOperator](#c.QfFermionOperator) \*op, uint32_t \*support_out)

Gets the support of an operator, i.e. the mode indices acted upon by the operator. 

The indices are written in **ascending** order, and every index appears exactly once no matter
how many terms act upon it.

### Example

```c
bool actions[3] = {true, false, true};
uint32_t modes[3] = {2, 0, 2};
QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[3] = {0, 2, 3};
QfFermionOperator *op = qf_ferm_op_new(2, 3, coeffs, actions, modes, boundaries);

uint32_t num_support = qf_ferm_op_num_support(op);
uint32_t *support_out = malloc(num_support * sizeof(uint32_t));

qf_ferm_op_get_support(op, support_out);

assert(support_out[0] == 0);
assert(support_out[1] == 2);

free(support_out);
```

 #### NOTE
Unlike the other getters, this does not hand out a pointer into the operator: the support is
computed on demand rather than stored, so the caller provides the output buffer. Query its
required length with [`qf_ferm_op_num_support()`](#c.qf_ferm_op_num_support) first.

* **Parameters:**
  * **op** – A pointer to the fermionic operator whose support to get. 
  * **support_out** – A pointer to the integer array into which to write the support. It must be sized to at least :c:func:`[qf_ferm_op_num_support](#group__qf__ferm__op_1ga414948ea3919af7826c8a6ad2051cc51)` elements.

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_zero(void)

Constructs the additive identity operator. 

Adding the operator that is constructed by this method to another one has no effect.

### Example

```c
QfFermionOperator *zero = qf_ferm_op_zero();

QfFermionOperator *op_plus_zero = qf_ferm_op_add(op, zero);

assert(qf_ferm_op_equal(op, op_plus_zero));
```

 * **Returns:**
  A pointer to the created operator.

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_one(void)

Constructs the multiplicative identity operator. 

Composing the operator that is constructed by this method with another one has no effect.

### Example

```c
QfFermionOperator *one = qf_ferm_op_one();

QfFermionOperator *op_times_one = qf_ferm_op_compose(op, one);

assert(qf_ferm_op_equal(op, op_times_one));
```

 * **Returns:**
  A pointer to the created operator.

### bool qf_ferm_op_has_groups(const [QfFermionOperator](#c.QfFermionOperator) \*op)

Checks whether this operator tracks group indices. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);
QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);

bool has_groups = qf_ferm_op_has_groups(op);

assert(!has_groups);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator to be checked.
* **Returns:**
  Whether the provided operator has a `groups` attribute.

### uint32_t qf_ferm_op_num_groups(const [QfFermionOperator](#c.QfFermionOperator) \*op)

Gets the number of groups from an operator. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);
QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);

uint32_t num_groups = qf_ferm_op_num_groups(op);
```

 #### NOTE
The number of groups is evaluated lazily as the largest occurring group index plus 1.

* **Parameters:**
  * **op** – A pointer to the fermionic operator whose number of groups to get.
* **Returns:**
  The number of group indices from the operator’s `groups` attribute.

### void qf_ferm_op_get_groups(const [QfFermionOperator](#c.QfFermionOperator) \*op, uint32_t \*\*groups_out, uint64_t \*groups_len)

Gets the group indices for all operator terms. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfFermionOperator *op = ...;
uint32_t *groups_out;
uint32_t groups_len;

qf_ferm_op_get_groups(op, &groups_out, &groups_len);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator whose group indices to get. 
  * **groups_out** – A pointer to the integer array into which to write the group indices. 
  * **groups_len** – A pointer to the integer into which to write the length of the output array.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ferm_op_set_groups([QfFermionOperator](#c.QfFermionOperator) \*op, const uint32_t \*groups_in, uint64_t groups_len)

Sets the `groups` attribute of the provided operator. 

The length is validated here because nothing downstream re-checks it: too few indices would
silently drop the trailing terms wherever terms are iterated together with their groups, and too
many would make [`qf_ferm_op_num_groups()`](#c.qf_ferm_op_num_groups) report groups that no term carries.

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfFermionOperator *op = ...;

uint32_t num_terms = 4;
uint32_t groups_in[4] = {0, 1, 0, 1};
qf_ferm_op_set_groups(op, groups_in, num_terms);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator whose `groups` attribute to write. 
  * **groups_in** – A pointer to the `groups` integer array to write into the operator. 
  * **groups_len** – The number of terms in the `groups_in` array.
* **Returns:**
  An exit code.
  * `[QfExitCode_Success](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a3e14e3ca9bf1cb919c6913131132da37)` upon success
  * `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if `groups_len` differs from the number of terms in the operator

### void qf_ferm_op_del_groups([QfFermionOperator](#c.QfFermionOperator) \*op)

Deletes the `groups` attribute from the provided operator. 

#### SEE ALSO
The explanation on [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation).

### Example

```c
QfFermionOperator *op = ...;

qf_ferm_op_del_groups(op);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator whose `groups` attribute to delete.

### void qf_ferm_op_split_out_groups(const [QfFermionOperator](#c.QfFermionOperator) \*op, const uint32_t \*group_indices, uint64_t num_indices, [QfFermionOperator](#c.QfFermionOperator) \*\*group_ops_out)

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
uint64_t num_actions = 8;
bool actions[8] = {true, false, true, false, true, false, true, false};
uint32_t modes[8] = {0, 1, 2, 3, 1, 0, 3, 2};
QkComplex64 coeffs[4] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[5] = {0, 2, 4, 6, 8};
QfFermionOperator *op =
    qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

uint32_t groups_in[4] = {0, 1, 0, 1};
qf_ferm_op_set_groups(op, groups_in, num_terms);

// build every group, in index order
QfFermionOperator *group_ops[2];
qf_ferm_op_split_out_groups(op, NULL, 0, group_ops);

// build only group 1
uint32_t group_indices[1] = {1};
QfFermionOperator *group_op[1];
qf_ferm_op_split_out_groups(op, group_indices, 1, group_op);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator whose `groups` to split out. 
  * **group_indices** – A pointer to the array of group indices for which to build operators, in the desired output order. May be `NULL`, in which case every group is built, in index order (equivalent to passing every index from `0` to :c:func:`[qf_ferm_op_num_groups](#group__qf__ferm__op_1ga7346961fb39b1608511fd4b85b399c0c)` `- 1`). 
  * **num_indices** – The number of indices in the `group_indices` array. Ignored if `group_indices` is `NULL`. 
  * **group_ops_out** – A pointer to the array of :c:struct:`QfFermionOperator` into which to write the operators for each requested group. Must be sized to `num_indices` when `group_indices` is non-`NULL`, or to :c:func:`[qf_ferm_op_num_groups](#group__qf__ferm__op_1ga7346961fb39b1608511fd4b85b399c0c)` when it is `NULL`.

### void qf_ferm_op_add_term([QfFermionOperator](#c.QfFermionOperator) \*op, uint64_t num_actions, const bool \*actions, const uint32_t \*modes, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*coeff)

Adds a term to an existing operator. 

Any of the pointer arguments can be `NULL` if and only if their corresponding length is zero.

#### CAUTION
This function resets the operator’s `groups` attribute to `NULL`.

### Example

```c
QfFermionOperator *one = qf_ferm_op_one();

QfFermionOperator *op = qf_ferm_op_zero();
bool actions[0] = {};
uint32_t modes[0] = {};
QkComplex64 coeff = {1.0, 0.0};

qf_ferm_op_add_term(op, 0, actions, modes, &coeff);

assert(qf_ferm_op_equal(op, one));
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator to be modified. 
  * **num_actions** – The length of the actions array. 
  * **actions** – A pointer to an array of actions. The length of this array should be `num_actions`. 
  * **modes** – A pointer to an array of action modes. The length of this array should be `num_actions`. 
  * **coeff** – A pointer to the complex coefficient.

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_add(const [QfFermionOperator](#c.QfFermionOperator) \*left, const [QfFermionOperator](#c.QfFermionOperator) \*right)

Adds two operators together. 

### Example

```c
QfFermionOperator *one = qf_ferm_op_one();
QfFermionOperator *zero = qf_ferm_op_zero();

QfFermionOperator *result = qf_ferm_op_add(one, zero);

assert(qf_ferm_op_equal(result, one));
```

 * **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator.
* **Returns:**
  A pointer to the resulting operator.

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_scaled_add(const [QfFermionOperator](#c.QfFermionOperator) \*left, const [QfFermionOperator](#c.QfFermionOperator) \*right, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*factor)

Adds two operators together, scaling the coefficients of the right one. 

This fuses the scaling into the addition, which avoids building a fully scaled copy of `right`
just to append it. Passing a factor of `-1` therefore subtracts, which is why no separate
subtraction function is provided.

### Example

```c
QfFermionOperator *one = qf_ferm_op_one();
QfFermionOperator *other = qf_ferm_op_one();

// Subtracting via a factor of -1 leaves the two terms cancelling each other out.
QkComplex64 factor = {-1.0, 0.0};
QfFermionOperator *result = qf_ferm_op_scaled_add(one, other, &factor);

QfFermionOperator *simplified = qf_ferm_op_simplify(result, 1e-10);
assert(qf_ferm_op_len(simplified) == 0);
```

 #### NOTE
Like [`qf_ferm_op_add()`](#c.qf_ferm_op_add), this appends the terms of `right` without combining them with
those of `left`, so the result tracks no group indices. Call [`qf_ferm_op_simplify()`](#c.qf_ferm_op_simplify) to
collect equal terms afterwards.

* **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator. 
  * **factor** – A pointer to the factor to scale the right operator’s coefficients with.
* **Returns:**
  A pointer to the resulting operator, `left + factor * right`.

### void qf_ferm_op_add_inplace([QfFermionOperator](#c.QfFermionOperator) \*left, const [QfFermionOperator](#c.QfFermionOperator) \*right)

Adds an operator into another one, in place. 

The in-place counterpart of [`qf_ferm_op_add()`](#c.qf_ferm_op_add), which saves copying `left` into a freshly
allocated result.

#### CAUTION
This function resets the operator’s `groups` attribute to `NULL`.

### Example

```c
QfFermionOperator *left = qf_ferm_op_one();
QfFermionOperator *right = qf_ferm_op_one();

qf_ferm_op_add_inplace(left, right);

assert(qf_ferm_op_len(left) == 2);
```

 * **Parameters:**
  * **left** – A pointer to the operator to add into. 
  * **right** – A pointer to the operator to add.

### void qf_ferm_op_scaled_add_inplace([QfFermionOperator](#c.QfFermionOperator) \*left, const [QfFermionOperator](#c.QfFermionOperator) \*right, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*factor)

Adds an operator into another one in place, scaling the coefficients of the right one. 

The in-place counterpart of [`qf_ferm_op_scaled_add()`](#c.qf_ferm_op_scaled_add), computing `left + factor * right`
without allocating a result. As there, a factor of `-1` subtracts.

#### CAUTION
This function resets the operator’s `groups` attribute to `NULL`.

### Example

```c
QfFermionOperator *left = qf_ferm_op_one();
QfFermionOperator *right = qf_ferm_op_one();

// Subtract, leaving two terms that cancel each other out.
QkComplex64 factor = {-1.0, 0.0};
qf_ferm_op_scaled_add_inplace(left, right, &factor);

QfFermionOperator *simplified = qf_ferm_op_simplify(left, 1e-10);
assert(qf_ferm_op_len(simplified) == 0);
```

 * **Parameters:**
  * **left** – A pointer to the operator to add into. 
  * **right** – A pointer to the operator to add. 
  * **factor** – A pointer to the factor to scale the right operator’s coefficients with.

### void qf_ferm_op_mul_inplace([QfFermionOperator](#c.QfFermionOperator) \*op, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*scalar)

Multiplies an operator by a scalar, in place. 

The in-place counterpart of [`qf_ferm_op_mul()`](#c.qf_ferm_op_mul), which saves copying the operator into a
freshly allocated result.

### Example

```c
QfFermionOperator *op = qf_ferm_op_one();
QkComplex64 scalar = {2.0, 0.0};

qf_ferm_op_mul_inplace(op, &scalar);
```

 #### NOTE
Unlike the two in-place additions above, this **preserves** the `groups` attribute: scaling
the coefficients leaves the number of terms, and hence the one-index-per-term invariant,
untouched.

* **Parameters:**
  * **op** – A pointer to the operator to scale. 
  * **scalar** – A pointer to the scalar.

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_mul(const [QfFermionOperator](#c.QfFermionOperator) \*op, const [QkComplex64](https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-complex-64#qkcomplex64) \*scalar)

Multiplies an operator by a scalar. 

### Example

```c
QfFermionOperator *one = qf_ferm_op_one();
QkComplex64 coeff = {2.0, 0.0};
QfFermionOperator *result = qf_ferm_op_mul(one, &coeff);

QfFermionOperator *expected = qf_ferm_op_zero();
bool actions[0] = {};
uint32_t modes[0] = {};
qf_ferm_op_add_term(expected, 0, actions, modes, &coeff);

assert(qf_ferm_op_equal(result, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator. 
  * **scalar** – A pointer to the scalar.
* **Returns:**
  A pointer to the resulting operator.

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_compose(const [QfFermionOperator](#c.QfFermionOperator) \*left, const [QfFermionOperator](#c.QfFermionOperator) \*right)

Composes two operators with each other. 

### Example

```c
QfFermionOperator *one = qf_ferm_op_one();
QfFermionOperator *zero = qf_ferm_op_zero();

QfFermionOperator *result = qf_ferm_op_compose(one, zero);

assert(qf_ferm_op_equal(result, zero));
```

 #### NOTE
The composition of two operators tracks no groups, even when both operands do.

* **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator.
* **Returns:**
  A pointer to `left.compose(right)`, which equals the operator `result = right @ left` in terms of the matrix multiplication `@`. In other words, `right` is applied first. To obtain `left @ right`, swap the arguments.

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_adjoint(const [QfFermionOperator](#c.QfFermionOperator) \*op)

Returns the Hermitian conjugate (or adjoint) of an operator. 

This affects the terms and coefficients as follows:

* the actions in each term reverse their order and flip between creation and annihilation
* the coefficients are complex conjugated

### Example

```c
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[0] = {};
uint32_t modes[0] = {};
QkComplex64 coeff = {0.0, 1.0};
qf_ferm_op_add_term(op, 0, actions, modes, &coeff);

QfFermionOperator *adjoint = qf_ferm_op_adjoint(op);

QfFermionOperator *expected = qf_ferm_op_zero();
QkComplex64 coeff_adj = {0.0, -1.0};
qf_ferm_op_add_term(expected, 0, actions, modes, &coeff_adj);

assert(qf_ferm_op_equal(adjoint, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator.
* **Returns:**
  A pointer to the created operator.

### void qf_ferm_op_ichop([QfFermionOperator](#c.QfFermionOperator) \*op, double atol)

Removes terms whose coefficient magnitude lies below the provided threshold. 

#### CAUTION
This functions truncates coefficients greedily! If the acted upon operator might contain
separate coefficients for duplicate terms consider calling [`qf_ferm_op_simplify()`](#c.qf_ferm_op_simplify)
instead!

### Example

```c
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[0] = {};
uint32_t modes[0] = {};
QkComplex64 coeff = {1e-8};
qf_ferm_op_add_term(op, 0, actions, modes, &coeff);

qf_ferm_op_ichop(op, 1e-6);

QfFermionOperator *expected = qf_ferm_op_zero();

assert(qf_ferm_op_equal(op, expected));
```

 * **Parameters:**
  * **op** – A pointer to the operator. 
  * **atol** – The absolute tolerance for coefficient truncation.

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_simplify(const [QfFermionOperator](#c.QfFermionOperator) \*op, double atol)

Returns an equivalent but simplified operator. 

The simplification process first sums all coefficients that belong to equal terms and then
only retains those whose total coefficient exceeds the specified tolerance (just like
[`qf_ferm_op_ichop()`](#c.qf_ferm_op_ichop)).

When an operator has been arithmetically manipulated or constructed in a way that does not
guarantee unique terms, this method should be called before applying any method that
filters numerically small coefficients to avoid loss of information. See the example below
which showcases how [`qf_ferm_op_ichop()`](#c.qf_ferm_op_ichop) can truncate terms that sum to a total
coefficient magnitude which should not be truncated:

```c
uint64_t num_terms = 100000;
uint64_t num_actions = 0;
bool actions[0] = {};
uint32_t modes[0] = {};
QkComplex64 coeffs[100000];
uint32_t boundaries[100001];
for (int i = 0; i < 100000; i++) {
  coeffs[i].re = 1e-5;
  coeffs[i].im = 0.0;
  boundaries[i] = 0;
}
boundaries[100000] = 0;
QfFermionOperator *op = qf_ferm_op_new(num_terms, num_actions, coeffs,
                                       actions, modes, boundaries);

QfFermionOperator *canon = qf_ferm_op_simplify(op, 1e-4);

QfFermionOperator *one = qf_ferm_op_one();
bool canon_is_equal = qf_ferm_op_equiv(canon, one, 1e-6);

qf_ferm_op_ichop(op, 1e-4);

QfFermionOperator *zero = qf_ferm_op_zero();
bool ichop_is_equal = qf_ferm_op_equiv(op, zero, 1e-6);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator to be simplified. 
  * **atol** – The absolute tolerance for coefficient truncation.
* **Returns:**
  An equivalent but simplified operator.

### [QfFermionOperator](#c.QfFermionOperator) \*qf_ferm_op_normal_ordered(const [QfFermionOperator](#c.QfFermionOperator) \*op, const bool \*sandwich)

Returns an equivalent operator with normal ordered terms. 

The normal order of an operator term is defined such that all creation actions appear before all annihilation actions. Within each group, the acted-upon modes are ordered lexicographically. Whether their order is ascending or descending depends upon the value of the `sandwich` argument:

* `NULL`: both groups are ordered lexicographically descending (e.g. `+_1 +_0 -_1 -_0`)
* `True`: larger indices appear towards the middle, i.e. creation actions are lexicographically ascending while annihilation ones are descending (e.g. `+_0 +_1 -_1 -_0`)
* `False`: smaller indices appear towards the middle, i.e. creation actions are lexicographically descending while annihilation ones are ascending (e.g. `+_1 +_0 -_0 -_1`)

### Example

```c
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[4] = {false, true, false, true};
uint32_t modes[4] = {1, 1, 0, 0};
QkComplex64 coeff = {1.0, 0.0};
qf_ferm_op_add_term(op, 4, actions, modes, &coeff);

QfFermionOperator *normal_ordered = qf_ferm_op_normal_ordered(op, NULL);

uint64_t num_terms = 4;
uint64_t num_actions = 8;
bool actions_exp[8] = {true, false, true, false, true, true, false, false};
uint32_t modes_exp[8] = {0, 0, 1, 1, 1, 0, 1, 0};
QkComplex64 coeffs_exp[4] = {
    {1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}};
uint32_t boundaries_exp[5] = {0, 0, 2, 4, 8};
QfFermionOperator *expected =
    qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp,
                   modes_exp, boundaries_exp);

assert(qf_ferm_op_equal(normal_ordered, expected));
```

 #### NOTE
When a term is being reordered, the anti-commutation relations have to be taken into
account, $a_i a^\dagger_j = \delta_{ij} - a^\dagger_j a^i$, implying that the
number of terms might change.

* **Parameters:**
  * **op** – A pointer to the operator. 
  * **sandwich** – A pointer to a boolean value. This pointer can be `NULL`.
* **Returns:**
  A pointer to the created operator.

### bool qf_ferm_op_is_hermitian(const [QfFermionOperator](#c.QfFermionOperator) \*op, double atol)

Checks whether an operator is Hermitian. 

### Example

```c
QfFermionOperator *op = qf_ferm_op_zero();
bool actions1[2] = {true, false};
uint32_t modes1[2] = {0, 1};
QkComplex64 coeff1 = {0.0, 1.00001};
qf_ferm_op_add_term(op, 2, actions1, modes1, &coeff1);
bool actions2[2] = {true, false};
uint32_t modes2[2] = {1, 0};
QkComplex64 coeff2 = {0.0, -1};
qf_ferm_op_add_term(op, 2, actions2, modes2, &coeff2);

assert(qf_ferm_op_is_hermitian(op, 1e-4));
assert(!qf_ferm_op_is_hermitian(op, 1e-8));
```

 #### NOTE
This check is implemented using [`qf_ferm_op_equiv()`](#c.qf_ferm_op_equiv) on the
[`qf_ferm_op_normal_ordered()`](#c.qf_ferm_op_normal_ordered) difference of `op` and its
[`qf_ferm_op_adjoint()`](#c.qf_ferm_op_adjoint) and [`qf_ferm_op_zero()`](#c.qf_ferm_op_zero).

* **Parameters:**
  * **op** – A pointer to the fermionic operator to be checked. 
  * **atol** – The absolute tolerance upto which coefficients are considered equal.
* **Returns:**
  Whether the provided operator is Hermitian.

### uint32_t qf_ferm_op_max_rank(const [QfFermionOperator](#c.QfFermionOperator) \*op)

Checks the maximum rank of an operator. 

### Example

```c
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[4] = {true, false, true, false};
uint32_t modes[4] = {0, 1, 2, 3};
QkComplex64 coeff = {1.0, 0.0};
qf_ferm_op_add_term(op, 4, actions, modes, &coeff);

assert(qf_ferm_op_max_rank(op), 4);
```

 #### NOTE
The length of the longest term can depend on the operator’s form which means that (for
example) operator simplification or normal-ordering can result in a different maximum rank.

* **Parameters:**
  * **op** – A pointer to the fermionic operator to be checked.
* **Returns:**
  The maximum rank of the operator.

### bool qf_ferm_op_conserves_particle_number(const [QfFermionOperator](#c.QfFermionOperator) \*op)

Checks whether an operator is particle-number conserving. 

This is the special case of [`qf_ferm_op_conserves_sector()`](#c.qf_ferm_op_conserves_sector) that treats all modes as a
single block. Use that function to additionally require conservation *within* mode blocks, for
example to check conservation of the spin z-component alongside the particle number.

### Example

```c
// A hopping term conserves the particle number, ...
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[2] = {true, false};
uint32_t modes[2] = {0, 2};
QkComplex64 coeff = {1.0, 0.0};
qf_ferm_op_add_term(op, 2, actions, modes, &coeff);

assert(qf_ferm_op_conserves_particle_number(op));

// ... whereas a bare creation operator does not.
QfFermionOperator *cre = qf_ferm_op_zero();
bool cre_actions[1] = {true};
uint32_t cre_modes[1] = {0};
qf_ferm_op_add_term(cre, 1, cre_actions, cre_modes, &coeff);

assert(!qf_ferm_op_conserves_particle_number(cre));
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator to be checked.
* **Returns:**
  Whether the provided operator is particle-number conserving.

### bool qf_ferm_op_conserves_sector(const [QfFermionOperator](#c.QfFermionOperator) \*op, const uint32_t \*block_sizes, uint64_t num_blocks)

Checks whether an operator conserves the particle number within each mode block. 

The `block_sizes` partition the mode range into consecutive, non-overlapping blocks: block
`b` spans the modes `[start_b, start_b + block_sizes[b])` where `start_b` is the sum of all
preceding block sizes. A term conserves the sector if and only if, in *every* block, its number
of creation operators equals its number of annihilation operators. A term acting on a mode beyond
the last block never conserves the sector, because there is no block whose count it could
balance.

For a spin-orbital layout of `norb` spatial orbitals, `[norb]` checks the particle number of
a spinless system, while `[norb, norb]` requires the alpha block `[0, norb)` and the beta
block `[norb, 2 * norb)` to be conserved *separately*, i.e. strict conservation of both the
particle number and the z-component of the spin.

### Example

```c
// This term hops an electron from the beta block into the alpha block.
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[2] = {true, false};
uint32_t modes[2] = {0, 2};
QkComplex64 coeff = {1.0, 0.0};
qf_ferm_op_add_term(op, 2, actions, modes, &coeff);

// The total particle number is conserved, ...
uint32_t one_block[1] = {4};
assert(qf_ferm_op_conserves_sector(op, one_block, 1));
assert(qf_ferm_op_conserves_sector(op, NULL, 0));

// ... but the two spin sectors are not conserved individually.
uint32_t spin_blocks[2] = {2, 2};
assert(!qf_ferm_op_conserves_sector(op, spin_blocks, 2));
```

 #### NOTE
Passing `NULL` (with `num_blocks` set to `0`) treats all modes as one unbounded block,
making this equivalent to [`qf_ferm_op_conserves_particle_number()`](#c.qf_ferm_op_conserves_particle_number).

* **Parameters:**
  * **op** – A pointer to the fermionic operator to be checked. 
  * **block_sizes** – A pointer to the array of block sizes partitioning the mode range. May be `NULL` to treat all modes as a single block. 
  * **num_blocks** – The number of entries in the `block_sizes` array.
* **Returns:**
  Whether the provided operator conserves the particle number within every block.

### bool qf_ferm_op_equal(const [QfFermionOperator](#c.QfFermionOperator) \*left, const [QfFermionOperator](#c.QfFermionOperator) \*right)

Compare two operators for equality. 

### Example

```c
QfFermionOperator *one = qf_ferm_op_one();
QfFermionOperator *zero = qf_ferm_op_zero();

assert(qf_ferm_op_equal(one, one));
assert(!qf_ferm_op_equal(one, zero));
```

 * **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator.
* **Returns:**
  Whether the two operators are equal.

### bool qf_ferm_op_equiv(const [QfFermionOperator](#c.QfFermionOperator) \*left, const [QfFermionOperator](#c.QfFermionOperator) \*right, double atol)

Compare two operators for equivalence. 

Equivalence in this context means approximate equality up to the specified absolute tolerance. To be more precise, this method returns `True`, when all the absolute values of the coefficients in the difference `other - self` are below the specified threshold `atol`.

### Example

```c
QfFermionOperator *zero = qf_ferm_op_zero();

QfFermionOperator *op = qf_ferm_op_zero();
bool actions[0] = {};
uint32_t modes[0] = {};
QkComplex64 coeff = {1e-7, 0.0};
qf_ferm_op_add_term(op, 0, actions, modes, &coeff);

assert(qf_ferm_op_equiv(op, zero, 1e-6));
assert(!qf_ferm_op_equiv(op, zero, 1e-8));
```

 * **Parameters:**
  * **left** – A pointer to the left operator. 
  * **right** – A pointer to the right operator. 
  * **atol** – The absolute tolerance for coefficient equivalence.
* **Returns:**
  Whether the two operators are equivalent.

### size_t qf_ferm_op_len(const [QfFermionOperator](#c.QfFermionOperator) \*op)

Returns the length (or number of terms) of the provided operator. 

### Example

```c
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[4] = {true, false, true, false};
uint32_t modes[4] = {0, 1, 2, 3};
QkComplex64 coeff = {1.0, 0.0};
qf_ferm_op_add_term(op, 4, actions, modes, &coeff);

assert(qf_ferm_op_len(op) == 1);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator.
* **Returns:**
  The length (or number of terms) of the operator.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ferm_op_relabel_modes([QfFermionOperator](#c.QfFermionOperator) \*op, uint64_t num_modes, const uint32_t \*permutation)

Relabels the indices of the provided operator. 

### Example

```c
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[4] = {true, false, true, false};
uint32_t indices[4] = {0, 1, 2, 3};
QkComplex64 coeff = {1.0, 0.0};
qf_ferm_op_add_term(op, 4, actions, indices, &coeff);

uint32_t permutation[4] = {3, 2, 1, 0};

QfExitCode exit = qf_ferm_op_relabel_modes(op, 4, permutation);

assert(exit == QfExitCode_Success);
```

 * **Parameters:**
  * **op** – A pointer to the fermionic operator. 
  * **num_modes** – The number of mode indices in the provided permutation list. 
  * **permutation** – The index permutation list.
* **Returns:**
  An exit code.
  * `[QfExitCode_Success](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a3e14e3ca9bf1cb919c6913131132da37)` upon success
  * `[QfExitCode_DuplicateIndexError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a01f24cac283f08bcfa375ff2a59772ef)` if duplicate indices were found in the permutation
  * `[QfExitCode_IndexError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1ae5c84289c900faf721fde93e2f249e3b)` for any other index errors, such as invalid indices.
