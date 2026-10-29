# OperatorTrait

### *class* OperatorTrait(\*args, \*\*kwargs)

A protocol indicating all methods implemented by operator classes.

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

### Attributes

#### groups

Returns the groups indices.

### Methods

#### adjoint()

Returns the adjoint of this operator.

* **Return type:**
  [*Self*](https://docs.python.org/3/library/typing.html#typing.Self)

#### equiv(other, atol)

Checks this operator with another for equivalence up to the specified absolute tolerance.

* **Parameters:**
  * **other** ([*Self*](https://docs.python.org/3/library/typing.html#typing.Self))
  * **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float))
* **Return type:**
  [bool](https://docs.python.org/3/builtins/functions.html#bool)

#### *classmethod* from_terms(terms)

Constructs a new operator from an iterator (see also [`iter_terms()`](#qiskit_fermions.operators.OperatorTrait.iter_terms)).

* **Parameters:**
  **terms** ([*Iterable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable))
* **Return type:**
  [*Self*](https://docs.python.org/3/library/typing.html#typing.Self)

#### *classmethod* from_terms_with_groups(terms)

Constructs a new operator from an iterator (see also [`iter_terms_with_groups()`](#qiskit_fermions.operators.OperatorTrait.iter_terms_with_groups)).

* **Parameters:**
  **terms** ([*Iterable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable))
* **Return type:**
  [*Self*](https://docs.python.org/3/library/typing.html#typing.Self)

#### get_coeffs()

Returns the term coefficients.

* **Return type:**
  [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[complex](https://docs.python.org/3/builtins/functions.html#complex)]

#### get_support()

Returns the set of mode indices which this operator acts upon.

* **Return type:**
  [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[int](https://docs.python.org/3/builtins/functions.html#int)]

#### has_groups()

Returns whether this operator tracks group indices.

This is equivalent to (but cheaper than) checking `op.groups is not None`, because it does
not copy the group indices out of the operator in order to inspect them.

* **Return type:**
  [bool](https://docs.python.org/3/builtins/functions.html#bool)

#### ichop(atol)

Trims coefficients below the absolute tolerance from this operator.

* **Parameters:**
  **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float))

#### is_hermitian(atol=1e-08)

Returns whether this operator is Hermitian.

The check reduces the difference of this operator and its [`adjoint()`](#qiskit_fermions.operators.OperatorTrait.adjoint) to a normal form
(see [`normal_ordered()`](#qiskit_fermions.operators.OperatorTrait.normal_ordered)) and compares the result against [`zero()`](#qiskit_fermions.operators.OperatorTrait.zero). It is therefore
only as strong as the normal form that the specific operator type reduces through.

#### NOTE
A `True` result is always reliable. A `False` result is reliable only for operator
types whose normal form is a genuine canonical form; for the others this check is
*conservative* and can return `False` for an operator that is in fact Hermitian.
Consult the specific implementation to find out which of the two applies.

* **Parameters:**
  **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The numerical accuracy upto which coefficients are considered equal. This value
  defaults to `1e-8`.
* **Returns:**
  Whether this operator is Hermitian.
* **Return type:**
  [bool](https://docs.python.org/3/builtins/functions.html#bool)

#### iter_terms()

Iterates over the terms of this operator.

* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)

#### iter_terms_with_groups()

Iterates over the terms of this operator with their group indices.

* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)

#### normal_ordered(\*args, \*\*kwargs)

Returns the normal-ordered form of this operator.

#### NOTE
A specific implementation of this method may take additional arguments.

* **Return type:**
  [*Self*](https://docs.python.org/3/library/typing.html#typing.Self)

#### num_groups()

Returns the number of groups.

* **Return type:**
  [int](https://docs.python.org/3/builtins/functions.html#int) | None

#### *classmethod* one()

Constructs the multiplicative identity operator.

* **Return type:**
  [*Self*](https://docs.python.org/3/library/typing.html#typing.Self)

#### relabel_modes(permutation)

Relabels the modes of the operator.

* **Parameters:**
  **permutation** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *]*)
* **Return type:**
  [*Self*](https://docs.python.org/3/library/typing.html#typing.Self)

#### simplify(atol)

Simplifies the terms of this operator, discarding those below the absolute tolerance.

* **Parameters:**
  **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float))
* **Return type:**
  [*Self*](https://docs.python.org/3/library/typing.html#typing.Self)

#### split_out_groups(group_indices=None)

Splits this operator into an optional list of new operators based on its [`groups`](#qiskit_fermions.operators.OperatorTrait.groups).

#### NOTE
If `group_indices` is omitted, every group is built, in index order. Otherwise, only
the requested indices are built, in the given order.

* **Parameters:**
  **group_indices** ([*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *]*  *|* *None*)
* **Return type:**
  [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[*Self*](https://docs.python.org/3/library/typing.html#typing.Self)]

#### *classmethod* zero()

Constructs the additive identity operator.

* **Return type:**
  [*Self*](https://docs.python.org/3/library/typing.html#typing.Self)
