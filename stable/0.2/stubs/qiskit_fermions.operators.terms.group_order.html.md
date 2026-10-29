# group_order

### group_order(op)

Returns a copy of an operator with its terms ordered by group index.

The terms are sorted by their group index alone, which makes each group one contiguous run of
terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
relative order: ordering canonically first and by group second therefore refines the canonical
order rather than replacing it with an arbitrary permutation. The terms themselves are left
untouched – this only reorders them, it does not simplify or normal-order the operator.

This works for any of the built-in operator types implementing the
[`OperatorTrait`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait) protocol; the returned operator is of the same
type as the input.

Group indices say only which terms belong together, so this changes the operator’s *layout*, not
its value. What the layout buys is lookup cost:
[`split_out_groups()`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.split_out_groups) has to scan every term to find
the requested groups in general, but on a group-ordered operator each group’s terms form one
range it can locate by binary search, so a lookup costs what the *requested* groups cost rather
than what the *held* terms cost. Hoisting this call out of a loop that repeatedly samples a few
groups from a large operator is what makes those lookups affordable.

#### NOTE
An operator tracking no groups (see for example
[`groups`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator.groups)) has nothing to order by and is
returned as an unchanged copy, so this composes in a pipeline without a guard at every step.
The result tracks no groups either.

```pycon
>>> from qiskit_fermions.operators import FermionOperator
>>> from qiskit_fermions.operators.terms.ordering import group_order
>>> op = FermionOperator.from_dict(
...     {
...         ((True, 0), (False, 1)): 1.0,
...         ((True, 1), (False, 0)): 2.0,
...         ((True, 2), (False, 3)): 3.0,
...     }
... )
>>> op.groups = [1, 0, 1]  # group 1's terms are not contiguous
>>> ordered = group_order(op)
>>> print(ordered.groups)
[0, 1, 1]
```

* **Parameters:**
  **op** – the operator whose terms to reorder.
* **Returns:**
  A new operator of the same type with its terms ordered by group index.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `op` is not a supported operator type (see
      [`OperatorTrait`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait)).
