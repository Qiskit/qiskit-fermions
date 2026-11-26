# canonical_order

### canonical_order(op)

Returns a copy of an operator with its terms sorted into a canonical order.

The terms are sorted into a canonical order that depends only on each term’s structure (the
operator string it represents) and not on its coefficient. The order is therefore deterministic
for a given set of terms regardless of how the operator was assembled. The terms themselves are
left untouched — this only reorders them, it does not simplify or normal-order the operator.

This works for any of the built-in operator types implementing the
[`OperatorTrait`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait) protocol; the returned operator is of the same
type as the input.

#### NOTE
Any group indices (see e.g.
[`groups`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator.groups)) are preserved: each term carries
its group index along as it moves, since a group index is a per-term tag independent of term
order. Terms of the same group are simply no longer contiguous afterwards. If the input has
no groups, neither does the result.

```pycon
>>> from qiskit_fermions.operators import FermionOperator
>>> from qiskit_fermions.operators.terms.ordering import canonical_order
>>> op = FermionOperator.from_dict(
...     {
...         ((True, 1), (False, 0)): 1.0,  # a†_1 a_0
...         ((True, 0), (False, 1)): 2.0,  # a†_0 a_1
...     }
... )
>>> ordered = canonical_order(op)
>>> list(ordered.iter_terms())
[([(True, 0), (False, 1)], (2+0j)), ([(True, 1), (False, 0)], (1+0j))]
```

* **Parameters:**
  **op** – the operator whose terms to reorder.
* **Returns:**
  A new operator of the same type with its terms in canonical order.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `op` is not a supported operator type (see
      [`OperatorTrait`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait)).
