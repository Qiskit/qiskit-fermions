# order_terms

### order_terms(op, key, , reverse=False)

Returns a copy of an operator with its terms reordered by a user-provided key.

Unlike [`canonical_order()`](qiskit_fermions.operators.terms.canonical_order.html.md#qiskit_fermions.operators.terms.canonical_order), which imposes a single fixed, coefficient-independent order,
this function lets you sort the terms by any criterion you like: the `key` function is applied
to each term exactly as [`sorted()`](https://docs.python.org/3/builtins/functions.html#sorted) would, and the terms are rebuilt into a new operator of
the same type in the resulting order. This is useful whenever the *order* of the terms matters
for a downstream computation (e.g. randomized product-formula sampling), since it gives you a
deterministic, reproducible order independent of how the operator happened to be assembled.

The terms themselves are left untouched – this only reorders them, it does not simplify or
normal-order the operator. This works for any of the built-in operator types implementing the
[`OperatorTrait`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait) protocol; the returned operator is of the same
type as the input.

#### NOTE
The `key` receives each term as the same tuple that the operator’s
[`iter_terms()`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.iter_terms) yields – or, when the operator
tracks groups (see [`groups`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.groups)), the longer tuple
from [`iter_terms_with_groups()`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.iter_terms_with_groups), whose trailing
element is the term’s group index. The exact layout of the term itself depends on the
operator type. Group indices are preserved – each term carries its group index along as it
moves – and if the input tracks no groups, neither does the result.

```pycon
>>> from qiskit_fermions.operators import FermionOperator
>>> from qiskit_fermions.operators.terms.ordering import order_terms
>>> op = FermionOperator.from_dict(
...     {
...         ((True, 1), (False, 0)): 1.0,  # a†_1 a_0, |coeff| = 1
...         ((True, 0), (False, 1)): 3.0,  # a†_0 a_1, |coeff| = 3
...     }
... )
>>> ordered = order_terms(op, key=lambda term: abs(term[1]), reverse=True)
>>> list(ordered.iter_terms())
[([(True, 0), (False, 1)], (3+0j)), ([(True, 1), (False, 0)], (1+0j))]
```

* **Parameters:**
  * **op** ([*OperatorTrait*](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait)) – the operator whose terms to reorder.
  * **key** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *]* *,* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any) *]*) – a function of a single term, used exactly as the `key` argument of [`sorted()`](https://docs.python.org/3/builtins/functions.html#sorted). It
    receives each term as the tuple described in the note above.
  * **reverse** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – if `True`, the terms are sorted in descending order of `key`.
* **Returns:**
  A new operator of the same type with its terms in the requested order.
* **Return type:**
  [*OperatorTrait*](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait)
