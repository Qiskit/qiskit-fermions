# group_terms_by_non_commutation

### group_terms_by_non_commutation(op)

Group the operator’s terms so that the terms sharing a group index mutually commute.

Colors the operator’s *clash graph* - the graph whose nodes are its terms and whose edges join
the pairs that do **not** commute, available as `op.non_commutation_edges()`. A proper coloring
places two non-commuting terms in different groups by construction. Group indices are assigned in
place, replacing any the operator already carried.

As for `op.non_commutation_edges()`, a term with an exactly zero coefficient clashes with nothing,
while every other term counts as written: simplify the operator first to remove near-zero and
duplicate terms, which would otherwise inflate the number of groups.

The coloring is the greedy *largest-first* heuristic (terms with the most clashes are placed
first), so the number of groups is an upper bound on the smallest achievable rather than a
minimum. The graph itself is never built: memory grows linearly with the number of terms, where
the list of clashing pairs grows quadratically. This makes the grouping usable on operators whose
`op.non_commutation_edges()` would not fit in memory. Building the graph from those edges
remains the way to color it with a different strategy.

* **Parameters:**
  **op** – the operator whose terms to group.
* **Raises:**
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `op` is not one of the four operator types.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `op` has more than $2^{32}$ terms, or is a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)
        with a term of degree above 20 and a non-zero coefficient, exactly when
        `op.non_commutation_edges()` raises. The operator’s groups are then left untouched.

#### NOTE
Whether these commuting groups agree with the line flow sets used to compile the time
evolution of lattice Hamiltonians depends on the operator representation, not on the lattice.
A flow set forbids two arrows that share a source or a target while permitting a head-to-tail
chain. A transfer $T_{jk}$ acts on the odd generator of its source and the even
generator of its target, so for the hopping terms of a [`TransferVertexOperator`](qiskit_fermions.operators.TransferVertexOperator.html.md#qiskit_fermions.operators.TransferVertexOperator) the
clash relation *is* that rule. Majorana and edge-vertex bilinears instead act on the same
generator of a shared mode however they meet, so there a head-to-tail chain does **not**
commute and the two partitions diverge. See the [grouping guide](../guides/grouping.html.md#grouping-explanation).

### Examples

```pycon
>>> from qiskit_fermions.operators import MajoranaOperator
>>> from qiskit_fermions.operators.terms.grouping import (
...     group_terms_by_non_commutation,
... )
>>>
>>> # built from the sparse arrays rather than a dictionary, because only the former fixes
>>> # the term order that the group indices below are paired with
>>> op = MajoranaOperator([1.0, 1.0, 1.0], [0, 1, 1, 2, 3, 4], [0, 2, 4, 6])
>>> group_terms_by_non_commutation(op)
>>> op.groups
[0, 1, 0]
```

The two clashing terms land in different groups, while the term commuting with both rejoins
the first group.
