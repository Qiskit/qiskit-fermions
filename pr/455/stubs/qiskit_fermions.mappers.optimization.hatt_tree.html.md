# hatt_tree

### hatt_tree(hamiltonian, num_modes=None, , weighted=False)

Grow a ternary tree adapted to a Hamiltonian, following the HATT algorithm.

The algorithm is topology-blind: it shapes the tree around the Hamiltonian’s term structure.
It works on the Majorana representation, so any other operator is first converted with
[`majorana_operator()`](qiskit_fermions.mappers.library.majorana_operator.html.md#qiskit_fermions.mappers.library.majorana_operator).

The tree is grown bottom-up, one node per step, following the HATT algorithm <sup>[1](#id2)</sup>. Each step
chooses the subtrees for a new node’s `X` and `Z` links so that the Hamiltonian’s Pauli
weight on that node’s qubit is as small as possible. The `Y` link is not chosen freely: it gets
the subtree holding the partner of the Majorana operator that the `X` side contributes, so the
two Majorana operators of a mode always meet on one qubit as `X` and `Y`. This is the
vacuum-preserving variant of the algorithm, and its bookkeeping matches how a
[`TernaryTreeEncoding`](qiskit_fermions.mappers.TernaryTreeEncoding.html.md#qiskit_fermions.mappers.TernaryTreeEncoding) pairs legs into Majorana operators. Node `j` of the returned tree
hosts mode `j`.

The search is greedy: it minimizes each step’s cost in turn, not their sum. Summed over all
steps, these costs are the total Pauli weight of the Hamiltonian’s Majorana terms as given, each
term’s image counted on its own. Nothing is normal-ordered, merged, or dropped first, so terms
that would cancel, and terms with negligible coefficients, all count. To target the Pauli weight
of the simplified mapped operator instead, normal-order and simplify the Majorana form yourself
before passing it in: its distinct terms then map to distinct Pauli strings, and the summed cost
equals the total Pauli weight of the mapped operator after the same simplification.

* **Parameters:**
  * **hamiltonian** ([*MajoranaOperator*](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator) *|* [*SupportsMajoranaOperator*](qiskit_fermions.protocols.SupportsMajoranaOperator.html.md#qiskit_fermions.protocols.SupportsMajoranaOperator)) – the operator to adapt to, either a [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator) or any operator
    implementing [`SupportsMajoranaOperator`](qiskit_fermions.protocols.SupportsMajoranaOperator.html.md#qiskit_fermions.protocols.SupportsMajoranaOperator).
  * **num_modes** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* *None*) – the number of fermionic modes, which is also the number of qubits. Inferred from
    the operator’s support when omitted, which is what you want unless the encoding should
    cover modes the Hamiltonian does not act on.
  * **weighted** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to weight each term by `|coefficient|` when scoring a candidate merge.
    The default, `False`, counts terms and ignores their coefficients, as in the HATT
    publication.
* **Returns:**
  The tree.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `num_modes` is not given and the operator acts on no mode, for example
        because it is empty or a constant.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `num_modes` is zero, or more than a ternary tree can hold.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the Hamiltonian acts outside `num_modes`.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if a coefficient is infinite or NaN.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `num_modes` is negative or does not fit in 32 bits.
* **Return type:**
  [TernaryTree](qiskit_fermions.mappers.TernaryTree.html.md#qiskit_fermions.mappers.TernaryTree)

```pycon
>>> from qiskit_fermions.mappers.library import majorana_operator
>>> from qiskit_fermions.mappers.optimization import hatt_tree
>>> from qiskit_fermions.operators import FermionOperator
>>>
>>> hamiltonian = FermionOperator.from_dict(
...     {((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 1.0}
... )
>>> # The conversion keeps all eight Majorana terms; four of them cancel.
>>> majorana = majorana_operator(hamiltonian)
>>> len(majorana), len(majorana.normal_ordered().simplify(1e-10))
(8, 2)
>>> tree = hatt_tree(majorana.normal_ordered().simplify(1e-10))
>>> tree.num_nodes, tree.num_legs
(2, 5)
```

#### NOTE
This is a greedy heuristic, and Pauli weight is not the circuit cost, so it does not always
beat Jordan-Wigner. Against Jordan-Wigner, reference 1 reports on average about 15% lower
Pauli weight and 26% fewer CNOT gates on molecular Hamiltonians, but also cases where
Jordan-Wigner gives fewer CNOT gates or a shallower circuit. Measure on your own Hamiltonian
rather than assuming an improvement.

#### NOTE
Reference 1 does not say how to break ties between equal-cost candidates. Ties go to the
lowest subtree indices (legs first, in Majorana order, then nodes in the order they were
created), so the result is deterministic. Because legs are numbered by mode, relabeling the
modes of the Hamiltonian changes which candidate wins a tie.

* <a id='id2'>**[1]**</a> Y. Liu, K. Yao, J. Hong, J. Froustey, E. Rrapaj, C. Iancu, G. Li, and Y. Shi, HATT: Hamiltonian Adaptive Ternary Tree for Optimizing Fermion-to-Qubit Mapping (2024), [arXiv:2409.02010](https://arxiv.org/abs/2409.02010).
