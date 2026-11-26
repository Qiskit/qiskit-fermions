<a id="qf-ternary-tree"></a>

# Ternary Trees

A ternary tree on $N$ nodes defines a fermion-to-qubit encoding of $N$ fermionic modes onto
$N$ qubits. Each node is a qubit carrying three downward **links**, labeled `X`, `Y` and
`Z`. A link either descends to a child node (an *edge*) or terminates (a *leg*), and each leg’s path
back to the root spells a Pauli string: crossing the link labeled $P$ that descends *from* node
$u$ contributes $P$ on qubit $u$, with the identity everywhere else.

Every tree of this shape yields a valid encoding, and the encodings named in the literature are simply
particular shapes:

| Encoding         | Construction                                    | Pauli weight                   |
|------------------|-------------------------------------------------|--------------------------------|
| Jordan-Wigner    | `qf_ternary_tree_chain(n, QfPauliLabel_Z, ...)` | $n$                            |
| Parity           | `qf_ternary_tree_chain(n, QfPauliLabel_X, ...)` | $n$                            |
| Balanced         | `qf_ternary_tree_breadth_first(n, 3, ...)`      | $\lceil \log_3(2n+1) \rceil$   |
| Binary-branching | `qf_ternary_tree_breadth_first(n, 2, ...)`      | $\lfloor \log_2 n \rfloor + 1$ |

The balanced tree’s weight is optimal over *all* fermion-to-qubit mappings, so the reduction from
$n$ to $\lceil \log_3(2n+1) \rceil$ is the reason to reach for one. Both trees use exactly
$n$ qubits, so the saving is in Pauli weight rather than qubit count.

These two constructors cover only the uniform families. The general interface is
[`qf_ternary_tree_new()`](#c.qf_ternary_tree_new), which takes the root and, per node, its parent and the label of the link
from that parent: the trees worth searching for are usually neither a chain nor uniform, since an
algorithm such as Bonsai grows a spanning tree of a device’s coupling graph and its branching
therefore follows that connectivity.

Not every encoding in the family has a convenience constructor. Bravyi-Kitaev, for instance, is a tree
that is not uniform, so it is built by computing its specification: the binary-branching tree above
matches its Pauli weight but is a different tree, whatever its label order.

Choosing a tree is its own problem. [`qf_ternary_tree_hatt()`](#c.qf_ternary_tree_hatt) shapes the tree around a
Hamiltonian’s own term structure, so the operators that actually appear in it land close together in
the tree.

Once built, a tree is compiled into a `QfTernaryTreeEncoding` with
[`qf_ternary_tree_encoding_new()`](#c.qf_ternary_tree_encoding_new), which pairs its legs into Majorana operators and builds their
Pauli strings once. That encoding is then passed to the mappers in [Mapper Library](qf-mappers-library.html.md#qf-mapper-library), and should
be reused across every operator mapped through it.

#### NOTE
Some identities that hold under Jordan-Wigner are properties of the *chain* rather than of ternary
trees. A vertex operator is the weight-1 Pauli $Z_l$ only there, where the two Majorana
strings’ $Z$ chains cancel; on the balanced tree at four modes it has weight 3.

| [`qf_ternary_tree_new()`](#c.qf_ternary_tree_new)                   | Construct a tree from a parent-and-label<br/>specification.                   |
|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`qf_ternary_tree_chain()`](#c.qf_ternary_tree_chain)                 | Construct a linear chain descending along one<br/>label.                      |
| [`qf_ternary_tree_breadth_first()`](#c.qf_ternary_tree_breadth_first)         | Construct a tree by breadth-first filling.                                    |
| [`qf_ternary_tree_num_nodes()`](#c.qf_ternary_tree_num_nodes)             | The number of nodes, i.e. modes and qubits.                                   |
| [`qf_ternary_tree_num_legs()`](#c.qf_ternary_tree_num_legs)              | The number of legs, always `2 * num_nodes + 1`.                               |
| [`qf_ternary_tree_max_weight()`](#c.qf_ternary_tree_max_weight)            | The largest Pauli weight of any single Majorana<br/>operator.                 |
| [`qf_ternary_tree_leg_labels()`](#c.qf_ternary_tree_leg_labels)            | The labels of a node’s links that terminate<br/>in a leg.                     |
| [`qf_ternary_tree_reparent()`](#c.qf_ternary_tree_reparent)              | Move a node, and the subtree below it, onto<br/>another node’s link.          |
| [`qf_ternary_tree_reroot()`](#c.qf_ternary_tree_reroot)                | Re-root the tree at a given node.                                             |
| [`qf_ternary_tree_relabel()`](#c.qf_ternary_tree_relabel)               | Permute the Pauli labels of a node’s links.                                   |
| [`qf_ternary_tree_free()`](#c.qf_ternary_tree_free)                  | Free a tree.                                                                  |
| [`qf_ternary_tree_encoding_new()`](#c.qf_ternary_tree_encoding_new)          | Compile a tree into the Pauli images of its<br/>Majorana operators.           |
| [`qf_ternary_tree_encoding_num_modes()`](#c.qf_ternary_tree_encoding_num_modes)    | The number of modes the encoding covers.                                      |
| [`qf_ternary_tree_encoding_max_weight()`](#c.qf_ternary_tree_encoding_max_weight)   | The largest Pauli weight of any single Majorana<br/>operator of the encoding. |
| [`qf_ternary_tree_encoding_total_parity()`](#c.qf_ternary_tree_encoding_total_parity) | Build the total fermionic parity observable of<br/>the encoding.              |
| [`qf_ternary_tree_encoding_free()`](#c.qf_ternary_tree_encoding_free)         | Free a compiled encoding.                                                     |
| [`qf_ternary_tree_hatt()`](#c.qf_ternary_tree_hatt)                  | Grow a tree adapted to a Hamiltonian.                                         |

---

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_hatt(const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*hamiltonian, uint32_t num_modes, bool weighted, QfTernaryTree \*\*out)

Grows a ternary tree adapted to a Hamiltonian, following the HATT algorithm. 

The algorithm is topology-blind: it shapes the tree around which Majorana operators appear
together in the Hamiltonian being mapped.

The tree is grown bottom-up. Each step picks the `X` and `Z` children of a new node to minimize
the Hamiltonian’s Pauli weight on its qubit, and gives the `Y` link the subtree holding the
partner of the Majorana operator on the `X` side. That is the vacuum-preserving variant of the
algorithm, and it matches how [`qf_ternary_tree_encoding_new()`](#c.qf_ternary_tree_encoding_new) pairs legs into Majorana
operators. Node `j` of the returned tree hosts mode `j`.

Summed over all steps, the cost is the total Pauli weight of the terms of `hamiltonian` as given,
each term’s image counted on its own: nothing is normal-ordered, merged or dropped first. If
`hamiltonian` is normal-ordered and simplified beforehand, its distinct terms map to distinct
Pauli strings, and the cost equals the total Pauli weight of the mapped operator after the same
simplification.

Ties between equal-cost candidates, which the reference leaves open, go to the lowest subtree
indices, legs first in Majorana order. So the result is deterministic, and relabeling the modes of
`hamiltonian` changes which candidate wins a tie.

```c
// gamma_0 gamma_3, one term of a hopping between modes 0 and 1.
uint32_t modes[2] = {0, 3};
QkComplex64 coeff = {0.0, 1.0};
QfMajoranaOperator *ham = qf_maj_op_zero();
qf_maj_op_add_term(ham, 2, modes, &coeff);

QfTernaryTree *tree;
QfExitCode code = qf_ternary_tree_hatt(ham, 2, false, &tree);
assert(code == QfExitCode_Success);

qf_ternary_tree_free(tree);
qf_maj_op_free(ham);
```

The tree that comes back is a valid encoding like any other, so it is passed to
[`qf_ternary_tree_encoding_new()`](#c.qf_ternary_tree_encoding_new) and then to the mappers exactly as a
[`qf_ternary_tree_breadth_first()`](#c.qf_ternary_tree_breadth_first) tree would be.

## Safety

<a id="group__qf__ternary__tree_1autotoc_md10"></a>

`hamiltonian` must be a valid pointer to a `QfMajoranaOperator`, and `out` must be non-null and aligned. A null or misaligned pointer is reported rather than dereferenced. 

#### WARNING
This is a greedy heuristic, and Pauli weight is not the circuit cost, so it does not always
win: the reference itself reports cases where Jordan-Wigner gives fewer CNOT gates or a
shallower circuit. Measure on your own Hamiltonian rather than assuming an improvement.

* **Parameters:**
  * **hamiltonian** – A pointer to the Majorana operator to adapt to. 
  * **num_modes** – The number of fermionic modes, which is also the number of qubits. 
  * **weighted** – Whether to weight each term by the magnitude of its coefficient when scoring a candidate merge. `false` counts terms and ignores their coefficients, as in the HATT publication. 
  * **out** – A pointer to where the created tree will be written on success. It is left untouched if the search fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred. In particular, a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` is returned if `num_modes` is zero or larger than a ternary tree can hold, if `hamiltonian` acts on a mode the requested encoding does not cover, or if a coefficient is infinite or NaN. A null or misaligned pointer is reported as `[QfExitCode_NullPointerError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1abde2a279ad83c8d85eb70c5297d2ca7c)` or `[QfExitCode_AlignmentError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a52d26ce30c09015d37c8287c8fd106b5)`.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_new(uint32_t num_nodes, uint32_t root, const uint32_t \*parents, const QfPauliLabel \*labels, QfTernaryTree \*\*out)

Constructs a ternary tree from a parent-and-label specification. 

A ternary tree on $N$ nodes defines a fermion-to-qubit encoding of $N$ fermionic modes
onto $N$ qubits. Each node is a qubit carrying three downward **links**, labeled `X`,
`Y` and `Z`. A link either descends to a child node (an *edge*) or terminates (a *leg*), and
each leg’s path back to the root spells a Pauli string: crossing the link labeled $P$ that
descends *from* node $u$ contributes $P$ on qubit $u$.

Every tree of this shape yields a valid encoding, and the familiar ones are particular shapes. This
is the general constructor; [`qf_ternary_tree_chain()`](#c.qf_ternary_tree_chain) and
[`qf_ternary_tree_breadth_first()`](#c.qf_ternary_tree_breadth_first) build the two uniform families more conveniently.

The tree must be freed with [`qf_ternary_tree_free()`](#c.qf_ternary_tree_free).

## Example

```c
// A tree in which node 0, the root, branches three ways. The root's entries are never read,
// so any value works.
#define IGNORED 0

uint32_t parents[4] = {IGNORED, 0, 0, 0};
QfPauliLabel labels[4] = {IGNORED, QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};

QfTernaryTree *tree;
QfExitCode exit = qf_ternary_tree_new(4, 0, parents, labels, &tree);

assert(exit == QfExitCode_Success);
```

## Safety

<a id="group__qf__ternary__tree_1autotoc_md11"></a>

`parents` and `labels` must each point to an array of at least `num_nodes` initialized values (they are not read when `num_nodes` is zero), and `out` must be non-null and aligned. The entries of `labels` need not hold valid labels: an invalid one is reported as `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)`. 

* **Parameters:**
  * **num_nodes** – The number of nodes, which is both the number of fermionic modes and the number of qubits the encoding uses. 
  * **root** – The index of the root node. 
  * **parents** – An array of `num_nodes` entries: the index of each node’s parent. The root has no parent, so its entry is never read and may hold any value, zero included. The array is not read when `num_nodes` is zero, and may then be `NULL`. 
  * **labels** – An array of `num_nodes` `QfPauliLabel` values: the label of the link descending from each node’s parent to it. The root has no such link, so its entry is never read and may hold any value, zero included. The array is not read when `num_nodes` is zero, and may then be `NULL`. 
  * **out** – A pointer to where the created tree will be written on success. It is left untouched if construction fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred. In particular, a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` is returned if the specification does not describe a single connected ternary tree – if it is empty, `root` is not a node, a label is repeated on one node, a non-root entry of `labels` is not a `QfPauliLabel` value, a parent is not a node, or some node is unreachable from the root.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_chain(uint32_t num_nodes, uint8_t label, QfTernaryTree \*\*out)

Constructs a linear chain descending along one label. 

`QfPauliLabel_Z` gives the Jordan-Wigner encoding and `QfPauliLabel_X` the parity encoding.
Both have Pauli weight `num_nodes`, the worst a tree on that many nodes can do; see
[`qf_ternary_tree_breadth_first()`](#c.qf_ternary_tree_breadth_first) for the optimal one.

This is only a convenience: it is [`qf_ternary_tree_breadth_first()`](#c.qf_ternary_tree_breadth_first) with a branching of
one, whose `order` then needs just the one label.

The tree must be freed with [`qf_ternary_tree_free()`](#c.qf_ternary_tree_free).

## Example

```c
// The Jordan-Wigner encoding of four modes: a chain descending along Z.
QfTernaryTree *chain;
QfExitCode exit = qf_ternary_tree_chain(4, QfPauliLabel_Z, &chain);

assert(exit == QfExitCode_Success);
// A chain is the worst case: one Majorana operator spans the whole register.
assert(qf_ternary_tree_max_weight(chain) == 4);

qf_ternary_tree_free(chain);
```

## Safety

<a id="group__qf__ternary__tree_1autotoc_md12"></a>

`out` must be non-null and aligned. `label` need not hold a valid label: an invalid one is reported as `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)`. 

* **Parameters:**
  * **num_nodes** – The number of nodes, and hence of modes and qubits. 
  * **label** – The link the chain descends along, as a `QfPauliLabel` value. It is passed as a plain byte, so that an invalid value is reported as a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)`. 
  * **out** – A pointer to where the created tree will be written on success.
* **Returns:**
  An exit code. This is `>0` if an error occurred, in particular a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if `num_nodes` is zero or `label` is not a `QfPauliLabel` value.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_breadth_first(uint32_t num_nodes, uint8_t branching, const QfPauliLabel \*order, QfTernaryTree \*\*out)

Constructs a tree by attaching up to `branching` children to each node in turn. 

A `branching` of 3 gives the balanced tree, whose Pauli weight
$\lceil \log_3(2N+1) \rceil$ is optimal over *all* fermion-to-qubit mappings. A `branching`
of 2 gives a binary-branching tree of weight $\lfloor \log_2 N \rfloor + 1$. That matches the
Bravyi-Kitaev weight but is *not* that encoding, for any `order`: Bravyi-Kitaev’s tree is the
irregular partial-sum shape, built through [`qf_ternary_tree_new()`](#c.qf_ternary_tree_new).

`order` is part of the encoding’s definition: it changes which Pauli strings are produced, though
not their weight.

The tree must be freed with [`qf_ternary_tree_free()`](#c.qf_ternary_tree_free).

## Example

```c
QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};

QfTernaryTree *tree;
QfExitCode exit = qf_ternary_tree_breadth_first(13, 3, order, &tree);

assert(exit == QfExitCode_Success);
// The balanced tree needs weight 3 where a chain of 13 modes needs 13.
assert(qf_ternary_tree_max_weight(tree) == 3);
```

## Safety

<a id="group__qf__ternary__tree_1autotoc_md13"></a>

`order` must point to an array of at least 3 initialized `QfPauliLabel` entries, and `out` must be non-null and aligned. The entries need not hold valid labels: an invalid one is reported as `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)`. 

* **Parameters:**
  * **num_nodes** – The number of nodes, and hence of modes and qubits. 
  * **branching** – How many children to attach per node, from 1 to 3. 
  * **order** – The three labels to attach children along, in order. Only the first `branching` entries are used, and they must be distinct, but all three must be `QfPauliLabel` values. 
  * **out** – A pointer to where the created tree will be written on success.
* **Returns:**
  An exit code. This is `>0` if an error occurred, in particular if `num_nodes` is zero, `branching` is not between 1 and 3, an entry of `order` is not a `QfPauliLabel` value, or the used part of `order` repeats a label.

### uint32_t qf_ternary_tree_num_nodes(const QfTernaryTree \*tree)

Returns the number of nodes of a tree, i.e. its mode and qubit count. 

## Safety

<a id="group__qf__ternary__tree_1autotoc_md14"></a>

`tree` must be a non-null, aligned pointer to a `QfTernaryTree`. 

* **Parameters:**
  * **tree** – A pointer to the tree.
* **Returns:**
  The number of nodes.

### uint32_t qf_ternary_tree_num_legs(const QfTernaryTree \*tree)

Returns the number of legs of a tree, always `2 * num_nodes + 1`. 

This holds for every shape: a node with fewer children has more *legs*, not fewer Pauli strings. Of
these, `2 * num_nodes` pair into the Majorana operators and exactly one is left over, carrying the
total fermionic parity – see [`qf_ternary_tree_encoding_total_parity()`](#c.qf_ternary_tree_encoding_total_parity).

## Safety

<a id="group__qf__ternary__tree_1autotoc_md15"></a>

`tree` must be a non-null, aligned pointer to a `QfTernaryTree`. 

* **Parameters:**
  * **tree** – A pointer to the tree.
* **Returns:**
  The number of legs.

### uint32_t qf_ternary_tree_max_weight(const QfTernaryTree \*tree)

Returns the largest Pauli weight of any single Majorana operator the tree generates. 

This is the figure of merit a tree is chosen for. Note that it is the weight of one *Majorana*
operator, and that no Pauli term of a fermionic creation or annihilation operator is heavier,
since that is a sum of two Majorana operators. The two terms generally act on different qubits,
though, so the operator as a whole can be supported on more.

## Safety

<a id="group__qf__ternary__tree_1autotoc_md16"></a>

`tree` must be a non-null, aligned pointer to a `QfTernaryTree`. 

* **Parameters:**
  * **tree** – A pointer to the tree.
* **Returns:**
  The maximum Pauli weight.

### void qf_ternary_tree_free(QfTernaryTree \*tree)

Frees a ternary tree. 

## Safety

<a id="group__qf__ternary__tree_1autotoc_md17"></a>

`tree` must be a pointer written by one of the `qf_ternary_tree_*` functions that create a tree (constructors and edits alike) and not already freed, or `NULL`. 

* **Parameters:**
  * **tree** – A pointer to the tree to free. Passing `NULL` is a no-op.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_leg_labels(const QfTernaryTree \*tree, uint32_t node, QfPauliLabel \*out_labels, uint64_t \*out_len)

Writes the labels of a node’s links that terminate in a leg. 

These are the links a subtree may be attached to with [`qf_ternary_tree_reparent()`](#c.qf_ternary_tree_reparent). A node
with three children writes nothing and reports a length of 0.

Because a node has exactly three links, the output is bounded and the caller supplies the buffer;
nothing is allocated and there is nothing to free.

## Safety

<a id="group__qf__ternary__tree_1autotoc_md18"></a>

`tree` must be a valid `QfTernaryTree` pointer, and `out_labels` must point to space for at least 3 `QfPauliLabel` values. `out_len` must be non-null and aligned. A null or misaligned pointer is reported as `[QfExitCode_NullPointerError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1abde2a279ad83c8d85eb70c5297d2ca7c)` or `[QfExitCode_AlignmentError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a52d26ce30c09015d37c8287c8fd106b5)`. 

* **Parameters:**
  * **tree** – A pointer to the tree. 
  * **node** – The node to inspect. 
  * **out_labels** – A pointer to an array of at least **3** `QfPauliLabel` values, into which the labels are written. 
  * **out_len** – A pointer to where the number written will be stored, between 0 and 3. Both outputs are left untouched if an error occurs.
* **Returns:**
  An exit code. This is `>0` if an error occurred, in particular a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if `node` is out of range.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_reparent(const QfTernaryTree \*tree, uint32_t node, uint32_t new_parent, uint8_t label, QfTernaryTree \*\*out)

Moves a node, and the subtree below it, onto another node’s link. 

The subtree below `node` moves with it. This is the primitive a local search over trees spends
most of its time in. Use [`qf_ternary_tree_leg_labels()`](#c.qf_ternary_tree_leg_labels) to find the links that can accept
a subtree. Moving `node` to the link it already hangs off succeeds and returns an equal tree.

The input tree is left untouched: the edit returns a new tree, which must be freed with
[`qf_ternary_tree_free()`](#c.qf_ternary_tree_free).

## Safety

<a id="group__qf__ternary__tree_1autotoc_md19"></a>

`tree` must be a valid `QfTernaryTree` pointer, and `out` must be non-null and aligned. A null or misaligned pointer is reported as `[QfExitCode_NullPointerError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1abde2a279ad83c8d85eb70c5297d2ca7c)` or `[QfExitCode_AlignmentError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a52d26ce30c09015d37c8287c8fd106b5)`. `label` need not hold a valid label: an invalid one is reported as `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)`. 

* **Parameters:**
  * **tree** – A pointer to the tree to edit. It is not modified. 
  * **node** – The node to move. This may not be the root. 
  * **new_parent** – The node to attach it to. This may not be `node` itself or lie inside its subtree. 
  * **label** – Which of `new_parent`’s links to use. It must currently terminate in a leg, unless it is the link `node` already hangs off. 
  * **out** – A pointer to where the new tree will be written on success. It is left untouched if the edit fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred. In particular, a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` is returned if `node` is the root, if `new_parent` is `node` or lies inside its subtree (which would create a cycle), if the link already descends to another child, if either index is out of range, or if `label` is not a `QfPauliLabel`.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_reroot(const QfTernaryTree \*tree, uint32_t node, QfTernaryTree \*\*out)

Re-roots a tree at a given node. 

Each link on the path from `node` to the old root is re-pointed the other way. A reversed link
has to occupy a slot that is free at its new parent: it keeps its old label when that slot is
free and otherwise takes the lowest free slot in `X`, `Y`, `Z` order. The rule is
deterministic, but a label that had to change is not changed back, so re-rooting is not in
general an exact inverse of itself. Re-rooting at the current root succeeds and returns an equal
tree.

The new tree must be freed with [`qf_ternary_tree_free()`](#c.qf_ternary_tree_free).

## Safety

<a id="group__qf__ternary__tree_1autotoc_md20"></a>

`tree` must be a valid `QfTernaryTree` pointer, and `out` must be non-null and aligned. A null or misaligned pointer is reported as `[QfExitCode_NullPointerError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1abde2a279ad83c8d85eb70c5297d2ca7c)` or `[QfExitCode_AlignmentError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a52d26ce30c09015d37c8287c8fd106b5)`. 

#### NOTE
The at-most-two-children requirement is not a formality: re-rooting gives `node` a fourth
downward link (the one pointing back at its former parent), and a node has only three.
Re-rooting and [`qf_ternary_tree_relabel()`](#c.qf_ternary_tree_relabel) never change which nodes are joined, so a
search that should reach trees of a different shape also needs
[`qf_ternary_tree_reparent()`](#c.qf_ternary_tree_reparent).

* **Parameters:**
  * **tree** – A pointer to the tree to edit. It is not modified. 
  * **node** – The node to make the root. It must have at most two children. 
  * **out** – A pointer to where the new tree will be written on success. It is left untouched if the edit fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred, in particular a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if `node` already has three children or is out of range.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_relabel(const QfTernaryTree \*tree, uint32_t node, const QfPauliLabel \*permutation, QfTernaryTree \*\*out)

Permutes the Pauli labels of a node’s three links. 

This changes which Pauli strings the encoding produces without moving anything, so no leg changes
depth and [`qf_ternary_tree_max_weight()`](#c.qf_ternary_tree_max_weight) is unchanged. It is the only edit guaranteed to
leave every depth unchanged. Only the links that descend to a child are stored, and the legs fill
whichever slots remain, so relabeling a leaf returns an equal tree.

The new tree must be freed with [`qf_ternary_tree_free()`](#c.qf_ternary_tree_free).

## Safety

<a id="group__qf__ternary__tree_1autotoc_md21"></a>

`tree` must be a valid `QfTernaryTree` pointer, `permutation` must point to 3 initialized `QfPauliLabel` values, and `out` must be non-null and aligned. A null or misaligned pointer is reported as `[QfExitCode_NullPointerError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1abde2a279ad83c8d85eb70c5297d2ca7c)` or `[QfExitCode_AlignmentError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a52d26ce30c09015d37c8287c8fd106b5)`. 

* **Parameters:**
  * **tree** – A pointer to the tree to edit. It is not modified. 
  * **node** – The node to relabel. 
  * **permutation** – A pointer to an array of **3** `QfPauliLabel` values, which must be a permutation of `X`, `Y`, `Z`. Entry `k` becomes the new label of the link currently labeled `QfPauliLabel_X`, `_Y`, `_Z` respectively. 
  * **out** – A pointer to where the new tree will be written on success. It is left untouched if the edit fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred, in particular a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if `permutation` repeats a label, holds a value that is not a `QfPauliLabel`, or `node` is out of range.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_encoding_new(const QfTernaryTree \*tree, const uint32_t \*mode_map, QfTernaryTreeEncoding \*\*out)

Compiles a ternary tree into the Pauli images of its Majorana operators. 

Pairing the tree’s legs into Majorana operators and building their Pauli strings is done once,
here, rather than per mapped operator, so one encoding should be reused across every operator
mapped through it. The encoding carries no qubit count of its own: the mappers take one, so a
single encoding can serve registers of different widths.

The pairing follows Algorithm 1 of [[Bonsai]](#bonsai): from each node, descend its `X` link and then `Z`
links until reaching a leg, which carries $\gamma$ of that node’s mode; the `Y` link
likewise gives $\gamma'$. This is what makes the all-zero computational state the fermionic
vacuum, so it is not an adjustable convention.

The encoding must be freed with [`qf_ternary_tree_encoding_free()`](#c.qf_ternary_tree_encoding_free).

## Example

`mode_map` is indexed by *node* and names the *mode* that node serves, so it reassigns which
mode’s Majorana operators each pair of Pauli strings represents without changing the strings
themselves. Pass `NULL` when node `j` should serve mode `j`.

```c
QfTernaryTree *tree;
qf_ternary_tree_chain(4, QfPauliLabel_Z, &tree);

// Node j serves mode j.
QfTernaryTreeEncoding *identity;
QfExitCode exit = qf_ternary_tree_encoding_new(tree, NULL, &identity);
assert(exit == QfExitCode_Success);

// Reverse the assignment: node 0 now serves mode 3, node 1 mode 2, and so on.
uint32_t mode_map[4] = {3, 2, 1, 0};
QfTernaryTreeEncoding *reversed;
exit = qf_ternary_tree_encoding_new(tree, mode_map, &reversed);
assert(exit == QfExitCode_Success);

// Both cover the same modes; only which node serves which mode differs.
assert(qf_ternary_tree_encoding_num_modes(reversed) == 4);

qf_ternary_tree_encoding_free(identity);
qf_ternary_tree_encoding_free(reversed);
qf_ternary_tree_free(tree);
```

Reordering the modes of a fixed tree is one of the cheapest ways to lower the Pauli weight of a
particular Hamiltonian, which is why the assignment is separate from the tree.

## Safety

<a id="group__qf__ternary__tree_1autotoc_md22"></a>

`tree` must be `NULL` or a valid `QfTernaryTree` pointer, `mode_map` must be `NULL` or point to an array of at least `qf_ternary_tree_num_nodes(tree)` initialized `uint32_t` values, and `out` must be `NULL` or aligned. 

* **Parameters:**
  * **tree** – A pointer to the tree defining the encoding. 
  * **mode_map** – Which mode each node serves, as an array of `qf_ternary_tree_num_nodes(tree)` entries indexed by node, or `NULL` for the identity assignment. 
  * **out** – A pointer to where the created encoding will be written on success.
* **Returns:**
  An exit code. This is `>0` if an error occurred. In particular, a `[QfExitCode_NullPointerError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1abde2a279ad83c8d85eb70c5297d2ca7c)` is returned if `tree` or `out` is `NULL`, and a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if `mode_map` is not a permutation of the tree’s nodes.

### uint32_t qf_ternary_tree_encoding_num_modes(const QfTernaryTreeEncoding \*encoding)

Returns the number of fermionic modes a compiled encoding covers. 

## Safety

<a id="group__qf__ternary__tree_1autotoc_md23"></a>

`encoding` must be a non-null, aligned pointer to a `QfTernaryTreeEncoding`. 

* **Parameters:**
  * **encoding** – A pointer to the encoding.
* **Returns:**
  The number of modes, equal to the tree’s node count.

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_ternary_tree_encoding_total_parity(const QfTernaryTreeEncoding \*encoding, uint32_t num_qubits, QkObs \*\*out)

Builds the total fermionic parity observable of a compiled encoding. 

A tree’s $2N+1$ legs pair into $2N$ Majorana operators, leaving exactly one over, and
that leftover is the total parity $\prod_j (1 - 2 n_j)$.

#### IMPORTANT
This is an **observable to measure, not a constraint to impose**. Both eigenvalues are
physical, so do not project a state onto the $+1$ sector: a ternary-tree encoding spends
one qubit per mode and is a unitary isomorphism onto the *whole* $2^N$-dimensional space,
with both parity sectors represented at multiplicity $2^{N-1}$ and no unphysical subspace
to project away.

Note also that the Pauli string is *not* in general the all-$Z$ string: it is the leg reached
by the all-$Z$ *path*, which acts as the identity on every node not on that path. On the
balanced tree at four modes it is $Z_3 Z_0$.

## Example

```c
QkObs *parity;
QfExitCode exit = qf_ternary_tree_encoding_total_parity(encoding, 4, &parity);

assert(exit == QfExitCode_Success);

qk_obs_free(parity);
```

## Safety

<a id="group__qf__ternary__tree_1autotoc_md24"></a>

`encoding` must be `NULL` or a valid `QfTernaryTreeEncoding` pointer, and `out` must be `NULL` or aligned. 

* **Parameters:**
  * **encoding** – A pointer to the compiled encoding. 
  * **num_qubits** – The number of qubits for the resulting observable. This must be at least the encoding’s mode count; a larger value pads with the identity. 
  * **out** – A pointer to where the total-parity observable will be written on success: a single :math:`Z` string with coefficient :math:`+1`. The caller takes ownership and must free it with `qk_obs_free`. It is left untouched if the construction fails.
* **Returns:**
  An exit code. This is `>0` if an error occurred. In particular, a `[QfExitCode_NullPointerError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1abde2a279ad83c8d85eb70c5297d2ca7c)` is returned if `encoding` or `out` is `NULL`, and a `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if `num_qubits` is below the encoding’s mode count.

### uint32_t qf_ternary_tree_encoding_max_weight(const QfTernaryTreeEncoding \*encoding)

Returns the largest Pauli weight of any single Majorana operator of a compiled encoding. 

This is the encoding-level counterpart of [`qf_ternary_tree_max_weight()`](#c.qf_ternary_tree_max_weight), and always agrees
with it: although it ranges only over the `2N` legs that carry a Majorana operator, the leftover
parity leg is never deeper than all of them. It lets an encoding report its weight once the tree it
was compiled from has been freed.

## Safety

<a id="group__qf__ternary__tree_1autotoc_md25"></a>

`encoding` must be a non-null, aligned pointer to a `QfTernaryTreeEncoding`. 

* **Parameters:**
  * **encoding** – A pointer to the encoding.
* **Returns:**
  The largest weight over the encoding’s `2N` Majorana operators.

### void qf_ternary_tree_encoding_free(QfTernaryTreeEncoding \*encoding)

Frees a compiled ternary-tree encoding. 

## Safety

<a id="group__qf__ternary__tree_1autotoc_md26"></a>

`encoding` must be a pointer returned by `[qf_ternary_tree_encoding_new](#group__qf__ternary__tree_1ga78d34b6790a65aa3ea8acc620961ae74)` and not already freed, or `NULL`. 

* **Parameters:**
  * **encoding** – A pointer to the encoding to free. Passing `NULL` is a no-op.
