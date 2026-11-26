# TernaryTree

### *class* TernaryTree(spec)

A ternary tree defining a fermion-to-qubit encoding.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A ternary tree on $N$ nodes encodes $N$ fermionic modes onto $N$ qubits. Each
node is a qubit carrying three downward **links**, labeled `X`, `Y` and `Z`. A link
either descends to a child node (an *edge*) or terminates (a *leg*). Each leg’s upward path to
the root spells a Pauli string: crossing the link labeled $P$ that descends *from* node
$u$ contributes $P$ on qubit $u$, with the identity everywhere else.

Every tree of this shape yields a valid encoding, so the constructor accepts an arbitrary one.
The familiar encodings are particular shapes:

| Encoding                      | Construction                             | Pauli weight                          |
|-------------------------------|------------------------------------------|---------------------------------------|
| Jordan-Wigner                 | `TernaryTree.chain(n, "Z")`              | $n$                                   |
| Parity                        | `TernaryTree.chain(n, "X")`              | $n$                                   |
| Balanced <sup>[1](#id3)</sup> | `TernaryTree.breadth_first(n, 3)`        | $\lceil \log_3(2n+1) \rceil$, optimal |
| Binary-branching              | `TernaryTree.breadth_first(n, 2, "ZXY")` | $\lfloor \log_2 n \rfloor + 1$        |

[`TernaryTree.chain()`](#qiskit_fermions.mappers.TernaryTree.chain) and [`TernaryTree.breadth_first()`](#qiskit_fermions.mappers.TernaryTree.breadth_first) cover the *uniform* families
only, which is why the constructor also takes an arbitrary tree: a parent-and-label
specification, one entry per node, giving that node’s parent and the label of the link
descending to it (`None` for the root). The trees that motivate the framework need it – the
Bonsai algorithm <sup>[2](#id4)</sup> grows a spanning tree of a device’s coupling graph, so its branching
follows that connectivity and is neither a chain nor uniform.

Not every encoding in the family has a convenience constructor: Bravyi-Kitaev, for instance, is
a tree that is not uniform, so it is built by computing its specification. The
[mappers guide](../guides/mappers.html.md#bravyi-kitaev-by-hand) works that case through as an example.

## Usage

```pycon
>>> from qiskit_fermions.mappers import TernaryTree
>>> tree = TernaryTree.breadth_first(4, 3)
>>> tree.num_nodes, tree.num_legs, tree.max_weight
(4, 9, 2)
```

The balanced tree’s weight grows logarithmically where Jordan-Wigner’s grows linearly:

```pycon
>>> TernaryTree.chain(13, "Z").max_weight
13
>>> TernaryTree.breadth_first(13, 3).max_weight
3
```

An arbitrary tree is given as one `(parent, label)` pair per node, with `None` for the root:

```pycon
>>> tree = TernaryTree([None, (0, "X"), (0, "Y"), (1, "Z")])
>>> tree.num_nodes, tree.num_legs
(4, 9)
```

Two trees compare equal when they have the same shape, labels and node numbering, however they
were built, which is what lets a set or dictionary deduplicate the candidates a tree search
produces. A chain, for instance, is just [`TernaryTree.breadth_first()`](#qiskit_fermions.mappers.TernaryTree.breadth_first) with a branching of
one, and only the first label of its `order` matters:

```pycon
>>> chain = TernaryTree.chain(3, "Z")
>>> chain == TernaryTree.breadth_first(3, 1, "ZXY")
True
>>> chain == TernaryTree.breadth_first(3, 1, "ZYX")
True
>>> len({chain, TernaryTree.breadth_first(3, 1, "ZXY"), TernaryTree.chain(3, "X")})
2
```

A tree is what a search produces and what defines an encoding, so it is the thing worth carrying
between processes, and it can be pickled:

```pycon
>>> import pickle
>>> tree = TernaryTree.breadth_first(4, 3)
>>> pickle.loads(pickle.dumps(tree)) == tree
True
```

* <a id='id3'>**[1]**</a> Z. Jiang, A. Kalev, W. Mruczkiewicz and H. Neven, Optimal fermion-to-qubit mapping via ternary trees with applications to reduced quantum states learning, Quantum 4, 276 (2020), [arXiv:1910.10746](https://arxiv.org/abs/1910.10746).
* <a id='id4'>**[2]**</a> A. Miller, Z. Zimborás, S. Knecht, S. Maniscalco and G. García-Pérez, Bonsai algorithm: grow your own fermion-to-qubit mappings, PRX Quantum 4, 030314 (2023), [arXiv:2212.09731](https://arxiv.org/abs/2212.09731).

### Attributes

#### max_weight

The largest Pauli weight of any single Majorana operator this tree generates.

No Pauli term of a fermionic creation or annihilation operator is heavier than this, since
it is a sum of two Majorana operators. The two terms generally act on different qubits,
though, so the operator as a whole can be supported on more.

#### num_legs

The number of legs, always `2 * num_nodes + 1`.

This holds for every shape: a node with fewer children has more legs, not fewer strings. Of
these, `2 * num_nodes` pair into the Majorana operators and one is left over, carrying the
total fermionic parity.

#### num_nodes

The number of nodes, which is both the number of modes and the number of qubits.

#### root

The root node.

### Methods

#### *static* breadth_first(num_nodes, branching, order='XYZ')

Build a tree by attaching up to `branching` children to each node in turn.

`branching=3` gives the balanced tree, whose Pauli weight
$\lceil \log_3(2n+1) \rceil$ is optimal over all fermion-to-qubit mappings.

#### NOTE
`branching=2` gives a binary-branching tree of weight
$\lfloor \log_2 n \rfloor + 1$. That matches the Bravyi-Kitaev weight but is *not*
that encoding, for any `order`: Bravyi-Kitaev’s tree is the irregular partial-sum
shape. See the [mappers guide](../guides/mappers.html.md#bravyi-kitaev-by-hand) for how to build it.

* **Parameters:**
  * **num_nodes** – the number of modes, and hence qubits.
  * **branching** – how many children to attach per node, from 1 to 3.
  * **order** – the labels to attach children along, as a 3-character string. Changing it changes
    which Pauli strings the encoding produces, though not their weight.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `num_nodes` is zero, `branching` is not between 1 and 3, `order`
        is not 3 Pauli labels, or its first `branching` labels are not distinct.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `num_nodes` is negative or does not fit in 32 bits.

#### *static* chain(num_nodes, label)

Build a linear chain descending along `label`.

`label="Z"` is the Jordan-Wigner encoding and `label="X"` the parity encoding; both have
Pauli weight `num_nodes`.

This is only a convenience: it is [`breadth_first()`](#qiskit_fermions.mappers.TernaryTree.breadth_first) with `branching=1`, where
`order` needs just the one label.

* **Parameters:**
  * **num_nodes** – the number of modes, and hence qubits.
  * **label** – the link the chain descends along.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `num_nodes` is zero or `label` is not a Pauli label.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `num_nodes` is negative or does not fit in 32 bits.

#### children(node)

Return the children of `node`, as a `{label: child}` mapping over its edges.

Links that terminate in a leg are omitted.

```pycon
>>> from qiskit_fermions.mappers import TernaryTree
>>> TernaryTree.breadth_first(4, 3).children(0)
{'X': 1, 'Y': 2, 'Z': 3}
```

* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `node` is out of range.

#### leg_labels(node)

Return the labels of `node`’s links that terminate in a leg.

These are the links a subtree can be attached to with [`reparent()`](#qiskit_fermions.mappers.TernaryTree.reparent). A node with three
children returns an empty list.

* **Parameters:**
  **node** – the node to inspect.
* **Returns:**
  The labels of the links that terminate in a leg, in `X`, `Y`, `Z` order.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `node` is out of range.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `node` is negative or does not fit in 32 bits.

```pycon
>>> from qiskit_fermions.mappers import TernaryTree
>>> tree = TernaryTree.chain(3, "Z")
>>> tree.leg_labels(0), tree.leg_labels(2)
(['X', 'Y'], ['X', 'Y', 'Z'])
```

The chain’s interior nodes each spend their `Z` link on the next node, so only the last one
has all three free.

#### leg_paths()

Return the Pauli string of every leg, as `(qubit, label)` pairs ordered leaf-to-root.

```pycon
>>> from qiskit_fermions.mappers import TernaryTree
>>> TernaryTree.chain(2, "Z").leg_paths()
[[(0, 'X')], [(0, 'Y')], [(1, 'X'), (0, 'Z')], [(1, 'Y'), (0, 'Z')], [(1, 'Z'), (0, 'Z')]]
```

The qubits within a path are *not* sorted: they follow the tree’s depth rather than the node
numbering. Use [`TernaryTreeEncoding.majorana_image()`](qiskit_fermions.mappers.TernaryTreeEncoding.html.md#qiskit_fermions.mappers.TernaryTreeEncoding.majorana_image) for the sorted form the mapped
observables use.

#### relabel(node, permutation)

Return a new tree with `node`’s three links relabeled.

`permutation[k]` becomes the new label of the link currently labeled `"XYZ"[k]`. This
changes which Pauli strings the encoding produces without moving anything, so no leg changes
depth and [`max_weight`](#qiskit_fermions.mappers.TernaryTree.max_weight) is unchanged. Only the links that descend to a child are
stored, and the legs fill whichever slots remain, so relabeling a leaf returns an equal tree.

* **Parameters:**
  * **node** – the node to relabel.
  * **permutation** – the three labels, as a 3-character string, in `X`, `Y`, `Z` slot order.
* **Returns:**
  A new tree.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `node` is out of range.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `permutation` is not a permutation of `"XYZ"`.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `node` is negative or does not fit in 32 bits.

```pycon
>>> from qiskit_fermions.mappers import TernaryTree
>>> tree = TernaryTree.breadth_first(4, 3)
>>> shuffled = tree.relabel(0, "YZX")
>>> shuffled.max_weight == tree.max_weight
True
>>> shuffled.leg_paths()[0] != tree.leg_paths()[0]
True
```

Relabeling is the only edit guaranteed to leave every depth unchanged.

#### reparent(node, new_parent, label)

Return a new tree in which `node` hangs off `new_parent`’s `label` link.

The subtree rooted at `node` moves with it. This is the primitive a local search spends
most of its time in. Moving `node` to the link it already hangs off returns an equal tree.

* **Parameters:**
  * **node** – the node to move. May not be the root.
  * **new_parent** – the node to attach it to. May not be `node` itself or lie inside its
    subtree.
  * **label** – which of `new_parent`’s links to use. Must currently terminate in a leg (see
    [`leg_labels()`](#qiskit_fermions.mappers.TernaryTree.leg_labels)), unless it is the link `node` already hangs off.
* **Returns:**
  A new tree. This class is immutable, so the original is untouched.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if either index is out of range.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `node` is the root.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `new_parent` is `node` itself.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `new_parent`’s `label` link already descends to another child.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `new_parent` lies inside `node`’s subtree, which would create a cycle.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if either index is negative or does not fit in 32 bits.

```pycon
>>> from qiskit_fermions.mappers import TernaryTree
>>> chain = TernaryTree.chain(6, "Z")
>>> chain.max_weight
6
>>> # Peel the chain's tail onto a free link at the root.
>>> chain.reparent(5, 0, "X").max_weight
5
```

Moving a subtree is how a local search lowers the Pauli weight of a poorly shaped tree.

#### reroot(node)

Return a new tree rooted at `node`, reversing the links along the path to the old root.

Re-rooting at the current root is a no-op.

A reversed link has to occupy a slot that is free at its new parent. It keeps its old label
when that slot is free and otherwise takes the lowest free slot in `X`, `Y`, `Z` order,
which is deterministic. In a chain along `"Z"`, for instance, every interior node has
already spent its `Z` slot on the next node, so the `Z` link arriving from above must take
a different one. A consequence worth knowing: a label that had to change is not changed back,
so re-rooting is **not** in general an exact inverse of itself. Rooting away and back gives the
same shape, but not necessarily the same Pauli strings.

* **Parameters:**
  **node** – the node to make the root. It must have at most two children.
* **Returns:**
  A new tree.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `node` is out of range.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `node` already has three children.
  * [**OverflowError**](https://docs.python.org/3/builtins/exceptions.html#OverflowError) – if `node` is negative or does not fit in 32 bits.

#### NOTE
The at-most-two-children requirement is not a formality: re-rooting gives `node` a
fourth downward link (the one pointing back at its former parent), and a node has only
three. Re-rooting and [`relabel()`](#qiskit_fermions.mappers.TernaryTree.relabel) never change which nodes are joined, so a search
that should reach trees of a different shape also needs [`reparent()`](#qiskit_fermions.mappers.TernaryTree.reparent).
