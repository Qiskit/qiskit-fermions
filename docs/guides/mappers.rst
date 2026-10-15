.. _mappers_explanation:

Map operator representations
============================

**Mappers** are routines that transform operators from one representation to another
while preserving mathematical equivalence. Use them to work flexibly across
different operator algebras and to prepare operators for quantum circuit execution.

Implement custom mappers
------------------------

The mappers module makes it straightforward to implement custom
mappings. Mappers work by transforming the fundamental
actions (creation, annihilation, Majorana, and so on) that compose an operator, then
combining these transformed actions according to the target representation's rules.

As a concrete example, consider a Jordan-Wigner transformation that maps
a :class:`.MajoranaOperator` to a qubit operator using :func:`.map_majorana_action_generators`:

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit.quantum_info import SparseObservable
       >>> from qiskit_fermions.mappers import map_majorana_action_generators
       >>> from qiskit_fermions.operators import MajoranaOperator
       >>>
       >>> # Create a Majorana operator
       >>> maj_op = MajoranaOperator.from_dict({(0, 1): 1.0, (2,): 0.5})
       >>>
       >>> # Define how each Majorana action maps to Pauli strings
       >>> num_qubits = 2
       >>> def map_action(mode: int) -> SparseObservable:
       ...     idx = mode // 2
       ...     qubits = list(range(idx + 1))
       ...     pauli = "Y" if mode % 2 else "X"
       ...     return SparseObservable.from_sparse_list(
       ...         [("Z" * idx + pauli, qubits, 1.0)],
       ...         num_qubits=num_qubits,
       ...     )
       >>>
       >>> # Apply the mapping to transform the operator
       >>> custom_observable = map_majorana_action_generators(
       ...     maj_op,
       ...     map_action,
       ...     identity=lambda: SparseObservable.identity(num_qubits),
       ...     compose=SparseObservable.compose,
       ... )
       >>> print(custom_observable.simplify())
       <SparseObservable with 2 terms on 2 qubits: (0+1j)(Z_0) + (0.5+0j)(X_1 Z_0)>

    .. code-block:: c

       // The C API provides direct mapping functions for common transformations.
       // Custom mapper prototyping as shown in Python is primarily a Python feature.
       // For custom mappings in C, you will need to perform the iteration and
       // conversion logic entirely by yourself.

The key steps to implement any custom mapper are:

1. **Define action transformation**: Specify how each action (indexed by mode) maps
   to the target representation. In the example above, Majorana actions map to Pauli
   strings according to Jordan-Wigner rules.

2. **Use one of the provided mapping functions**: The :mod:`~qiskit_fermions.mappers`
   module provides utility functions to handle iterating over the operator terms and
   subsequent applications of the custom action map for the operator
   representations provided by the :mod:`~qiskit_fermions.operators` module. For example,
   :func:`.map_majorana_action_generators` works with Majorana operators.

3. **Get the result**: The mapping function returns an operator in the target
   representation with the transformation applied consistently across all terms.

This design makes it easy to prototype custom mappings without deep knowledge of
internal operator structures. The transformation is applied automatically to every
term in the operator while preserving mathematical equivalence.

.. hint::

   You can also iterate the terms of an operator manually rather than
   using one of the provided iterator functions. See the section on :ref:`term iteration
   <term_iteration_and_reconstruction>` in the operators guide for details.

Library implementations
-----------------------

The :mod:`qiskit_fermions.mappers.library` module provides efficient, thoroughly
tested implementations of common mappings. These follow the same underlying pattern
as custom mappers but are optimized for production use and are available in both the
Python and C APIs.

The example below applies a library mapper to the same :class:`.MajoranaOperator`
from the custom mapper section, demonstrating equivalent results:

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.mappers.library import majorana_to_fermion, jordan_wigner
       >>>
       >>> # Convert MajoranaOperator to FermionOperator using library mapper
       >>> ferm_op = majorana_to_fermion(maj_op)
       >>>
       >>> # Apply Jordan-Wigner mapper to get qubit operator
       >>> sparse_observable = jordan_wigner(ferm_op, num_qubits=2)
       >>> difference = sparse_observable - custom_observable
       >>> print(difference.simplify() == SparseObservable.zero(num_qubits))
       True

    .. code-block:: c

       #include <qiskit_fermions.h>

       // Create a MajoranaOperator with the same terms as maj_op above
       uint64_t num_terms = 2;
       uint64_t num_modes = 3;
       uint32_t modes[3] = {0, 1, 2};
       QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.5, 0.0}};
       uint32_t boundaries[3] = {0, 2, 3};
       QfMajoranaOperator *maj_op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

       // Convert MajoranaOperator to FermionOperator using library mapper
       QfFermionOperator *ferm_op = qf_majorana_to_fermion(maj_op);

       // Apply Jordan-Wigner mapper to get qubit operator
       QkObs *qubit_op;
       QfExitCode exit = qf_ferm_op_jordan_wigner(ferm_op, 2, &qubit_op);
       assert(exit == QfExitCode_Success);

       // Clean up
       qf_ferm_op_free(ferm_op);
       qf_maj_op_free(maj_op);
       qk_obs_free(qubit_op);

.. note::
   The two-step above is written out to show two library mappers composing. In practice you would map
   a :class:`.MajoranaOperator` in one step instead, with :func:`.majorana_jordan_wigner`
   (:c:func:`qf_maj_op_jordan_wigner` in the C API), and likewise for the other operator types --
   :func:`.edge_vertex_jordan_wigner` and :func:`.transfer_vertex_jordan_wigner`
   (:c:func:`qf_edge_op_jordan_wigner`, :c:func:`qf_transfer_op_jordan_wigner`). Going via a
   :class:`.FermionOperator` also inflates the intermediate result: every generator of these algebras
   maps onto a single Pauli string, whereas each fermionic action maps onto a two-term sum, so a term
   built from several generators expands only to be merged back down again. That saving grows with the
   length of the terms; for the single-generator terms of a hopping Hamiltonian the two routes cost
   about the same.

In the Python API, some of these library mappers are also reachable through type-agnostic
convenience mappers. :func:`.fermion_operator` and :func:`.majorana_operator` sit in front of
:func:`.majorana_to_fermion` and :func:`.fermion_to_majorana` (among others), delegating to the
appropriate library implementation internally based on the type of object they are given,
rather than requiring you to pick the right function yourself. The same applies to
:func:`.jordan_wigner` and to :func:`.ternary_tree` (introduced below), each of which dispatches to
whichever direct implementation matches the operator it is given. See
:mod:`qiskit_fermions.protocols` for how this dispatch is implemented internally.


Choose an encoding with ternary trees
-------------------------------------

Jordan-Wigner is not the only fermion-to-qubit encoding, and it is not the cheapest. Its
:math:`Z` strings grow with the mode index, so a Majorana operator on the last of :math:`n` modes maps
onto a Pauli string of weight :math:`n`. Encodings that spread the parity information across a tree
instead of a chain do asymptotically better.

A **ternary tree** on :math:`n` nodes captures this whole family. Each node is a qubit carrying three
downward links labelled ``X``, ``Y`` and ``Z``; a link either descends to a child node (an *edge*) or
terminates (a *leg*). Each leg's path back to the root spells a Pauli string: crossing the link
labelled :math:`P` that descends *from* node :math:`u` contributes :math:`P` on qubit :math:`u`. Every
such tree yields a valid encoding, and the familiar ones are simply particular shapes -- a chain along
``Z`` is Jordan-Wigner, while a tree of branching rate 3 is balanced.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
       >>>
       >>> # Jordan-Wigner is a chain descending along Z, with X and Y as legs at every node.
       >>> TernaryTree.chain(13, "Z").max_weight
       13
       >>> # The balanced tree spreads the same information over depth log_3(2n+1).
       >>> TernaryTree.breadth_first(13, 3).max_weight
       3

    .. code-block:: c

       #include <qiskit_fermions.h>

       QfTernaryTree *chain;
       qf_ternary_tree_chain(13, QfPauliLabel_Z, &chain);

       QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
       QfTernaryTree *balanced;
       qf_ternary_tree_breadth_first(13, 3, order, &balanced);

       assert(qf_ternary_tree_max_weight(chain) == 13);
       assert(qf_ternary_tree_max_weight(balanced) == 3);

       qf_ternary_tree_free(chain);
       qf_ternary_tree_free(balanced);

That weight is optimal: no fermion-to-qubit mapping of :math:`n` modes can do better than
:math:`\lceil \log_3(2n+1) \rceil` on average. Both trees use exactly :math:`n` qubits, so the saving
is in Pauli weight, not qubit count.

To map an operator, compile a tree into an encoding and hand that to a mapper along with the qubit
count for the result. Compiling pairs the tree's legs into Majorana operators and builds their Pauli
strings once, so the same encoding should be reused across many operators. The qubit count stays with
the mapping call rather than the encoding, so one encoding can serve registers of different widths; a
count above the tree's node count pads with the identity.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.mappers.library import fermion_ternary_tree
       >>> from qiskit_fermions.operators import FermionOperator
       >>>
       >>> hamiltonian = FermionOperator.from_dict(
       ...     {((True, 0), (False, 3)): 1.0, ((True, 3), (False, 0)): 1.0}
       ... )
       >>> encoding = TernaryTreeEncoding(TernaryTree.breadth_first(4, 3))
       >>> fermion_ternary_tree(hamiltonian, encoding, 4).simplify()
       <SparseObservable with 2 terms on 4 qubits: (0.5+0j)(Y_3 Z_1 Y_0) + (0.5+0j)(X_3 Z_2 X_0)>

    .. code-block:: c

       // Create the hopping Hamiltonian a^dagger_0 a_3 + a^dagger_3 a_0
       QfFermionOperator *hamiltonian = qf_ferm_op_zero();
       QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
       bool actions[4] = {true, false, true, false};
       uint32_t indices[4] = {0, 3, 3, 0};
       for (int i = 0; i < 2; i++) {
           qf_ferm_op_add_term(hamiltonian, 2, actions + 2 * i, indices + 2 * i, &coeffs[i]);
       }

       QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
       QfTernaryTree *tree;
       qf_ternary_tree_breadth_first(4, 3, order, &tree);

       // Compile the tree once, then reuse the encoding for every operator mapped through it.
       QfTernaryTreeEncoding *encoding;
       qf_ternary_tree_encoding_new(tree, NULL, &encoding);

       QkObs *qubit_op;
       QfExitCode exit = qf_ferm_op_ternary_tree(hamiltonian, encoding, 4, &qubit_op);
       assert(exit == QfExitCode_Success);

       // Clean up
       qf_ferm_op_free(hamiltonian);
       qf_ternary_tree_encoding_free(encoding);
       qf_ternary_tree_free(tree);
       qk_obs_free(qubit_op);

Mapping through a chain along ``Z`` reproduces :func:`.fermion_jordan_wigner`
(:c:func:`qf_ferm_op_jordan_wigner` in the C API) exactly, which is the sense in which this generalizes
it:

.. doctest::

    >>> from qiskit_fermions.mappers.library import fermion_jordan_wigner
    >>> chain = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
    >>> direct = fermion_jordan_wigner(hamiltonian, 4)
    >>> (fermion_ternary_tree(hamiltonian, chain, 4) - direct).simplify().num_terms
    0

Every operator representation has a ternary-tree mapper, mirroring the Jordan-Wigner set; see
:mod:`qiskit_fermions.mappers.library` for the full list, and :ref:`qf_mapper_library` in the C API.
In the Python API they are also reachable through :func:`.ternary_tree`, the type-agnostic wrapper
that picks the right one for the operator it is given, just as :func:`.jordan_wigner` does:

.. doctest::

    >>> from qiskit_fermions.mappers.library import ternary_tree
    >>> ternary_tree(hamiltonian, encoding, 4).simplify()
    <SparseObservable with 2 terms on 4 qubits: (0.5+0j)(Y_3 Z_1 Y_0) + (0.5+0j)(X_3 Z_2 X_0)>

.. note::
   Some identities that hold under Jordan-Wigner are properties of the *chain* rather than of ternary
   trees. The number operator is :math:`n_j = (1 - Z_j)/2` only there; on another tree it is a signed
   :math:`Z` string over a set of qubits. Likewise a vertex operator :math:`V_j` is the weight-1 Pauli
   :math:`Z_j` only on the chain, where the two Majorana strings' :math:`Z` chains cancel.

Build an arbitrary tree
~~~~~~~~~~~~~~~~~~~~~~~

The chain and breadth-first constructors cover the uniform families, but the general interface takes
one ``(parent, label)`` pair per node, with ``None`` for the root. This matters because the trees worth
searching for are usually neither a chain nor uniform -- the Bonsai algorithm, for instance, grows a
spanning tree of a device's coupling graph, so its branching follows that connectivity.

.. tab-set-code::

    .. code-block:: python

       >>> # Node 0 branches three ways, node 1 twice, and node 4 carries a single child.
       >>> tree = TernaryTree([None, (0, "X"), (0, "Y"), (0, "Z"), (1, "X"), (1, "Z"), (4, "Y")])
       >>> tree.num_nodes, tree.num_legs, tree.max_weight
       (7, 15, 4)
       >>> dict(tree.children(0))
       {'X': 1, 'Y': 2, 'Z': 3}

    .. code-block:: c

       // Node 0 branches three ways, node 1 twice, and node 4 carries a single child.
       QfTernaryTreeNode spec[7] = {
           {0, QfPauliLabel_Z, true},  // the root; parent and label are ignored
           {0, QfPauliLabel_X, false}, {0, QfPauliLabel_Y, false}, {0, QfPauliLabel_Z, false},
           {1, QfPauliLabel_X, false}, {1, QfPauliLabel_Z, false}, {4, QfPauliLabel_Y, false},
       };

       QfTernaryTree *tree;
       QfExitCode exit = qf_ternary_tree_new(7, spec, &tree);
       assert(exit == QfExitCode_Success);

       assert(qf_ternary_tree_num_nodes(tree) == 7);
       assert(qf_ternary_tree_num_legs(tree) == 15);
       assert(qf_ternary_tree_max_weight(tree) == 4);

       qf_ternary_tree_free(tree);

Note the leg count: it is always :math:`2n+1`, whatever the shape. A node with fewer children does not
generate fewer Pauli strings -- its unused links are legs, and a leg spells a string just as an edge's
descendants do. Of those :math:`2n+1` strings, :math:`2n` pair into the Majorana operators of the
:math:`n` modes, and exactly one is left over.

Total parity
~~~~~~~~~~~~

That leftover leg is the total fermionic parity :math:`\prod_j (1 - 2 n_j)`. It is a property of the
encoding rather than a mapping of any operator, so it is a method on the encoding itself:

.. tab-set-code::

    .. code-block:: python

       >>> TernaryTreeEncoding(TernaryTree.chain(4, "Z")).total_parity(4)
       <SparseObservable with 1 term on 4 qubits: (1+0j)(Z_3 Z_2 Z_1 Z_0)>
       >>> TernaryTreeEncoding(TernaryTree.breadth_first(4, 3)).total_parity(4)
       <SparseObservable with 1 term on 4 qubits: (1+0j)(Z_3 Z_0)>

    .. code-block:: c

       QfTernaryTree *tree;
       qf_ternary_tree_chain(4, QfPauliLabel_Z, &tree);

       QfTernaryTreeEncoding *encoding;
       qf_ternary_tree_encoding_new(tree, NULL, &encoding);

       // On the chain this is the full Z string; on the balanced tree it is not.
       QkObs *parity = qf_ternary_tree_encoding_total_parity(encoding, 4);

       qf_ternary_tree_encoding_free(encoding);
       qf_ternary_tree_free(tree);
       qk_obs_free(parity);

As the balanced case shows, this is *not* in general the all-:math:`Z` string: it is the leg reached by
the all-:math:`Z` *path*, which acts as the identity on every node not on that path.

.. important::
   Total parity here is an **observable to measure, not a constraint to impose**. Both of its
   eigenvalues are physical, so do not project a state onto the :math:`+1` sector: a ternary-tree
   encoding spends one qubit per mode and is onto the full :math:`2^n`-dimensional space, leaving no
   unphysical subspace to project away.

   Local encodings that add ancilla qubits, such as the flow sets in
   :ref:`1D <1d_fermi_hubbard>` and :ref:`2D <2d_fermi_hubbard>`, do work that way: there
   :math:`n_\text{qubits} > n_\text{modes}`, the extra dimensions are unphysical, and a stabilizer
   subspace is what makes the encoding faithful. Ternary trees add no such qubits and have no such
   subspace.


.. _bravyi_kitaev_by_hand:

Worked example: a Bravyi-Kitaev tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifications are more often computed than typed out, since the tree for a real encoding is rarely small
enough to write by hand. Bravyi-Kitaev is a good example: it belongs to this family, but its tree is not
uniform, so no convenience constructor covers it.

Its tree is built by a recursive split over the nodes, one per mode. Within a range of nodes
:math:`[l, r]`, let :math:`h` be the largest power of two that fits; the node placed for that range is
:math:`l + h - 1`, the last one in that block. Its ``X`` link descends to the tree built from the block
before it and its ``Z`` link to the tree built from the remainder after it, leaving its ``Y`` link as a
leg:

.. doctest::

    >>> def bravyi_kitaev_tree(num_modes: int) -> TernaryTree:
    ...     """Builds the ternary tree of the Bravyi-Kitaev encoding on ``num_modes`` modes."""
    ...     spec: list[tuple[int, str] | None] = [None] * num_modes
    ...     def build(left: int, right: int) -> int | None:
    ...         if left > right:
    ...             return None  # an empty range: the parent's link stays a leg
    ...         block = 1
    ...         while 2 * block <= right - left + 1:
    ...             block *= 2
    ...         node = left + block - 1
    ...         for child, label in (
    ...             (build(left, node - 1), "X"),   # the block before this node
    ...             (build(node + 1, right), "Z"),  # the remainder after it
    ...         ):
    ...             if child is not None:
    ...                 spec[child] = (node, label)
    ...         return node
    ...     build(0, num_modes - 1)
    ...     return TernaryTree(spec)

At four modes the whole range is a single block, so the node placed first is 3, with :math:`[0, 2]`
before it and nothing after; that sub-range gives node 1 over the single nodes 0 and 2. Compiling the tree
confirms the encoding is Bravyi-Kitaev and not merely a tree of the same shape class -- its first Majorana
image is :math:`X_0 X_1 X_3`:

.. doctest::

    >>> tree = bravyi_kitaev_tree(4)
    >>> dict(tree.children(1))
    {'X': 0, 'Z': 2}
    >>> encoding = TernaryTreeEncoding(tree)
    >>> encoding.majorana_image(0)
    [(0, 'X'), (1, 'X'), (3, 'X')]

Its total parity makes the point of the previous section concrete: the leftover leg is the one reached
by the all-:math:`Z` path, and here that path stops at the root, so the observable is the single-qubit
:math:`Z_3` rather than a string over all four qubits:

.. doctest::

    >>> encoding.total_parity(4)
    <SparseObservable with 1 term on 4 qubits: (1+0j)(Z_3)>

Scaling up needs nothing but a larger argument. Because no node ever spends its ``Y`` link on an edge,
each branches at most two ways, so the Pauli weight grows logarithmically rather than linearly:

.. doctest::

    >>> [bravyi_kitaev_tree(n).max_weight for n in (4, 8, 16, 32, 64)]
    [3, 4, 5, 6, 7]
